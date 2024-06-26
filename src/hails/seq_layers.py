# Code adapted from: https://github.com/cure-lab/LTSF-Linear
import math

import torch
import torch.nn as nn


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x: torch.Tensor):
        moving_mean: torch.Tensor = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinear(nn.Module):
    """
    Decomposition-Linear
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        individual: bool = True,
        dim_out: int = 1,
    ):
        super(DLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.dim_out = dim_out

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = individual
        self.channels = enc_in

        if self.individual:
            fan_in = self.seq_len
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            self.Linear_Seasonal = nn.Parameter(
                torch.randn(self.channels, self.seq_len, self.pred_len * self.dim_out)
                * bound
            )
            self.Linear_Trend = nn.Parameter(
                torch.randn(self.channels, self.seq_len, self.pred_len * self.dim_out)
                * bound
            )

        # Use this two lines if you want to visualize the weights
        # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len * self.dim_out)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len * self.dim_out)

            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x: torch.Tensor):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = (
            seasonal_init.permute(0, 2, 1),
            trend_init.permute(0, 2, 1),
        )  # to [Batch, Channel, Input length]
        if self.individual:
            seasonal_output = torch.einsum(
                "bcl,clp->bcp", seasonal_init, self.Linear_Seasonal
            )
            trend_output = torch.einsum("bcl,clp->bcp", trend_init, self.Linear_Trend)
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)  # to [Batch, Output length, Channel]


class NLinear(nn.Module):
    """
    Normalization-Linear
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        individual: bool = True,
        dim_out: int = 1,
    ):
        super(NLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.dim_out = dim_out
        self.individual = individual
        self.channels = enc_in
        if self.individual:
            fan_in = self.seq_len
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            self.Linear = nn.Parameter(
                torch.randn(self.channels, self.seq_len, self.pred_len * self.dim_out)
                * bound
            )
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len * self.dim_out)

    def forward(self, x: torch.Tensor):
        # x: [Batch, Input length, Channel]
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        if self.individual:
            x = torch.einsum("blc,clp->bpc", x, self.Linear)
        else:
            x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + seq_last
        return x  # [Batch, Output length, Channel]

    def forward_for(self, x: torch.Tensor):
        # x: [Batch, Input length, Channel]
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        if self.individual:
            output = torch.zeros(
                [x.size(0), self.pred_len * self.dim_out, x.size(2)], dtype=x.dtype
            ).to(x.device)
            for i in range(self.channels):
                output[:, :, i] = self.Linear[i](x[:, :, i])
            x = output
        else:
            x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + seq_last
        return x  # [Batch, Output length, Channel]
