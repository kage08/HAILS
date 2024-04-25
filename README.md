# Hails

## Instructions

### Requirements:

- Python >=3.11
- [Rye](https://rye-up.com/guide/installation/)

### Installation

1. Run `./unzip_dow.sh`
2. Run `rye sync` to download the required dependencies
3. Run `rye run python pretraindow.py` for pre-training (atleast for 10 epochs)
4. Run `rye run python traindow.py` for training (atleast for 100 epochs)
  - Requires atleast 70 GB VRAM with Mixed precision
  - Toggle `SCALE_PREC = False` in `traindow.py` to use FP16 to run on GPUs of less than 40 GB VRAM
  - Mention the prediction length in Line 13 of `traindow.py` (default is 12)
