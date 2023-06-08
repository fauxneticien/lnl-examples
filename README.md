# lnl-examples

<p align="center"><img width="500" src="https://user-images.githubusercontent.com/9938298/244146091-1e3cf317-910a-4fcf-a0e2-6e755a4935c0.png"></p>

This repository contains a sequence of examples for using [PyTorch Lightning](https://github.com/Lightning-AI/lightning) and [Lhotse](https://github.com/lhotse-speech/lhotse), two Python libraries that help with a lot of the necessary plumbing that can otherwise be tricky/time-consuming to take care of yourself (e.g. automatic mixed precision, dynamic batching, etc.).

The examples are split between Google Colab notebooks in `notebooks` (e.g. `notebooks/01_mwe.ipynb`) and Python script in the root directory (e.g. `train_mwe.py`).
The notebooks contain a lot of prose aimed at newcomers and builds up a component-by-component introduction towards a given Python script (e.g. everything you need to understand `train_mwe.py` is found in `01_mwe.ipynb`).

## Usage

### Environment

We'll be using the latest stable versions of various packages as of early June 2023:

```
torch==2.0.1
torchaudio==2.02
lightning==2.0.2
lhotse==1.14.0
```

#### Conda

You can create and activate a conda environment from the `env.yaml` file using:

```bash
conda env create -n lnl-examples --file env.yaml
conda activate lnl-examples
```

### Run example

Once you've set up this environment (and optionally read the corresponding notebook), you can run a given example as:

```
python train_mwe.py
```
