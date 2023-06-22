# lnl-examples

<p align="center"><img width="500" src="https://user-images.githubusercontent.com/9938298/244146091-1e3cf317-910a-4fcf-a0e2-6e755a4935c0.png"></p>

This repository contains a sequence of examples for using [PyTorch Lightning](https://github.com/Lightning-AI/lightning) and [Lhotse](https://github.com/lhotse-speech/lhotse), two Python libraries that help with a lot of the necessary plumbing that can otherwise be tricky/time-consuming to take care of yourself (e.g. automatic mixed precision, dynamic batching, etc.).

The examples are split between Google Colab notebooks in `notebooks` (e.g. `notebooks/01_mwe.ipynb`) and Python script in the root directory (e.g. `train_mwe.py`).
The notebooks contain a lot of prose aimed at newcomers and builds up a component-by-component introduction towards a given Python script (e.g. everything you need to understand `train_mwe.py` is found in `01_mwe.ipynb`).

## Usage

### Environment

We'll be using the latest stable versions of various packages as of early June 2023:

```
# For minimal working example in train_mwe.py
torch==2.0.1
torchaudio==2.02
lightning==2.0.2
lhotse==1.14.0

# For extras in train_mwe-extras.py and trainlnl.py
jiwer==3.0.2
pandas==2.0.2
hydra-core==1.3.2
wandb==0.15.4
torchinfo==1.8.0
```

#### Conda

You can create and activate a conda environment at in the `env` directory from the `env.yaml` file using:

```bash
conda env create --prefix ./env --file env.yaml
conda activate ./env
```

#### Docker

If you have docker and docker-compose you can also launch a container using:

```bash
docker compose run lnl-examples
```

### Run examples

#### Minimal working examples

Once you've set up this environment (and optionally read the corresponding notebook), you can run a given example as:

```bash
python train_mwe.py

# or extras

python train_mwe-extras.py
```

#### LnL example

```bash
# To run without Weights & Biases configured
wandb disabled
python train_lnl.py \
    lnl.wnb_project=null \
    lnl.wnb_run=null

# To run with Weights & Biases configured
wandb enabled
python train_lnl.py \
    lnl.wnb_project=my_project \
    lnl.wnb_run='test run train_lnl.py'

# Use Hydra CLI overrides to change/add configs (see defaults in configs folder):
python train_lnl.py \
    lnl.wnb_project=my_project \
    lnl.wnb_run='test run train_lnl.py' \
    trainer.precision='16-mixed' \
    +trainer.gradient_clip_val=0.5
```
