import hydra
import torch
import warnings

from lightning import seed_everything
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base="1.3", config_path="./configs", config_name="train_lnl")
def train_lnl(cfg: DictConfig) -> None:
    
    # Force OmegaConf to check missing values
    # i.e. those declared as '???' in config_lnl.yaml but not supplied)
    _ = OmegaConf.to_container(cfg, throw_on_missing=True)

    # Resolve computed values, e.g.
    # i.e. ${multiply:${lnl.val_every_n_updates},${trainer.accumulate_grad_batches}}
    OmegaConf.register_new_resolver("multiply", lambda x, y: x*y)
    OmegaConf.resolve(cfg)

    print(cfg)

    seed_everything(cfg.lnl.seed)

    # Ignore non-relevant warnings from Lightning
    warnings.filterwarnings("ignore", message="The `srun` command is available on your system but is not used")
    warnings.filterwarnings("ignore", message="Starting from v1.9.0, `tensorboardX` has been removed as a dependency")

    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        ampere_or_later = major >= 8
        if ampere_or_later:
            # See https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
            torch.set_float32_matmul_precision("high")

    model = hydra.utils.instantiate(cfg.model)
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    trainer = hydra.utils.instantiate(cfg.trainer)
    
    trainer.fit(model, datamodule)

if __name__ == "__main__":
    train_lnl()
