"""
Should we really have mode branching logic in main.py?
Could just separate into  branching_flags.py or similar.
"""

import warnings

import hydra
import lightning.pytorch as pl
from hydra.utils import instantiate
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import OmegaConf

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg):
    print("Starting training with PyTorch Lightning")
    print("Configuration:\n", OmegaConf.to_yaml(cfg))
    # Think about this later
    # seed_everything(cfg.seed)

    model = instantiate(cfg.model)
    datamodule = instantiate(cfg.data)

    # This should also be instantiate(cfg.logger)?
    # Yes, but we assume only MLFlow for now.
    if cfg.logger.name == "mlflow":
        logger = MLFlowLogger(
            experiment_name=cfg.logger.experiment_name,
            tracking_uri=cfg.logger.tracking_uri,  # Should this be env derived?
            run_id=cfg.logger.run_name,
        )

    trainer = pl.Trainer(
        **cfg.trainer,
        logger=logger,
    )

    if "fit" in cfg.run.stages:
        trainer.fit(model, datamodule=datamodule)

    if "eval" in cfg.run.stages:
        trainer.validate(
            model,
            datamodule=datamodule,
            ckpt_path=cfg.run.ckpt_path,
        )

    if "test" in cfg.run.stages:
        trainer.test(
            model,
            datamodule=datamodule,
            ckpt_path=cfg.run.ckpt_path,
        )

    if "predict" in cfg.run.stages:
        trainer.predict(
            model,
            datamodule=datamodule,
            ckpt_path=cfg.run.ckpt_path,
        )


if __name__ == "__main__":
    main()
