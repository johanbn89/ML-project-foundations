import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import sys

import hydra
import lightning.pytorch as pl
from hydra.utils import instantiate
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import OmegaConf


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg):
    print("Starting training with PyTorch Lightning")
    print("Configuration:\n", OmegaConf.to_yaml(cfg, resolve=True))
    # Think about this later
    # seed_everything(cfg.seed)

    model = instantiate(cfg.model)
    datamodule = instantiate(cfg.data)

    # This should also be instantiate(cfg.logger)?
    # Yes, but we assume only MLFlow for now.
    if cfg.logger.name == "mlflow":
        logger = MLFlowLogger(
            experiment_name=cfg.logger.experiment_name,
            tracking_uri=cfg.logger.tracking_uri,
            run_name=cfg.logger.run_name,
            log_model=cfg.logger.log_model,
        )

    # Reconsider this
    callbacks = [instantiate(c) for c in cfg.trainer.callbacks]
    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer_kwargs.pop("callbacks", None)
    trainer = pl.Trainer(
        **trainer_kwargs,
        logger=logger,
        callbacks=callbacks,
    )

    # Log launch traces
    BASE_CMD = "uv run python -m ml_project_foundations.runner.main"
    cmd = " ".join([BASE_CMD, *sys.argv[1:]]).strip()
    client = logger.experiment
    run_id = logger.run_id
    client.log_text(run_id, cmd + "\n", "provenance/run_command.txt")
    client.log_text(run_id, OmegaConf.to_yaml(cfg, resolve=True), "provenance/resolved_config.yaml")

    if "fit" in cfg.run.stages:
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.run.ckpt_path)

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
