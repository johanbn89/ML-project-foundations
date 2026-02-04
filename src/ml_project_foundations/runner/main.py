import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import sys

import hydra
import lightning.pytorch as pl
import mlflow
import mlflow.pytorch
from hydra.utils import instantiate
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import OmegaConf


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg):
    print("Starting training with PyTorch Lightning")
    print("Configuration:\n", OmegaConf.to_yaml(cfg, resolve=True))

    model = instantiate(cfg.model)
    datamodule = instantiate(cfg.data)

    logger = MLFlowLogger(
        experiment_name=cfg.logger.experiment_name,
        tracking_uri=cfg.logger.tracking_uri,
        run_name=cfg.logger.run_name,
        log_model=cfg.logger.log_model,
    )
    mlflow.set_tracking_uri(cfg.logger.tracking_uri)

    callbacks = [instantiate(c) for c in cfg.trainer.callbacks]
    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer_kwargs.pop("callbacks", None)
    trainer = pl.Trainer(**trainer_kwargs, logger=logger, callbacks=callbacks)

    # Provenance
    BASE_CMD = "uv run python -m ml_project_foundations.runner.main"
    cmd = " ".join([BASE_CMD, *sys.argv[1:]]).strip()
    client = logger.experiment
    run_id = logger.run_id
    client.log_text(run_id, cmd + "\n", "provenance/run_command.txt")
    client.log_text(run_id, OmegaConf.to_yaml(cfg, resolve=True), "provenance/resolved_config.yaml")

    # Resume
    ckpt_path = None
    if cfg.run.ckpt:
        source_run_id = cfg.run.ckpt
        which = cfg.run.resume_which  # "last" or "epoch-epoch=3 etc."
        artifact_uri = f"runs:/{source_run_id}/model/checkpoints/{which}/{which}.ckpt"
        ckpt_path = mlflow.artifacts.download_artifacts(artifact_uri)

    # Train
    if "fit" in cfg.run.stages:
        trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

        # Log a registry-ready MLflow model package (separate from Lightning checkpoints)
        with mlflow.start_run(run_id=logger.run_id):
            mlflow.pytorch.log_model(model, artifact_path="mlflow_model")


if __name__ == "__main__":
    main()
