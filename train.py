from datetime import datetime
from typing import Optional

from pytorch_lightning import (
    Trainer, 
    seed_everything
)
from pytorch_lightning.loggers import WandbLogger, MLFlowLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor


from dataset import SpeechDataModule
from model import SpeechModel
import argparse
import gc
import yaml
import wandb
import warnings
import mlflow
import mlflow.pytorch

warnings.filterwarnings(action='ignore')

wandb.login()

def parse_args():
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--config_path", type=str, default="configs/config.yaml",help="config dir")
    args = parser.parse_args()
    return args

def read_config(config_path):
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

def main(config):
    gc.collect()
    seed_everything(config['seed'])
    artifacts_path = 'model'
    mlflow.set_tracking_uri('http://localhost:5002')

    wandb_logger = WandbLogger(name=config['Task'],
                               project=config['Experiment'],
                               log_model='all')
    mlf_logger = MLFlowLogger(experiment_name="speech-style-convert", tracking_uri="http://localhost:5002")

    lr_callback = LearningRateMonitor(logging_interval='step')
    early_stop_callback = EarlyStopping(monitor=config['callback']['monitor'], 
                                        min_delta=0.00, 
                                        patience=3, 
                                        verbose=False, 
                                        mode="max")
    checkpoint_callback = ModelCheckpoint(
                    monitor = config['callback']['monitor'],
                    dirpath = config['callback']['dirpath'],
                    save_top_k = config['callback']['save_top_k'],
                    filename= config['Task'] + config['Experiment'] + '-{epoch}')

    dm = SpeechDataModule(config)
    dm.setup("fit")
    
    model = SpeechModel(config)

    trainer = Trainer(
        max_epochs=config['trainer']['max_epochs'],
        accelerator=config['trainer']['accelerator'],
        logger=[wandb_logger, mlf_logger],
        devices=1,
        callbacks=[
            checkpoint_callback,
            lr_callback,
            early_stop_callback
        ]
    )

    with mlflow.start_run() as run:
        mlflow.log_param('epoch', config['trainer']['max_epochs'])
        mlflow.log_param('batch_size', config['data']['batch_size'])

        trainer.fit(model, datamodule=dm)

        mlflow.pytorch.log_model(trainer, artifact_path = artifacts_path)
        trainer.validate(model, dm)

        mlflow.log_metric('sacrebleu', eval)


if __name__ == "__main__":
    args = parse_args()
    config = read_config(args.config_path)
    main(config)