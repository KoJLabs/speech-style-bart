from datetime import datetime
from typing import Any, Optional

import datasets
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_cosine_with_hard_restarts_schedule_with_warmup,
)
from dataset import SpeechDataModule
import argparse
import gc
import yaml
import evaluate
import numpy as np

class SpeechModel(LightningModule):
    def __init__(
            self, 
            config: str,
            eval_splits: Optional[list] = None,
            **kwargs,
            ):
        super().__init__()
        self.loss = [] ## pytorch lightning version update
        self.logits = []
        self.labels = []
        self.save_hyperparameters()
        self.model_config = AutoConfig.from_pretrained(config['model']['text_mode_path'])
        self.tokenizer = AutoTokenizer.from_pretrained(config['model']['text_mode_path'])
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config['model']['text_mode_path'], config=self.model_config)
        self.metric = evaluate.load("sacrebleu")

    def forward(self, **inputs):
        return self.model(**inputs)
    
    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]

        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]
        self.loss.append(val_loss)
        self.logits.append(logits.argmax(dim=-1).detach().cpu())
        self.labels.append(batch["labels"].detach().cpu())

        return {'loss': val_loss, 'preds': logits, 'labels': batch['labels']}
    
    def on_validation_epoch_end(self):  
        preds = torch.cat(self.logits, dim=0)
        labels = torch.cat(self.labels)
        loss = torch.stack(self.loss).mean()

        decoded_preds = self.tokenizer.batch_decode(
            preds, skip_special_tokens=True
            )

        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(
            labels, skip_special_tokens=True
            )

        self.log("val_loss", loss.item(), prog_bar=True)

        scores = self.metric.compute(
            predictions=decoded_preds,
            references=decoded_labels
            )
                
        self.log("bleu score", scores['score'], prog_bar=True)
        # 단순 후처리
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]

        self.logits.clear()
        self.loss.clear()
        self.labels.clear()

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]