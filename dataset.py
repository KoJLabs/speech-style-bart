from datetime import datetime
from typing import Optional

import datasets
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    AutoConfig,
    DataCollatorForSeq2Seq,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup,
)

MAP ={
    'formal': '문어체',
    'informal': '구어체',
    'android': '안드로이드',
    'azae': '아재',
    'chat': '채팅',
    'choding': '초등학생',
    'emoticon': '이모티콘',
    'enfp': 'enfp',
    'gentle': '신사',
    'halbae': '할아버지',
    'halmae': '할머니',
    'joongding': '중학생',
    'king': '왕',
    'naruto': '나루토',
    'seonbi': '선비',
    'sosim': '소심한',
    'translator': '번역기'
}

class SpeechDataModule(LightningDataModule):
    def __init__(
            self,
            config,
            **kwargs,
            ):
        super().__init__()
        self.model_name_or_path = config['model']['text_model_path']
        self.max_seq_length = config['data']['max_length']
        self.train_batch_size = config['data']['batch_size']
        self.eval_batch_size = config['data']['batch_size']
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
        self.collate_fn = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

    def setup(self, stage: str):
        self.dataset = datasets.load_dataset("KoJLabs/speech")

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=False,
                remove_columns=self.dataset[split].column_names,
            )

    def prepare_data(self):
        datasets.load_dataset("KoJLabs/speech")
        AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def train_dataloader(self):
        return DataLoader(
            self.dataset['train'], 
            batch_size=self.train_batch_size,
            shuffle=True,
            collate_fn=self.collate_fn
            )
    
    def val_dataloader(self):
        return DataLoader(
            self.dataset['test'], 
            batch_size=self.train_batch_size, 
            shuffle=True,
            collate_fn=self.collate_fn
            )
    
    def test_dataloader(self):
        return DataLoader(
            self.dataset['test'], 
            batch_size=self.train_batch_size, 
            shuffle=True,
            collate_fn=self.collate_fn
            )
    
    def convert_to_features(self, example_batch, indices=None):
        map_style = MAP[example_batch['target_style']]
        source_text = f"{map_style} 형식으로 변환:" + example_batch['source']

        target_text = example_batch['target'] + self.tokenizer.eos_token

        features = self.tokenizer(
            source_text,
            text_target=target_text,
            max_length=self.max_seq_length,
            truncation=True,
            padding='max_length'
        )

        del features['token_type_ids']

        return features
