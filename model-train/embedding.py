import argparse, random, os, math, json, pathlib
from pathlib import Path
from typing import List
import subprocess

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import faiss

SEED = 7777
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--csv", default="feedback-bank.csv")
parser.add_argument("--model_id", default="meta-llama/Meta-Llama-3-8B") 
parser.add_argument("--checkpoint", type=str, default="result/epoch_8/model_checkpoint.pth")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch", type=int, default=8)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--temperature", type=float, default=0.07)
parser.add_argument("--output", default="hall-train")
args = parser.parse_args()

print(f"Loading dataset: {args.csv}")
df = pd.read_csv(args.csv).sample(frac=1)

required_cols = [
    "Tweet", "Predicted Reasoning", "Predicted Sentiment", "Final Feedback"
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"CSV missing required columns: {missing}")

class FeedbackDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.reset_index(drop=True)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        anchor = (
            f"A large language model was tasked with identifying sentiment (positive, negative, or neutral) of COVID-19 related tweets.\n\n"
            f"The tweet was:\n{row['Tweet']}\n\n"
            f"The response was:\nPredicted Sentiment: {row['Predicted Sentiment']}\n"
            f"Predicted Reasoning: {row['Predicted Reasoning']}\n\n"
            f"Please write a feedback for the model in one word _"
        )
        positive = row["Final Feedback"]
        return anchor, positive

train_ds = FeedbackDataset(df)
print(f"Dataset size: {len(train_ds):,}")

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    args.model_id,
    quantization_config=bnb,
    output_hidden_states=True,
    use_cache=False,
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(
    model,
    LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj","k_proj","v_proj","o_proj",
            "gate_proj","up_proj","down_proj","lm_head",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    ),
)

if args.checkpoint:
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    key = "model_state_dict" if "model_state_dict" in ckpt else None
    model.load_state_dict(ckpt[key] if key else ckpt, strict=False)
    print("loaded")

model.train().to(device)

tokenizer = AutoTokenizer.from_pretrained(args.model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

class InfoNCELoss(nn.Module):
    def __init__(self, T: float):
        super().__init__(); self.T = T
    def forward(self, anchors: torch.Tensor, positives: torch.Tensor):
        anchors = F.normalize(anchors, dim=-1)
        positives = F.normalize(positives, dim=-1)
        logits = anchors @ positives.T
        labels = torch.arange(anchors.size(0), device=anchors.device)
        return F.cross_entropy(logits / self.T, labels)

criterion = InfoNCELoss(args.temperature)
optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

def collate(batch):
    anchors, positives = zip(*batch)
    token_a = tokenizer(list(anchors), padding=True, return_tensors="pt")
    token_p = tokenizer(list(positives), padding=True, return_tensors="pt")
    return token_a.to(device), token_p.to(device)

dataloader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate)

for epoch in range(1, args.epochs + 1):
    total_loss, steps = 0.0, 0
    for token_a, token_p in dataloader:
        if steps % 6 == 0:
            print(f"Step: {steps}")
        optim.zero_grad()
        out_a = model(**token_a)
        out_p = model(**token_p)
        emb_a = out_a.hidden_states[-1][:, -1, :]
        emb_p = out_p.hidden_states[-1][:, -1, :]
        loss = criterion(emb_a, emb_p)
        loss.backward()
        optim.step()
        total_loss += loss.item()
        steps += 1

    print(f"Epoch {epoch} completed. Loss: {total_loss/steps:.4f}")

    result_dir = f"result/epoch_{epoch}"
    os.makedirs(result_dir, exist_ok=True)

    checkpoint_path = os.path.join(result_dir, "model_checkpoint.pth")
    torch.save(model.state_dict(), checkpoint_path)
    tokenizer.save_pretrained(result_dir)
    print(f"Saved model and tokenizer to {result_dir}")
