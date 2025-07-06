import argparse
import random
from typing import List

import torch
import torch.nn.functional as F
import faiss
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

device = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 7777
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

pa = argparse.ArgumentParser()
pa.add_argument("--csv", default="feedback-bank.csv")
pa.add_argument("--checkpoint_dir", default="result/epoch_8/model_checkpoint.pth", help="folder with PEFT weights + tokenizer")
pa.add_argument("--batch", type=int, default=35)
pa.add_argument("--index_out", default="result/epoch_8/feedback_vector_database.index")
pa.add_argument("--texts_out", default="feedback_texts.pth")
args = pa.parse_args()

df = pd.read_csv(args.csv)
print(f"Total rows in CSV: {len(df)}")
print(args.csv)

if "Final Feedback" not in df.columns:
    raise ValueError("CSV must contain a ‘Final Feedback’ column.")

feedback_texts = df["Final Feedback"].astype(str).tolist()
print(f"Loaded {len(feedback_texts):,} feedback sentences from {args.csv}")

model_id = "meta-llama/Meta-Llama-3-8B"

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
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

if args.checkpoint_dir:
    print(f"Loading checkpoint → {args.checkpoint_dir}")
    ckpt = torch.load(args.checkpoint_dir, map_location="cpu")
    key = "model_state_dict" if "model_state_dict" in ckpt else None
    model.load_state_dict(ckpt[key] if key else ckpt, strict=False)
    print("loaded")

model.eval().to(device)


tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Starting encoding feedback texts")
embeds = []
with torch.no_grad():
    for i in range(0, len(feedback_texts), args.batch):
        print(f"Processing batch starting at index {i}")
        batch = feedback_texts[i : i + args.batch]
        toks = tokenizer(batch, padding=True, return_tensors="pt").to(device)
        outs = model(**toks)
        hid = outs.hidden_states[-1][:, -1, :]
        embeds.append(F.normalize(hid, dim=-1).cpu())

emb_arr = torch.cat(embeds, dim=0).numpy()
print("Vector matrix shape:", emb_arr.shape)

index = faiss.IndexFlatIP(emb_arr.shape[1])
index.add(emb_arr)
faiss.write_index(index, args.index_out)
print(f"Wrote FAISS index: {args.index_out}")

torch.save(feedback_texts, args.texts_out)
print(f"Wrote feedback texts: {args.texts_out}")
