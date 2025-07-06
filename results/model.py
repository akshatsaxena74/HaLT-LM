import os, sys, subprocess, random, argparse, shutil, tempfile
from pathlib import Path
from typing import List, Tuple

import faiss
import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from openai import OpenAI
from dotenv import load_dotenv

SEED = 7777
random.seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

load_dotenv("key.env")
open_api = os.getenv("open_api")
client = OpenAI(api_key=open_api)

def load_encoder(model_dir):
    model_id = "meta-llama/Meta-Llama-3-8B"

    print("Building 4-bit LoRA model …")
    
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
    
    if model_dir:
        print(f"Loading checkpoint: {model_dir}")
        ckpt = torch.load(model_dir, map_location="cpu")
        key = "model_state_dict" if "model_state_dict" in ckpt else None
        model.load_state_dict(ckpt[key] if key else ckpt, strict=False)
        print("loaded")
    
    model.eval().to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def embed_anchor(tweet, pred_sent, pred_reason, model, tokenizer):
    anchor_prompt = (
        f"A large language model was tasked with identifying sentiment (positive, negative, or neutral) of COVID-19 related tweets.\n\n"
        f"The tweet was:\n{tweet}\n\n"
        f"The response was:\nPredicted Sentiment: {pred_sent}\nPredicted Reasoning: {pred_reason}\n\n"
        f"Please write a feedback for the model in one word _"
    )
    toks = tokenizer([anchor_prompt], padding=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**toks)
        emb = F.normalize(out.hidden_states[-1][:, -1, :], dim=-1)
    return emb.cpu().numpy()

def retrieve_feedback(emb, k=1):
    index = faiss.read_index("feedback_vector_database.index")
    feedback_texts: List[str] = torch.load("feedback_texts.pth")
    _, I = index.search(emb, k)
    return [feedback_texts[i] for i in I[0]]

sys_prompt = (
    "You are a sentiment-analysis expert. You will be provided with a tweet and must classify its sentiment."
)

def prompt_template(tweet) -> str:
    return f"""
                You are a sentiment-analysis expert tasked with analysing sentiment of the given tweet.
                Please identify the sentiment (positive, negative, neutral) of the following tweet:
                \"{tweet}\"
                Your answer must follow this exact format:
                Reasoning: <Explain your step-by-step reasoning here>
                Sentiment: <One of: Positive, Negative, Neutral>
            """

def call_gpt(prompt):
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    content = resp.choices[0].message.content.strip()
    reasoning, sentiment = content.split("Sentiment:")
    reasoning = reasoning.replace("Reasoning:", "").strip()
    sentiment = sentiment.strip()
    return reasoning, sentiment

def evaluate(csv_path, model_dir, output_dir):
    model, tokenizer = load_encoder(model_dir)
    df_eval = pd.read_csv(csv_path).sample(frac=1).reset_index(drop=True)
    results = []
    base_ok, fb_ok = 0, 0

    for i, row in df_eval.iterrows():
        tweet, gold = row["Tweet"], row["Actual Sentiment"].lower()
        base_prompt = prompt_template(tweet)
        base_reason, base_sent = row["Predicted Reasoning"], row["Predicted Sentiment"]
        fb = "unk"
        fb_reason, fb_sent = base_reason, base_sent

        if base_sent.lower() == gold:
            base_ok += 1

        if row['Hallucination'] == 0:
            emb = embed_anchor(tweet, base_sent, base_reason, model, tokenizer)
            fb = retrieve_feedback(emb, k=1)[0]
            fb_prompt = base_prompt + "\nYour previous response:\n\nPredicted Sentiment:{base_sent}\nPredicted Reasoning:{base_reason}\n\nOur critique model thinks that the reasoning and predictions are hallucinated. Based on the feedback below, please reconsider your answer." + f"\n### Retrieved feedback\n{fb}\n"
            fb_reason, fb_sent = call_gpt(fb_prompt)

        if fb_sent.lower() == gold:
            fb_ok += 1

        print(
            f"[{i+1}/{len(df_eval)}] baseline={base_ok/(i+1):.3f}  with_fb={fb_ok/(i+1):.3f}",
            end="\r",
            flush=True
        )
        results.append({
            "Tweet": tweet,
            "Gold": gold,
            "Base_Sentiment": base_sent,
            "Base_Reasoning": base_reason,
            "Feedback": fb,
            "Feedback_Sentiment": fb_sent,
            "Feedback_Reasoning": fb_reason
        })

    results_df = pd.DataFrame(results)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "evaluation_results.csv")
    results_df.to_csv(output_file, index=False)
    print("\nDone. Baseline acc:", base_ok/len(df_eval), "Feedback acc:", fb_ok/len(df_eval))

if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--csv", default="df_eval-part-2.csv", help="Evaluation CSV")
    pa.add_argument("--model_dir", default="result/epoch_8/model_checkpoint.pth", help="Path to model directory")
    pa.add_argument("--output_dir", default="result/epoch_8", help="Where to save evaluation results")
    args = pa.parse_args()
    evaluate(args.csv, args.model_dir, args.output_dir)