from dotenv import load_dotenv
import os
from openai import OpenAI
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import random
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import logging
from bitsandbytes.optim import Adam8bit
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from huggingface_hub import login
import math
random.seed(7777)

load_dotenv("key.env")
open_api = os.getenv("open_api")
hf_token = os.getenv("hf_key")
login(token=hf_token)
client = OpenAI(api_key = open_api)
device = "cuda"

model_id = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    output_hidden_states=True,
    use_cache=False,
    device_map="auto"
)

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj", "lm_head"
    ],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
tokenizer.pad_token = tokenizer.eos_token
df_eval = pd.read_csv('df_eval.csv')
class BinaryClassifier(nn.Module):
    def __init__(self, input_size=4096):
        super().__init__()
        self.fc = nn.Linear(input_size, 2)
        
    def forward(self, x):
        return self.fc(x)

classifier = BinaryClassifier().to(device := torch.device("cuda" if torch.cuda.is_available() else "cpu"))
checkpoint_path = Path("/home/jovyan/work/Experiment_Final/model_checkpoint_learned/llama/model_checkpoint_step_1.pth")
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
classifier.load_state_dict(checkpoint['classifier_state_dict'])
print("Loaded checkpoint!")

def collate_fn(prompt):
    encodings = tokenizer(
        prompt,
        max_length=350,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    return {
        "input_ids": encodings.input_ids,
        "attention_mask": encodings.attention_mask,
    }

class HallucinationDetector:
    def __init__(self, model, classifier, batch, device):
        self.model = model
        self.batch = batch
        self.classifier = classifier
        self.device = device

    def evaluate(self):
        self.model.eval()
        predictions = torch.tensor([], dtype=torch.float).to(self.device)        
        input_ids = self.batch['input_ids'].to(self.device)
        attention_mask = self.batch['attention_mask'].to(self.device)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.hidden_states[-1].mean(dim=1)
        logits = self.classifier(embeddings)
        probas = F.softmax(logits, dim = -1)
        prediction = torch.argmax(probas, dim=-1)
        return prediction


def prompt_template(tweet: str) -> str:
    return f"""
                You are a sentiment-analysis expert tasked with analysing sentiment of the given tweet.
                Please identify the sentiment (positive, negative, neutral) of the following tweet:
                \"{tweet}\"
                Your answer must follow this exact format:
                Reasoning: <Explain your step-by-step reasoning here>
                Sentiment: <One of: Positive, Negative, Neutral>
            """

def response_model(tweet):
    prompt = prompt_template(tweet)
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "developer",
                "content": "You are a sentiment-analysis expert. You will be provided with a tweet and must classify its sentiment."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )                            
    output = response.output_text.strip().split("Sentiment:")
    reasoning = output[0].replace("Reasoning:", "").strip()
    sentiment = output[1].strip() if len(output) > 1 else "Unknown"
    return {
        "Tweet": tweet,
        "Predicted Reasoning": reasoning,
        "Predicted Sentiment": sentiment
    }

responses = pd.DataFrame(columns = ['Tweet', 'Predicted Sentiment', 'Predicted Reasoning', 'Hallucination'])
# Gold == 1, correct sentiment prediction

def dataset(dataset_train, b):
  row_data = {
      'Tweet': b['Tweet'],
      'Actual Sentiment': b['Actual Sentiment'],
      'Predicted Sentiment': b['Predicted Sentiment'],
      'Predicted Reasoning': b['Predicted Reasoning'],
      'Hallucination': b['Hallucination'],
  }
  new_row_df = pd.DataFrame([row_data])
  return pd.concat([dataset_train, new_row_df], ignore_index=True)

def add_prompts(x):
    return (
        f"""
        You are a sentiment analysis expert. Analyze whether the predicted sentiment of the tweet and reasoning is correct:\nTweet: {x['Tweet']}\nPredicted Sentiment: {x['Predicted Sentiment']}\nReasoning: {x['Predicted Reasoning']}\nThe answer to the whether the predicted reasoning and predicted sentiment is hallucinating is """.strip()
    )

for _, response in tqdm(df_eval.iterrows(), total=len(df_eval)):
    prompt = add_prompts(response)
    batch = collate_fn(prompt)
    prediction  = HallucinationDetector(model, classifier, batch, device).evaluate()
    pred_int = int(prediction.cpu().item())
    response['Hallucination'] = pred_int
    responses = dataset(responses, response)
    
responses.to_csv("df_eval-part-2.csv", index=False)

total_samples       = len(df_eval)
print(f"Total samples: {total_samples}")