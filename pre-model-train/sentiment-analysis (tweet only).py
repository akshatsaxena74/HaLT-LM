import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset as TorchDataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import logging
from sklearn.metrics import classification_report
from bitsandbytes.optim import Adam8bit
import random
random.seed(7777)

device = "cuda"

print(torch.cuda.is_available())

class CustomDataset(TorchDataset):
    def __init__(self, dataframe):
        self.data = dataframe.reset_index(drop=True)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return {
            "Prompt": self.data.loc[idx, "Prompt"],
            "Sentiment": self.data.loc[idx, "Sentiment"]
        }

model_id = "meta-llama/Meta-Llama-3-8B"

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

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

class BinaryClassifier(nn.Module):
    def __init__(self, input_size=4096):
        super().__init__()
        self.fc = nn.Linear(input_size, 3)

    def forward(self, x):
        return self.fc(x)

classifier = BinaryClassifier().to(device := torch.device("cuda" if torch.cuda.is_available() else "cpu"))
checkpoint_path = Path("model_checkpoint_learned/llama/model_checkpoint_step_1.pth")
if checkpoint_path.exists():
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    print("Loaded checkpoint!")
else:
    print("Initializing new model")

def create_datasets():
    full_train = pd.read_csv("dataset-step-1.csv")
    full_train = full_train.iloc[::-1].reset_index(drop=True)
    train_df = full_train.iloc[:20000].reset_index(drop=True)
    remaining = full_train.iloc[20000:]
    eval_df = remaining.sample(frac=0.05, random_state=42).reset_index(drop=True)

    sentiment_map = {
        "Negative": 0,
        "Neutral": 1,
        "Positive": 2
    }

    def add_prompts(df):
        return df.assign(
            Prompt=df.apply(
                lambda x: f"You are tasked with classifying tweets related to COVID-19 into one of three sentiment classes: positive, neutral, or negative. The tweet is:\n{x['OriginalTweet']}\n\nBased on the content of the tweet, the sentiment is ",
                axis=1
            ),
            Sentiment=df["Sentiment"].map(sentiment_map)
        )[["Prompt", "Sentiment"]]

    return (
        CustomDataset(add_prompts(train_df)), 
        CustomDataset(add_prompts(eval_df)),
        None  
    )

def collate_fn(batch):
    encodings = tokenizer(
        [x["Prompt"] for x in batch],
        max_length=350,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    return {
        "input_ids": encodings.input_ids,
        "attention_mask": encodings.attention_mask,
        "labels": torch.tensor([x["Sentiment"] for x in batch], dtype=torch.long)
    }


class Trainer:
    def __init__(self, model, classifier, train_loader, eval_loader, config, device):
        self.model = model
        self.classifier = classifier
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.config = config
        self.optimizer = Adam8bit(
            list(model.parameters()) + list(classifier.parameters()), 
            lr=2e-5
        )
        self.device = device
        
    def train(self):
        self.model.train()
        self.classifier.train()
    
        total_steps = 0
        print(f"\n=== Training for {self.config.max_steps} steps ===")
        
        while total_steps < self.config.max_steps:
            for batch_idx, batch in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device).long()
    
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = outputs.hidden_states[-1][:, -1, :]
                logits = self.classifier(embeddings)
                loss = F.cross_entropy(logits, labels)
    
                loss.backward()
                self.optimizer.step()
    
                total_steps += 1
                if total_steps % 10 == 0:
                    print(f"Step {total_steps} | Loss: {loss.item():.4f}")
                if total_steps % self.config.eval_steps == 0:
                    self.evaluate()
                if total_steps % self.config.save_steps == 0:
                    self.save_checkpoint()
                
                if total_steps >= self.config.max_steps:
                    break  
    
        self.save_checkpoint()
        print("Training completed!")

    def evaluate(self):
        self.model.eval()
        self.classifier.eval()
        references = torch.tensor([], dtype=torch.long).to(self.device)
        predictions = torch.tensor([], dtype=torch.long).to(self.device)
        total_eval_samples = len(self.eval_loader.dataset)
        counter = 0
        print(f"\n--- Starting Validation ---")
        print(f"Total evaluation samples: {total_eval_samples}")
        
        with torch.no_grad():
            for batch in self.eval_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device).long()
                n = input_ids.size(0)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                last_hidden_state = outputs.hidden_states[-1]
                last_token_embedding = last_hidden_state[:, -1, :]
                logits = self.classifier(last_token_embedding)
                prediction = torch.argmax(logits, dim=1)
                references = torch.cat((references, labels), dim=0)
                predictions = torch.cat((predictions, prediction), dim=0)
                counter += n
                print(f'Out of {total_eval_samples} eval datapoints {counter} processed')
        result = classification_report(references.cpu().numpy(), predictions.cpu().numpy(), digits=4)
        logging.info(f"Classification Report:\n{result}")
        print(f"Classification Report:\n{result}")
        accuracy = (predictions == references).sum().item() / len(references)
        logging.info(f"Average eval accuracy is {accuracy * 100:.2f}%")
        print(f"Average eval accuracy is {accuracy * 100:.2f}%")
        print(f"--- Evaluation Complete ---\n")
        self.model.train()
        self.classifier.train()

    def save_checkpoint(self):
        save_dir = Path("model_checkpoint_learned/llama")
        save_dir.mkdir(exist_ok=True)        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'classifier_state_dict': self.classifier.state_dict()
        }, save_dir / "model_checkpoint_step_1.pth")
    
        print("Checkpoint saved at model_checkpoint_learned/llama/model_checkpoint_step_1.pth!")

if __name__ == "__main__":
    train_ds, eval_ds, _ = create_datasets()
    train_loader = DataLoader(train_ds, batch_size=2, collate_fn=collate_fn, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=2, collate_fn=collate_fn, shuffle=False)

    class Config:
        num_epochs = 1
        max_steps = 10000
        eval_steps = 5000
        save_steps = 5000 

    trainer = Trainer(model, classifier, train_loader, eval_loader, Config(), device)
    trainer.train()
    print(f"Batch size for validation loader: {len(eval_loader.dataset)}")
    print("Training completed successfully!")