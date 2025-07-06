import os, time, random
import pandas as pd
from dotenv import load_dotenv
from textwrap import dedent
from tqdm import tqdm
from joblib import Parallel, delayed
from openai import OpenAI, OpenAIError
from huggingface_hub import login

random.seed(7777)
load_dotenv("key.env")

open_api = os.getenv("open_api")
hf_token  = os.getenv("hf_key")

if hf_token:
    login(token=hf_token)

client = OpenAI(api_key=open_api)
df = pd.read_csv("...")

def prompt_template(tweet):
    return dedent(f"""
                You are a sentiment-analysis expert tasked with analysing sentiment of the given tweet.
                Please identify the sentiment (positive, negative, neutral) of the following tweet:
                "{tweet}"
                Your answer must follow this exact format:
                Reasoning: <Explain your step-by-step reasoning here>
                Sentiment: <One of: Positive, Negative, Neutral>
            """)


def run_sentiment_model(row, model_name = "gpt-4.1-mini", max_retries = 5):
    tweet = row["OriginalTweet"]
    gold  = row["Sentiment"]
    prompt = prompt_template(tweet)
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system",
                     "content": """
                            You are a sentiment-analysis expert. You will be provided with a tweet and must classify its sentiment.
                           """},
                    {"role": "user", "content": prompt}
                ]
            )
            text = resp.choices[0].message.content.strip()
            parts = text.split("Sentiment:")
            reasoning = parts[0].replace("Reasoning:", "").strip()
            sentiment = parts[1].strip() if len(parts) > 1 else "Unknown"
            if gold == sentiment: 
                continue
            return {
                "Tweet": tweet,
                "Actual Sentiment": gold,
                "Predicted Reasoning": reasoning,
                "Predicted Sentiment": sentiment
            }
        except OpenAIError as e:
            print(f"OpenAI error (attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(2**attempt)
    return {
        "Tweet": tweet,
        "Actual Sentiment": gold,
        "Predicted Reasoning": "ERROR",
        "Predicted Sentiment": "ERROR"
    }


print(f"Generating GPT predictions for {len(df)} tweets …")

predictions = Parallel(n_jobs=8, prefer = "threads")(
    delayed(run_sentiment_model)(row)
    for _, row in tqdm(df.iterrows(), total=len(df))
)

pred_df = pd.DataFrame(predictions)
filtered_df = pred_df[pred_df["Actual Sentiment"] != pred_df["Predicted Sentiment"]]
data = pd.read_csv("...")

data_aligned = data[filtered_df.columns.intersection(data.columns)]
combined_df = pd.concat([filtered_df, data_aligned], ignore_index=True)
combined_df = combined_df.drop_duplicates()

out_path = "..."
combined_df.to_csv(out_path, index=False)
print(f"Saved predictions to {out_path}")
print(len(combined_df))