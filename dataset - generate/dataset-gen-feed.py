import os, time, random, re, pandas as pd
from dotenv import load_dotenv
from textwrap import dedent
from tqdm import tqdm
from joblib import Parallel, delayed
from openai import OpenAI, OpenAIError

random.seed(7777)
load_dotenv("key.env")
open_api = os.getenv("open_api")

client_global = OpenAI(api_key=open_api)

pred_path = "..."
out_path  = "..."

df = pd.read_csv(pred_path)

def build_prompt(tweet, reasoning, pred_sent, gold_sent):
    return dedent(f"""
        Consider the following tweet:

        "{tweet}"

        My LLM predicted the sentiment *{pred_sent}* with this explanation:

        "{reasoning}"

        According to the dataset, the **actual sentiment is {gold_sent}**.

        Your goal is to first justify why the sentiment should be {gold_sent}, and then generate feedback that I can use to help my LLM classify this tweet and other tweets correctly. Your feedback should therefore be somewhat abstract.

        Note that you must write concise and abstract feedback at the end, after proper reasoning, in the following format:

        Final Feedback: <Your feedback>
    """).strip()


def extract_feedback(text: str) -> str:
    m = re.search(r'final\s*feedback\s*:\s*(.+)', text, flags=re.I | re.S)
    if m:
        return m.group(1).strip().splitlines()[0].strip(' "')
    return text.strip()

def get_feedback(row, model_name="gpt-4.1-mini", max_retries=5):
    tweet      = row["Tweet"]
    reasoning  = row["Predicted Reasoning"]
    pred_sent  = row["Predicted Sentiment"].lower()
    gold_sent  = row["Actual Sentiment"].lower()

    if str(pred_sent).lower() == str(gold_sent).lower():
        return {"Final Feedback": "—"}

    prompt = build_prompt(tweet, reasoning, pred_sent, gold_sent)

    for attempt in range(max_retries):
        try:
            resp = client_global.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system",
                     "content": "You are a sentiment-analysis feedback expert. First justify why the gold label is correct, then write concise, abstract feedback that can help an LLM improve."},
                    {"role": "user", "content": prompt}
                ]
            )
            text = resp.choices[0].message.content or ""
            return {"Final Feedback": extract_feedback(text)}
        except OpenAIError as e:
            print(f"OpenAI error (attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(2 ** attempt)

    return {"Final Feedback": "ERROR"}


print(f"Generating feedback for {len(df)} rows …")

results = Parallel(n_jobs=16, prefer="threads")(
    delayed(get_feedback)(row) for _, row in tqdm(df.iterrows(), total=len(df))
)

feedback_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
feedback_df.to_csv(out_path, index=False)
print(f"Saved feedback to {out_path}")
