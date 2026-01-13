import os
import re
import time
import random
import unicodedata

import pandas as pd
import requests
from dotenv import load_dotenv
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

INPUT_FILE = "idioms.csv"
OUTPUT_FILE = "idiom_evaluation_results.csv"

MODELS = [
    "mistralai/mistral-7b-instruct:free",
    "mistralai/devstral-2512:free",
    "mistralai/mistral-small-24b-instruct-2501:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemma-3-27b-it:free",
    "deepseek/deepseek-r1-0528:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "xiaomi/mimo-v2-flash:free",
    "tngtech/deepseek-r1t-chimera:free",
]


def normalize_si(text: str) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    text = text.strip('"').strip("'")
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200b", " ")
    text = re.sub(r"\s+", " ", text).strip()

    if "\n" in text:
        text = text.split("\n", 1)[0].strip()

    text = re.sub(r"^\s*Sinhala\s*Idiom\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    text = text.strip(" .,!?:;，。！？；៖|/\\")
    return text


def bleu1_score(reference: str, candidate: str) -> float:
    ref = normalize_si(reference)
    cand = normalize_si(candidate)

    if not cand or cand.lower().startswith("error"):
        return 0.0

    ref_tokens = ref.split()
    cand_tokens = cand.split()

    if not ref_tokens or not cand_tokens:
        return 0.0

    cc = SmoothingFunction()
    return float(sentence_bleu([ref_tokens], cand_tokens, weights=(1, 0, 0, 0), smoothing_function=cc.method1))


def char_bigram_f1(reference: str, candidate: str) -> float:
    ref = normalize_si(reference)
    cand = normalize_si(candidate)

    if not cand or cand.lower().startswith("error"):
        return 0.0

    ref = ref.replace(" ", "")
    cand = cand.replace(" ", "")

    if len(ref) < 2 or len(cand) < 2:
        return 1.0 if ref == cand and ref != "" else 0.0

    def bigrams(s: str):
        return [s[i : i + 2] for i in range(len(s) - 1)]

    ref_b = bigrams(ref)
    cand_b = bigrams(cand)

    ref_counts = {}
    for b in ref_b:
        ref_counts[b] = ref_counts.get(b, 0) + 1

    cand_counts = {}
    for b in cand_b:
        cand_counts[b] = cand_counts.get(b, 0) + 1

    overlap = 0
    for b, ccount in cand_counts.items():
        rcount = ref_counts.get(b, 0)
        overlap += min(rcount, ccount)

    precision = overlap / max(1, len(cand_b))
    recall = overlap / max(1, len(ref_b))

    if precision + recall == 0:
        return 0.0

    return float(2 * precision * recall / (precision + recall))


def levenshtein_similarity(reference: str, candidate: str) -> float:
    ref = normalize_si(reference).replace(" ", "")
    cand = normalize_si(candidate).replace(" ", "")

    if not cand or cand.lower().startswith("error"):
        return 0.0

    if ref == cand:
        return 1.0
    if not ref or not cand:
        return 0.0

    n = len(ref)
    m = len(cand)

    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        rch = ref[i - 1]
        for j in range(1, m + 1):
            cch = cand[j - 1]
            cost = 0 if rch == cch else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur

    dist = prev[m]
    return float(1.0 - (dist / max(n, m)))


def combined_score(reference: str, candidate: str) -> float:
    b1 = bleu1_score(reference, candidate)
    cf1 = char_bigram_f1(reference, candidate)
    lev = levenshtein_similarity(reference, candidate)

    score = (0.4 * b1) + (0.3 * cf1) + (0.3 * lev)
    if score < 0:
        score = 0.0
    if score > 1:
        score = 1.0
    return float(score)


def get_sinhala_idiom_prediction(model: str, english_idiom: str, english_example: str, max_retries: int = 10) -> str:
    if not OPENROUTER_API_KEY:
        return "Error: OPENROUTER_API_KEY environment variable is not set"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    prompt = (
        "Return only the Sinhala idiom phrase.\n"
        "No explanations. No quotes. No punctuation. No multiple options.\n\n"
        f"English Idiom: {english_idiom}\n"
        f"Context: {english_example}\n"
        "Sinhala Idiom:"
    )

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }

    url = "https://openrouter.ai/api/v1/chat/completions"

    last_status = None
    last_text = None

    for attempt in range(max_retries):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=90)

            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                if retry_after is not None:
                    try:
                        wait = float(retry_after)
                    except Exception:
                        wait = 10.0
                else:
                    wait = min(60.0, (2 ** attempt)) + random.uniform(0.5, 2.0)
                last_status = 429
                last_text = r.text[:300]
                time.sleep(wait)
                continue

            if r.status_code in (500, 502, 503, 504):
                wait = min(60.0, (2 ** attempt)) + random.uniform(0.5, 2.0)
                last_status = r.status_code
                last_text = r.text[:300]
                time.sleep(wait)
                continue

            if r.status_code == 401:
                return "Error: 401 Unauthorized (check OPENROUTER_API_KEY and account limits)"

            if r.status_code == 404:
                return f"Error: 404 Model not found or not available: {model}"

            r.raise_for_status()

            data = r.json()
            choices = data.get("choices", [])
            if not choices:
                return "Error: No choices in response"

            content = choices[0].get("message", {}).get("content", "")
            return normalize_si(content)

        except requests.exceptions.Timeout:
            wait = min(60.0, (2 ** attempt)) + random.uniform(0.5, 2.0)
            last_status = "timeout"
            last_text = "timeout"
            time.sleep(wait)
            continue

        except Exception as e:
            return f"Error: {str(e)}"

    detail = f"last_status={last_status}, last_body={last_text}"
    return f"Error: Retries exhausted ({detail})"


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    if not OPENROUTER_API_KEY:
        print("Error: OPENROUTER_API_KEY is not set.")
        return

    print("Loading idioms...")
    df = pd.read_csv(INPUT_FILE)

    required = {"English Idiom", "Figurative Example", "Sinhala Idiom"}
    if not required.issubset(set(df.columns)):
        print("Error: CSV must contain 'English Idiom', 'Figurative Example', and 'Sinhala Idiom' columns.")
        return

    results = []
    print(f"Starting evaluation on {len(df)} idioms across {len(MODELS)} models.")

    for model in MODELS:
        print(f"\nEvaluating Model: {model}")
        model_scores = []

        for idx, row in df.iterrows():
            english_idiom = str(row["English Idiom"])
            english_example = str(row["Figurative Example"])
            reference_sinhala = str(row["Sinhala Idiom"])

            print(f"  Idiom {idx + 1}/{len(df)}: '{english_idiom}'...", end="", flush=True)

            time.sleep(4.0 + random.uniform(0.0, 1.0))

            pred = get_sinhala_idiom_prediction(model, english_idiom, english_example)

            ref_norm = normalize_si(reference_sinhala)
            pred_norm = normalize_si(pred)

            exact = int(ref_norm == pred_norm and ref_norm != "")

            b1 = bleu1_score(reference_sinhala, pred)
            cf1 = char_bigram_f1(reference_sinhala, pred)
            lev = levenshtein_similarity(reference_sinhala, pred)
            score = combined_score(reference_sinhala, pred)

            if pred_norm.lower().startswith("error") or pred_norm == "":
                print(f" Failed. {pred}")
            else:
                print(f" Done. Score: {score:.4f}")

            model_scores.append(score)

            results.append(
                {
                    "Model": model,
                    "English Idiom": english_idiom,
                    "English Example": english_example,
                    "Reference Sinhala": reference_sinhala,
                    "Predicted Sinhala": pred,
                    "Reference Normalized": ref_norm,
                    "Predicted Normalized": pred_norm,
                    "Exact Match": exact,
                    "BLEU-1": b1,
                    "CharBigramF1": cf1,
                    "LevenshteinSim": lev,
                    "Final Score": score,
                }
            )

        avg_score = sum(model_scores) / len(model_scores) if model_scores else 0.0
        print(f"Model {model} average score: {avg_score:.4f}")

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_FILE, index=False)

    xlsx_file = OUTPUT_FILE.replace(".csv", ".xlsx")
    try:
        out_df.to_excel(xlsx_file, index=False)
        print(f"\nSaved: {OUTPUT_FILE}")
        print(f"Saved: {xlsx_file}")
    except Exception as e:
        print(f"\nSaved: {OUTPUT_FILE}")
        print(f"Excel save skipped: {e}")


if __name__ == "__main__":
    main()
