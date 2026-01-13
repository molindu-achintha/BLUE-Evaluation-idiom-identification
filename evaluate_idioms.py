import pandas as pd
import requests
import json
import os
import time
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import sys
from dotenv import load_dotenv

load_dotenv()


# Configuration
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

def get_sinhala_idiom_prediction(model, english_idiom, english_example):
    """
    Calls OpenRouter API to get the Sinhala idiom for a given English idiom and example.
    """
    if not OPENROUTER_API_KEY:
        print("Error: OPENROUTER_API_KEY environment variable is not set.")
        return None

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    prompt = f"English Idiom: '{english_idiom}'\nExample Context: '{english_example}'\n\nWhat is the Sinhala idiom that corresponds to this? Respond with the Sinhala idiom only. Do not provide explanations or translations of the meaning."
    
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['content'].strip()
        else:
            return "Error: No response content"
    except Exception as e:
        return f"Error: {str(e)}"

def calculate_bleu(reference, candidate):
    """
    Calculates BLEU score between reference key and candidate.
    Uses smoothing method 1 because idioms are short.
    """
    # Tokenize (simple split by space for Sinhala can be tricky, but basic split is a start)
    # Ideally, a Sinhala tokenizer should be used, but for general comparison split() is often used as a baseline.
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    
    cc = SmoothingFunction()
    return sentence_bleu([ref_tokens], cand_tokens, smoothing_function=cc.method1)

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    print("Loading idioms...")
    df = pd.read_csv(INPUT_FILE)
    
    # Check if necessary columns exist
    if 'English Idiom' not in df.columns or 'Sinhala Idiom' not in df.columns or 'Figurative Example' not in df.columns:
        print("Error: CSV must contain 'English Idiom', 'Figurative Example', and 'Sinhala Idiom' columns.")
        return

    results = []

    print(f"Starting evaluation on {len(df)} idioms across {len(MODELS)} models.")
    
    for model in MODELS:
        print(f"\nEvaluating Model: {model}")
        model_scores = []
        
        for index, row in df.iterrows():
            english_idiom = row['English Idiom']
            english_example = row['Figurative Example']
            reference_sinhala = row['Sinhala Idiom']
            
            print(f"  Idiom {index + 1}/{len(df)}: '{english_idiom}'...", end="", flush=True)
            
            # Add a small delay to avoid hitting rate limits too hard
            time.sleep(1) 
            
            prediction = get_sinhala_idiom_prediction(model, english_idiom, english_example)
            
            if prediction and not prediction.startswith("Error"):
                score = calculate_bleu(reference_sinhala, prediction)
                print(f" Done. BLEU: {score:.4f}")
            else:
                score = 0
                print(f" Failed. {prediction}")

            model_scores.append(score)
            results.append({
                "Model": model,
                "English Idiom": english_idiom,
                "English Example": english_example,
                "Reference Sinhala": reference_sinhala,
                "Predicted Sinhala": prediction,
                "BLEU Score": score
            })
        
        avg_bleu = sum(model_scores) / len(model_scores) if model_scores else 0
        print(f"Model {model} output score: {avg_bleu:.4f}")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)
    
    excel_output = OUTPUT_FILE.replace('.csv', '.xlsx')
    try:
        results_df.to_excel(excel_output, index=False)
        print(f"\nEvaluation complete. Results saved to {OUTPUT_FILE} and {excel_output}")
    except ImportError:
        print(f"\nEvaluation complete. Results saved to {OUTPUT_FILE}")
        print("To save as Excel, please install openpyxl: pip install openpyxl")
    except Exception as e:
        print(f"\nEvaluation complete. Results saved to {OUTPUT_FILE}")
        print(f"Could not save Excel file: {e}")

if __name__ == "__main__":
    try:
        import nltk
        try:
            pass 
        except LookupError:
            pass
    except ImportError:
        print("Please install nltk: pip install nltk")
    
    if not OPENROUTER_API_KEY:
        print("WARNING: OPENROUTER_API_KEY is not set. API calls will fail.")
        print("Usage: export OPENROUTER_API_KEY='your_key' && python evaluate_idioms.py")
    
    main()
