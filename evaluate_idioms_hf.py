import os
import time
import pandas as pd
from dotenv import load_dotenv
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from huggingface_hub import InferenceClient

load_dotenv()

# =========================
# Configuration
# =========================
HF_API_KEY = os.getenv("HF_API_KEY")
INPUT_FILE = "idioms.csv"
OUTPUT_FILE = "idiom_evaluation_results_hf.csv"

# Hugging Face models to evaluate
MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "meta-llama/Llama-3.2-3B-Instruct",
    "google/gemma-2-2b-it",
    "microsoft/Phi-3-mini-4k-instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "tiiuae/falcon-7b-instruct",
    "HuggingFaceH4/zephyr-7b-beta",
    "openchat/openchat-3.5-0106",
    "cognitivecomputations/dolphin-2.9-llama3-8b",
    "teknium/OpenHermes-2.5-Mistral-7B",
]

# Create ONE client (don’t recreate per idiom)
# provider="hf-inference" uses Hugging Face’s serverless inference provider
CLIENT = InferenceClient(provider="hf-inference", api_key=HF_API_KEY)


# =========================
# HF call (TEXT GENERATION, not chat)
# =========================
def get_sinhala_idiom_prediction(model: str, english_idiom: str, english_example: str) -> str:
    """
    Calls Hugging Face Inference Providers via huggingface_hub InferenceClient
    using TEXT GENERATION (avoids 'conversational' task errors).

    Returns:
      - predicted Sinhala idiom (string) OR
      - "Error: ..." string
    """
    if not HF_API_KEY:
        return "Error: HF_API_KEY environment variable is not set."

    prompt = (
        f"English Idiom: '{english_idiom}'\n"
        f"Example Context: '{english_example}'\n\n"
        "Give the corresponding Sinhala idiom ONLY. "
        "No explanations. No meaning translation. Output just the idiom."
    )

    try:
        text = CLIENT.text_generation(
            prompt,
            model=model,
            max_new_tokens=80,
            temperature=0.2,
            return_full_text=False,
        )

        # Robust cleanup: first non-empty line, strip quotes
        text = (text or "").strip()
        first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
        return first_line.strip("“”\"' ").strip()

    except Exception as e:
        return f"Error: {e}"


# =========================
# BLEU
# =========================
def calculate_bleu(reference: str, candidate: str) -> float:
    """
    Calculates BLEU score between reference and candidate.
    Uses smoothing because idioms are short.
    """
    ref_tokens = str(reference).split()
    cand_tokens = str(candidate).split()

    cc = SmoothingFunction()
    return sentence_bleu([ref_tokens], cand_tokens, smoothing_function=cc.method1)


# =========================
# Main evaluation loop
# =========================
def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    if not HF_API_KEY:
        print("WARNING: HF_API_KEY is not set. API calls will fail.")
        print("Usage: export HF_API_KEY='your_key' && python evaluate_idioms_hf.py")
        return

    print("Loading idioms...")
    df = pd.read_csv(INPUT_FILE)

    required_cols = {"English Idiom", "Figurative Example", "Sinhala Idiom"}
    if not required_cols.issubset(set(df.columns)):
        print("Error: CSV must contain columns: 'English Idiom', 'Figurative Example', 'Sinhala Idiom'")
        print("Found columns:", list(df.columns))
        return

    results = []
    print(f"Starting evaluation on {len(df)} idioms across {len(MODELS)} models.")

    for model in MODELS:
        print(f"\nEvaluating Model: {model}")
        model_scores = []

        for index, row in df.iterrows():
            english_idiom = row["English Idiom"]
            english_example = row["Figurative Example"]
            reference_sinhala = row["Sinhala Idiom"]

            print(f"  Idiom {index + 1}/{len(df)}: '{english_idiom}'...", end="", flush=True)

            # Small delay to reduce provider rate-limit risk
            time.sleep(2)

            prediction = get_sinhala_idiom_prediction(model, english_idiom, english_example)

            if prediction and not prediction.startswith("Error"):
                score = calculate_bleu(reference_sinhala, prediction)
                print(f" Done. BLEU: {score:.4f}")
            else:
                score = 0.0
                print(f" Failed. {prediction}")

            model_scores.append(score)
            results.append(
                {
                    "Model": model,
                    "English Idiom": english_idiom,
                    "English Example": english_example,
                    "Reference Sinhala": reference_sinhala,
                    "Predicted Sinhala": prediction,
                    "BLEU Score": score,
                }
            )

        avg_bleu = sum(model_scores) / len(model_scores) if model_scores else 0.0
        print(f"Model {model} Average BLEU: {avg_bleu:.4f}")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)

    excel_output = OUTPUT_FILE.replace(".csv", ".xlsx")
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
        import nltk  # noqa: F401
    except ImportError:
        print("Please install nltk: pip install nltk")

    main()
