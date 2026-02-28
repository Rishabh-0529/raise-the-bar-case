# ======================================
# FLAN-T5-BASE TEXT SUMMARIZER + POINTERS
# ======================================

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, logging
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
import torch
from pathlib import Path
import re

# -------------------------------
# CONFIG
# -------------------------------
MODEL_NAME = "google/flan-t5-base"
logging.set_verbosity_error()

# -------------------------------
# LOAD MODEL
# -------------------------------
print("Loading FLAN-T5-BASE...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"Model loaded successfully on {device}")

# -------------------------------
# TEXT CLEANING
# -------------------------------
def clean_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\r", "")
    lines = text.split("\n")

    cleaned_lines = []
    for line in lines:
        s = line.strip()

        if not s:
            continue
        if s.isupper() and len(s) < 200:
            continue
        if s.startswith("Video:"):
            continue
        if re.match(r"^(ADVERTISEMENT|STORIES YOU MAY LIKE)$", s, re.I):
            continue

        cleaned_lines.append(s)

    cleaned = " ".join(cleaned_lines)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r'\b(\w+)( \1\b)+', r'\1', cleaned)

    return cleaned.strip()

# -------------------------------
# PDF EXTRACTION
# -------------------------------
def extract_text_from_pdf(path: Path) -> str:
    try:
        reader = PdfReader(str(path))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return clean_text(text)

    except PdfReadError as e:
        print("PDF Read Error:", e)
        return ""
    except Exception as e:
        print("Unexpected Error:", e)
        return ""

# -------------------------------
# CHUNKING
# -------------------------------
def chunk_text(text, chunk_size=400):
    words = text.split()
    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]

# -------------------------------
# GENERATION FUNCTION
# -------------------------------
def generate_output(prompt, max_input_length=1024, max_output_length=200):
    inputs = tokenizer(
        prompt,
        max_length=max_input_length,
        truncation=True,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    output_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        max_length=max_output_length,
        min_length=80,
        num_beams=4,
        no_repeat_ngram_size=3,
        repetition_penalty=1.5,
        length_penalty=2.0,
        early_stopping=True
    )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# -------------------------------
# SUMMARIZATION
# -------------------------------
def summarize_text(text):
    prompt = f"Summarize the following article clearly in 6-8 sentences:\n{text}"
    return generate_output(prompt)

# -------------------------------
# POINTER GENERATION
# -------------------------------
def generate_pointers(text):
    prompt = f"Extract the key points from the following article as bullet points:\n{text}"
    return generate_output(prompt, max_output_length=250)

# -------------------------------
# LONG TEXT PROCESSING
# -------------------------------
def summarize_long_text(text):
    chunks = chunk_text(text)
    summaries = [summarize_text(chunk) for chunk in chunks]
    return " ".join(summaries)

def generate_long_pointers(text):
    chunks = chunk_text(text)
    pointers = [generate_pointers(chunk) for chunk in chunks]
    return "\n".join(pointers)

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":

    pdf_path = Path("Being_a_12th_Fail_A_Story_of_Resilience.pdf")
    text = ""

    if pdf_path.exists():
        print("Reading PDF...")
        text = extract_text_from_pdf(pdf_path)

    if not text:
        txt_path = Path("pdf1.txt")
        if txt_path.exists():
            print("Using pdf1.txt fallback...")
            text = txt_path.read_text(encoding="utf-8")
            text = clean_text(text)

    if not text:
        print("No valid input file found. Exiting.")
        exit()

    print("\n--- CLEANED INPUT PREVIEW ---\n")
    print(text[:800])

    print("\n--- GENERATING SUMMARY ---\n")
    summary = summarize_long_text(text)

    print("========== SUMMARY ==========\n")
    print(summary)

    print("\n--- GENERATING KEY POINTERS ---\n")
    pointers = generate_long_pointers(text)

    print("========== KEY POINTS ==========\n")
    print(pointers)