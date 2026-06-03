import os
import json
import random
import time
from pypdf import PdfReader
from groq import Groq
from tqdm import tqdm
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Initialize Groq Client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def extract_chunks_from_pdf(pdf_path, chunk_size=1000):
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        if page.extract_text():
            full_text += page.extract_text() + "\n"
    chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
    return [c.strip() for c in chunks if len(c.strip()) > 200]

def generate_synthetic_questions(pdf_path, target_count=100):
    print(f"[1/2] Processing PDF: {pdf_path}...")
    chunks = extract_chunks_from_pdf(pdf_path)
    
    if not chunks:
        raise ValueError("No readable text found in the PDF.")
        
    print(f"Extracted {len(chunks)} base text blocks. Generating {target_count} synthetic evaluation pairs...")
    dataset = []
    sampled_chunks = random.choices(chunks, k=target_count)
    
    for idx, chunk in enumerate(tqdm(sampled_chunks, desc="Generating Questions")):
        prompt = f"""
        You are an expert financial auditor evaluating an SEC document. 
        Based ONLY on the context snippet provided below, generate ONE highly specific, complex question that requires reading this exact snippet to answer.
        
        Context Snippet:
        \"\"\"{chunk}\"\"\"
        
        Provide your response in strict JSON format with exactly two keys:
        "question": "The complex question you generated"
        "ground_truth": "A direct, precise text sentence or phrase from the context snippet that answers it"
        """
        
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            data = json.loads(response.choices[0].message.content)
            dataset.append({
                "id": idx + 1,
                "question": data["question"],
                "ground_truth_text": data["ground_truth"],
                "source_chunk": chunk
            })
            
            # CRITICAL FIX: Sleep for 3 seconds to avoid Groq Rate Limits
            time.sleep(3) 
            
        except Exception as e:
            # CRITICAL FIX: Actually print the error so we aren't blind
            tqdm.write(f"\n[Warning] API Error on chunk {idx}: {str(e)}")
            time.sleep(5)  # Back off longer if we hit a limit
            continue

    output_file = "eval_dataset.json"
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=4)
    print(f"\n[Success] Generated {len(dataset)} evaluation pairs saved to {output_file}")

if __name__ == "__main__":
    PDF_FILE_PATH = "data/financial_report.pdf" 
    generate_synthetic_questions(PDF_FILE_PATH, target_count=100)