import numpy as np
from openai import OpenAI
import os
import sys

sys.path.append(os.path.abspath("cmake-build-release"))
try:
    import core_engine

    print(f"[SUCCESS] Core Engine loaded successfully from: {core_engine.__file__}")
except ImportError as e:
    print(f"[FATAL] 'core_engine' Load failed")
    print(f"Error: {e}")
    print(f"Check whether you're running Python 3.14 or not.")
    sys.exit(1)

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ.get("NVIDIA_API_KEY")
)
MODEL_NAME = "meta/llama-3.1-70b-instruct"

def analyze_batch(hits_data):
    prompt_content = "Here is a list of potential off-target sequences detected by CRISPR:\n"
    for i, (hid, loc, seq) in enumerate(hits_data):
        prompt_content += f"{i + 1}. SEQ: {seq}, LOC: {loc}\n"

    prompt_content += ("For each sequence, indicate whether it falls within a known coding region (Gene) and "
                       "estimate the clinical risk (High/Medium/Low) based on the location.\n")

    try:
        completion = client.chat.completions.create(model=MODEL_NAME, messages=[
            {
                "role": "system",
                "content": "You are an expert in genome editing. Analyze the sequences provided. "
                           "Estimate the risk of clinical off-target effects based on the probable genomic "
                           "location and chromatin accessibility. Be concise."
            },
            {
                "role": "user",
                "content": prompt_content
            }
        ], temperature=0.2, max_tokens=1024)
        return completion.choices[0].message.content
    except Exception as e:
        return f"[ERROR] NIM Call failed: {e}"


def main():
    engine = core_engine.CoreEngine("./data/hg38_full.fa", "./data/hg38_k562.epi")
    target_seq = "CACACTCTCCAGGAAGACTG"
    print(f"[SEARCH] Searching: {target_seq}")
    match_ids = engine.search(target_seq, max_mismatches=3)
    print(f"[RESULT] {len(match_ids)} hits found")
    if len(match_ids) == 0:
        return
    top_hits = match_ids[:100]
    hits_info = []
    for bid in top_hits:
        loc, seq = engine.resolve_location(bid)
        hits_info.append((bid, loc, seq))

    BATCH_SIZE = 50
    for i in range(0, len(hits_info), BATCH_SIZE):
        batch = hits_info[i: i + BATCH_SIZE]
        print(f"  > Sending batch {i // BATCH_SIZE + 1} ({len(batch)} seqs)...")

        analysis = analyze_batch(batch)

        print("-" * 60)
        print(f"Batch report {i // BATCH_SIZE + 1} :\n{analysis}")
        print("-" * 60)


if __name__ == "__main__":
    main()
