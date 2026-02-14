import json
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
MODEL_NAME = "nvidia/llama-3.3-nemotron-super-49b-v1.5"


def analyze_batch(hits_data):
    sequences_list = []
    for i, (hid, loc, seq) in enumerate(hits_data):
        sequences_list.append({"id": int(hid), "location": str(loc), "sequence": str(seq)})

    data_str = json.dumps(sequences_list, indent=2)

    system_prompt = (
        "You are a bioinformatics assistant specialized in CRISPR off-target risk assessment. "
        "Your task is to annotate genomic coordinates (hg38 assembly, cell line K562). "
        "Return ONLY a valid JSON object."
    )

    user_prompt = f"""
    Analyze the following list of potential off-target sites:
    {data_str}

    For each site:
    1. Identify the nearest Gene (Symbol). If intergenic, note 'Intergenic'.
    2. Determine the Genomic Region (Exon, Intron, Promoter, Enhancer).
    3. Infer Chromatin Accessibility based on the region (e.g., Promoters are usually 'Open', Heterochromatin is 'Closed').
    4. Assign a 'Clinical_Risk' (High/Medium/Low).
       - High: Coding exon or critical promoter.
       - Medium: Intron or regulatory region.
       - Low: Intergenic/Closed chromatin.

    Output format must be a JSON object with a key "analysis" containing the list of results.
    NO explanations, NO internal monologue, ONLY JSON.
    """

    try:
        completion = client.chat.completions.create(model=MODEL_NAME, messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ], temperature=0.2, max_tokens=2048, extra_body={"response_format": {"type": "json_object"}})
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
