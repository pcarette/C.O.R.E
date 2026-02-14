import mygene
import re
import subprocess
import sys
import os

CORE_BINARY = "./cmake-build-release/core_runner"
DATA_FASTA = "./data/hg38_full.fa"
DATA_EPI = "./data/hg38_k562.epi"


def get_gene_info(coordinates):
    if not coordinates:
        return {}

    mg = mygene.MyGeneInfo()
    queries = []
    for location in coordinates:
        try:
            chrom, pos = location.split(":")
            pos = int(pos)
            queries.append(f"{chrom}:{pos}-{pos + 23}")
        except ValueError:
            continue

    print(f"[API] Querying MyGene.info for {len(queries)} sites...")

    try:
        results = mg.querymany(queries, scopes='genomic_pos', fields='symbol,name,type_of_gene', species='human',
                               verbose=False)
    except Exception as e:
        print(f"[WARN] API error: {e}")
        return {}

    annot_map = {}
    for res in results:
        query_key = res.get('query')
        hit_data = None
        if 'symbol' in res:
            hit_data = res
        elif 'hits' in res and len(res['hits']) > 0:
            hit_data = res['hits'][0]

        if hit_data and 'symbol' in hit_data:
            symbol = hit_data['symbol']
            gtype = hit_data.get('type_of_gene', 'unknown')
            annot_map[query_key] = f"{symbol} ({gtype})"
        else:
            annot_map[query_key] = "."

    return annot_map


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 tools/annotate_gene.py <SEQUENCE>")
        sys.exit(1)

    query_seq = sys.argv[1]

    if not os.path.exists(CORE_BINARY):
        print(f"[FATAL] Binary not found at {CORE_BINARY}")
        sys.exit(1)

    cmd = [CORE_BINARY, DATA_FASTA, DATA_EPI, query_seq]

    try:
        print(f"[WRAPPER] Running C.O.R.E engine...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        raw_output = result.stdout

        if result.returncode != 0:
            print(f"[FATAL] Engine crashed:\n{result.stderr}")
            sys.exit(1)

    except Exception as e:
        print(f"[FATAL] Execution error: {e}")
        sys.exit(1)

    hits = []
    regex = r"Position:\s+([\w\d_]+:\d+)"
    print("[WRAPPER] Parsing output...")

    lines = raw_output.split('\n')
    valid_coords = []

    for line in lines:
        if "Position" in line:
            match = re.search(regex, line)
            if match:
                loc = match.group(1)
                hits.append({'raw_line': line.strip(), 'loc': loc})
                valid_coords.append(loc)

    if not hits:
        print("[WRAPPER] No hits found.")
        print(raw_output)
        return

    annotations = get_gene_info(valid_coords)

    print("-" * 120)
    print(f"{'GENE':<25} | {'LOCATION':<20} | {'SEQ + PAM':<35} | {'STRAND':<8} | {'MIS'}")
    print("-" * 120)

    for hit in hits:
        chrom, pos = hit['loc'].split(':')
        pos_int = int(pos)

        gene_label = "."
        target_query = f"{chrom}:{pos_int}-{pos_int + 23}"

        if target_query in annotations:
            gene_label = annotations[target_query]

        parts = hit['raw_line'].split()
        try:
            seq_idx = -1
            for i, p in enumerate(parts):
                if all(c in 'ACGTN' for c in p) and len(p) > 10:
                    seq_idx = i
                    break
            seq = parts[seq_idx] if seq_idx != -1 else "Unknown"
        except:
            seq = "Parsing Error"

        strand = "(?)"
        if "(+)" in hit['raw_line']: strand = "(+)"
        if "(-)" in hit['raw_line']: strand = "(-)"

        mis = "0"
        if "[Mis:" in hit['raw_line']:
            try:
                mis = hit['raw_line'].split("[Mis:")[1].split("]")[0].strip()
            except:
                pass

        print(f"{gene_label:<25} | {hit['loc']:<20} | {seq:<35} | {strand:<8} | {mis}")


if __name__ == "__main__":
    main()
