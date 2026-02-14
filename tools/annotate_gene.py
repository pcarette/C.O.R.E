import re
import subprocess
import sys
import os
import pandas as pd
from pybiomart import Server

CORE_BINARY = "./cmake-build-release/core_runner"
DATA_FASTA = "./data/hg38_full.fa"
DATA_EPI = "./data/hg38_hek293.epi"


def get_gene_info(coordinates):
    if not coordinates:
        return {}

    print("[API] Connecting to Ensembl Biomart (pybiomart)...")
    try:
        server = Server(host='http://www.ensembl.org')
        dataset = server.marts['ENSEMBL_MART_ENSEMBL'].datasets['hsapiens_gene_ensembl']

        gene_annotations = dataset.query(
            attributes=[
                'external_gene_name',
                'chromosome_name',
                'start_position',
                'end_position'
            ]
        )

        gene_annotations.columns = [
            'external_gene_name',
            'chromosome_name',
            'start_position',
            'end_position'
        ]

        gene_annotations['start_position'] = pd.to_numeric(gene_annotations['start_position'], errors='coerce')
        gene_annotations['end_position'] = pd.to_numeric(gene_annotations['end_position'], errors='coerce')

    except Exception as e:
        print(f"[WARN] Biomart connection failed: {e}")
        return {}

    print(f"[API] Mapping hits against {len(gene_annotations)} genes...")

    annot_map = {}

    try:
        genes_by_chr = dict(tuple(gene_annotations.groupby('chromosome_name')))
    except KeyError as e:
        print(f"[WARN] GroupBy failed: {e}")
        return {}

    for position in coordinates:
        try:
            chrom_full, pos_str = position.split(":")
            pos = int(pos_str)

            chrom = chrom_full.replace("chr", "")

            if chrom in genes_by_chr:
                ga_chr = genes_by_chr[chrom]

                hits = ga_chr[
                    (ga_chr["start_position"] <= pos) &
                    (ga_chr["end_position"] >= pos)
                    ]

                if not hits.empty:
                    genes = hits['external_gene_name'].unique()
                    valid_genes = [str(g) for g in genes if g is not None and str(g) != 'nan']
                    if valid_genes:
                        annot_map[position] = ",".join(valid_genes)
                    else:
                        annot_map[position] = "."
                else:
                    annot_map[position] = "."
            else:
                annot_map[position] = "."
        except Exception:
            annot_map[position] = "ERR"

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
        loc = hit['loc']
        gene_label = annotations.get(loc, ".")

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

        print(f"{gene_label:<25} | {loc:<20} | {seq:<35} | {strand:<8} | {mis}")


if __name__ == "__main__":
    main()
