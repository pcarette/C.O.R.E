import os
import sys
import time

sys.path.append(os.path.join(os.getcwd(), "cmake-build-release"))

try:
    import core_engine
except ImportError as e:
    print(f"\n[FATAL] Could not load 'core_engine' module: {e}")
    print("Ensure you have compiled the project correctly.\n")
    sys.exit(1)


def main():
    fasta_path = "./data/hg38_full.fa"
    config_path = "./config/cas12a_config.json"
    query_seq = "TTTGGGGTGATCAGACCCAACAGCAGG"

    print("\n" + "=" * 75)
    print("      C.O.R.E - High-Performance CRISPR Search Engine")
    print("=" * 75 + "\n")
    print(f"[INFO] Initializing Engine...")
    print(f"[INFO] Genome: {os.path.basename(fasta_path)}")

    t_start_load = time.time()
    try:
        engine = core_engine.Engine(fasta_path)
    except Exception as e:
        print(f"[FATAL] Engine initialization failed: {e}")
        sys.exit(1)

    t_load = time.time() - t_start_load
    print(f"[SUCCESS] Engine ready in {t_load:.2f}s\n")

    print(f"[INFO] Executing Search...")
    print(f"[INFO] Query:  {query_seq}")
    print(f"[INFO] Config: {os.path.basename(config_path)}")

    t_start_search = time.time()
    try:
        hits = engine.search(query_seq, config_path)
    except Exception as e:
        print(f"[FATAL] Search failed: {e}")
        sys.exit(1)

    t_search = time.time() - t_start_search
    hits.sort(key=lambda x: (x.mismatches, x.chrom, x.pos))
    print(f"\n[RESULT] Scan complete in {t_search:.4f}s. Found {len(hits)} hits.\n")

    header = f"| {'LOCATION':<30} | {'STRAND':<6} | {'MM':<3} | {'SEQUENCE':<30} |"
    separator = "-" * len(header)
    print(separator)
    print(header)
    print(separator)

    for h in hits:
        location = f"{h.chrom}:{h.pos}"
        mm_str = str(h.mismatches)
        print(f"| {location:<30} | {h.strand:<6} | {mm_str:<3} | {h.sequence:<30} |")

    print(separator + "\n")


if __name__ == "__main__":
    main()
