import argparse
import numpy as np
import pandas as pd
import os
import sys
import time

FEATURES = {
    'atac': 0,
    'blacklist': 1,
    'h3k4me3': 2,
    'h3k27ac': 3,
    'ctcf': 4,
    'tss': 5
}


def load_chrom_sizes(sizes_file):
    try:
        df = pd.read_csv(sizes_file, sep="\t", header=None, names=['chrom', 'size'])
    except Exception as e:
        print(f"[ERROR]: Failed to read {sizes_file}: {e}")
        sys.exit(1)

    chrom_mapa = {}
    current_offset = 0
    total_genome_size = 0

    for _, row in df.iterrows():
        chrom = row['chrom']
        size = row['size']

        chrom_mapa[chrom] = {
            'size': size,
            'offset': current_offset
        }

        total_genome_size += size
        current_offset += size

    print(f"[INFO] Total Genome Size: {total_genome_size / 1e9:.2f} Gbp")
    return chrom_mapa, total_genome_size


def process_bed_feature(bed_file, feature_name, chrom_map, epigenome_array):
    if not bed_file:
        return

    bit_pos = FEATURES.get(feature_name)
    if bit_pos is None:
        print(f"[WARN] Unknown feature {feature_name}, skipping.")
        return

    if not os.path.exists(bed_file):
        print(f"[WARN] File not found: {bed_file}. Skipping {feature_name}.")

    print(f"[INFO] Processing {feature_name.upper()} from {bed_file} (Target Bit: {bit_pos})...")

    mask_val = 1 << bit_pos
    chunk_size = 100000
    peaks_count = 0

    try:
        chunks = pd.read_csv(bed_file, sep='\t', usecols=[0, 1, 2], names=['chrom', 'start', 'end'],
                             chunksize=chunk_size, header=None, comment='#', on_bad_lines="skip")

        for chunk in chunks:

            for _, row in chunk.iterrows():
                chrom = row['chrom']
                start = int(row['start'])
                end = int(row['end'])

                if chrom in chrom_map:

                    offset = chrom_map[chrom]['offset']
                    chrom_size = chrom_map[chrom]['size']

                    if start < 0:
                        start = 0
                    if end > chrom_size:
                        end = chrom_size
                    if start >= end:
                        continue

                    global_start = offset + start
                    global_end = offset + end

                    epigenome_array[global_start:    global_end] |= mask_val
                    peaks_count += 1

        print(f"[INFO] Processed {peaks_count} peaks/regions for {feature_name}.")
    except Exception as e:
        print(f"[ERROR] Error processing {bed_file}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ATAC-seq BED file to Binary Genome Mask for C.O.R.E")
    parser.add_argument("--sizes", required=True, help="Path to chrom.sizes file")
    parser.add_argument("--output", required=True, help="Output .epi binary file")

    parser.add_argument("--atac", help="Path to ATAC-seq/DNase-seq BED file")
    parser.add_argument("--blacklist", help="Path to ENCODE Blacklist BED file")
    parser.add_argument("--h3k4me3", help="Path to H3K4me3 ChIP-seq BED file")
    parser.add_argument("--h3k27ac", help="Path to H3K27ac ChIP-seq BED file")
    parser.add_argument("--ctcf", help="Path to CTCF ChIP-seq BED file")
    parser.add_argument("--tss", help="Path to TSS Windows BED file")

    args = parser.parse_args()

    start_time = time.time()
    chromosome_map, total_size = load_chrom_sizes(args.sizes)

    print(f"[INFO] Allocating {total_size / 1024 ** 3:.2f} GB of RAM for Epigenome Atlas...")
    epigenome = np.zeros(total_size, dtype=np.uint8)

    process_bed_feature(args.atac, 'atac', chromosome_map, epigenome)
    process_bed_feature(args.blacklist, 'blacklist', chromosome_map, epigenome)
    process_bed_feature(args.h3k4me3, 'h3k4me3', chromosome_map, epigenome)
    process_bed_feature(args.h3k27ac, 'h3k27ac', chromosome_map, epigenome)
    process_bed_feature(args.ctcf, 'ctcf', chromosome_map, epigenome)
    process_bed_feature(args.tss, 'tss', chromosome_map, epigenome)

    print(f"[INFO] Saving binary atlas to {args.output}...")
    epigenome.tofile(args.output)

    print(f"[DONE] Total execution: {time.time() - start_time:.2f}s")
