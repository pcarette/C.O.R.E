import argparse
import random
import time
import os

def generate_fake_genome(filename, size_gb):
    CHUNK_SIZE = 1024 * 1024 * 64
    LINE_WIDTH = 80
    BASES = "ACGT"

    print(f"[GEN] Starting generation of {size_gb} GB synthetic genome...")
    print(f"[GEN] Target: {filename}")

    start_time = time.time()
    target_bytes = int(size_gb * 1024 * 1024 * 1024)
    bytes_written = 0
    print("[GEN] Pre-computing random patterns...")
    base_pool = ''.join(random.choices(BASES, k=1024 * 1024))
    lines = []
    for i in range(0, len(base_pool), LINE_WIDTH):
        chunk = base_pool[i:i+LINE_WIDTH]
        if len(chunk) == LINE_WIDTH:
            lines.append(chunk + "\n")

    write_buffer = "".join(lines).encode('utf-8')
    buffer_len = len(write_buffer)

    try:
        with open(filename, "wb") as f:
            header = f">SYNTHETIC_CHROMOSOME_1 length={target_bytes}\n".encode('utf-8')
            f.write(header)
            bytes_written += len(header)

            while bytes_written < target_bytes:
                remaining = target_bytes - bytes_written
                if remaining < buffer_len:
                    f.write(write_buffer[:remaining])
                    bytes_written += remaining
                else:
                    f.write(write_buffer)
                    bytes_written += buffer_len

                progress = (bytes_written / target_bytes) * 100
                if int(progress) % 10 == 0 and int(progress) > 0:
                    print(f"\r[GEN] Progress: {progress:.1f}% ({bytes_written // (1024*1024)} MB)", end="")

    except IOError as e:
        print(f"\n[ERROR] Disk I/O Error: {e}")
        return

    end_time = time.time()
    duration = end_time - start_time
    speed = (bytes_written / 1024 / 1024) / duration

    print(f"\n[GEN] Generation Complete.")
    print(f"       File Size: {bytes_written / 1024 / 1024 / 1024:.2f} GB")
    print(f"       Time:      {duration:.2f} seconds")
    print(f"       Write Speed: {speed:.2f} MB/s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate massive synthetic FASTA files.")
    parser.add_argument("--size", type=float, default=1.0, help="Size in GB (default: 1.0)")
    parser.add_argument("--out", type=str, default="synthetic_genome.fasta", help="Output filename")
    args = parser.parse_args()
    generate_fake_genome(args.out, args.size)

