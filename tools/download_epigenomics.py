import json
import urllib.request
import os
import gzip
import shutil
import time

REFERENCE_FILES = {
    "hg38_blacklist.bed": "https://www.encodeproject.org/files/ENCFF356LFX/@@download/ENCFF356LFX.bed.gz"
}

EXPERIMENTS = {
    "atac_peaks_k562.bed": "ENCSR483RKN",
    "h3k4me3_k562.bed":    "ENCSR000EWA",
    "h3k27ac_k562.bed":    "ENCSR000AKP",
    "ctcf_k562.bed":       "ENCSR000BPJ"
}

PRIORITY_TYPES = [
    'optimal IDR thresholded peaks',
    'conservative IDR thresholded peaks',
    'pseudoreplicated IDR thresholded peaks',
    'replicated peaks',
    'stable peaks',
    'peaks'
]

def get_best_file_url(experiment_id):
    print(f"[API] Querying experiment {experiment_id}...")
    api_url = f"https://www.encodeproject.org/experiments/{experiment_id}/?format=json"

    try:
        req = urllib.request.Request(api_url)
        req.add_header('User-Agent', 'Mozilla/5.0')
        req.add_header('Accept', 'application/json')

        with urllib.request.urlopen(req) as url:
            data = json.loads(url.read().decode())
            files = data.get('files', [])
            candidates = {}

            for f in files:
                if f.get('file_format') == 'bed' and \
                        f.get('assembly') == 'GRCh38' and \
                        f.get('status') == 'released':

                    out_type = f.get('output_type', '')
                    candidates[out_type] = "https://www.encodeproject.org" + f['href']

            for p_type in PRIORITY_TYPES:
                if p_type in candidates:
                    print(f"    -> Found best match: '{p_type}'")
                    return candidates[p_type]

            print(f"[WARN] No standard peak file found for {experiment_id}. Available types: {list(candidates.keys())}")

    except Exception as e:
        print(f"[ERROR] API Error for {experiment_id}: {e}")

    return None

def download_url(url, output_filename):
    print(f"[DL] Downloading to {output_filename}...")

    try:
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0')

        with urllib.request.urlopen(req) as response:
            total_size = int(response.info().get('Content-Length', 0).strip())
            downloaded = 0
            chunk_size = 1024 * 1024
            gz_tmp = output_filename + ".tmp.gz"

            with open(gz_tmp, 'wb') as out_file:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = downloaded / total_size * 100
                        print(f"\r    Progress: {percent:.1f}%", end='')

            print("\n[DL] Decompressing...")
            with gzip.open(gz_tmp, 'rb') as f_in:
                with open(output_filename, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            os.remove(gz_tmp)
            print(f"[SUCCESS] {output_filename} Ready.")
            return True

    except Exception as e:
        print(f"\n[FAIL] Error downloading {url}: {e}")
        return False

if __name__ == "__main__":
    print("[CORE] Starting Epigenomics Data Fetcher (Fixed Case)...")

    for fname, url in REFERENCE_FILES.items():
        if not os.path.exists(fname):
            download_url(url, fname)
        else:
            print(f"[SKIP] {fname} already exists.")

    for fname, exp_id in EXPERIMENTS.items():
        if os.path.exists(fname):
            print(f"[SKIP] {fname} already exists.")
            continue

        url = get_best_file_url(exp_id)
        if url:
            download_url(url, fname)
            time.sleep(0.5)
        else:
            print(f"[FATAL] Could not find any BED file for {exp_id}")

    print("[CORE] Data collection complete.")