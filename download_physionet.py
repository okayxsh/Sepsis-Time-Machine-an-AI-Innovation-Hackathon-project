"""Fallback data download: PhysioNet Challenge 2019 (open access), one .psv per patient.
Usage: python download_physionet.py   -> data/physionet/training_setA|B/*.psv
"""
import os
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "https://physionet.org/files/challenge-2019/1.0.0/training/"
S3 = "https://physionet-open.s3.amazonaws.com/challenge-2019/1.0.0/training/"
OUT =os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "physionet")
UA = {"User-Agent": "Mozilla/5.0 (research-download)"}


def get(url, timeout=15):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch(args):
    url, path = args
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return 0
    for attempt in range(4):
        try:
            data = get(url)
            with open(path, "wb") as f:
                f.write(data)
            return 1
        except Exception:
            continue
    return -1


def main():
    jobs = []
    for s in ["training_setA", "training_setB"]:
        d = os.path.join(OUT, s)
        os.makedirs(d, exist_ok=True)
        html = get(BASE + s + "/", timeout=120).decode("utf-8", "ignore")
        names = sorted(set(re.findall(r'href="(p\d+\.psv)"', html)))
        print(s, len(names), "files", flush=True)
        # physionet.org redirects each file to its open-data S3 mirror; go there directly (much faster)
        jobs += [(S3 + s + "/" + n, os.path.join(d, n)) for n in names]
    done = fail = 0
    with ThreadPoolExecutor(max_workers=48) as ex:
        for i, r in enumerate(ex.map(fetch, jobs), 1):
            fail += r == -1
            if i % 2000 == 0:
                print(f"{i}/{len(jobs)} (failed {fail})", flush=True)
    print(f"DONE {len(jobs)} files, failed {fail}", flush=True)
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
