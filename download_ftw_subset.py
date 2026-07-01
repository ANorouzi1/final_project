import argparse
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


BASE_URL = "https://data.source.coop/kerner-lab/fields-of-the-world"
REQUIRED_PATTERNS = [
    "s2_images/window_a/{aoi_id}.tif",
    "s2_images/window_b/{aoi_id}.tif",
    "label_masks/semantic_2class/{aoi_id}.tif",
    "label_masks/instance/{aoi_id}.tif",
]


def ensure_metadata(root, country):
    country_dir = root / country
    parquet = country_dir / f"chips_{country}.parquet"
    if parquet.exists() and parquet.stat().st_size > 0:
        return parquet

    url = f"{BASE_URL}/{country}/chips_{country}.parquet"
    status, path = download_file(url, parquet)
    if status not in {"downloaded", "skipped"}:
        raise FileNotFoundError(
            f"Could not download metadata parquet for country '{country}'. "
            f"Expected URL: {url}. {status}"
        )
    return path


def download_file(url, path, retries=3):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return "skipped", path

    for attempt in range(1, retries + 1):
        result = subprocess.run(
            ["wget", "-q", "-c", "-O", os.fspath(path), url],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0 and path.exists() and path.stat().st_size > 0:
            return "downloaded", path
        if attempt == retries:
            return f"failed: {result.stderr.strip() or 'wget exited non-zero'}", path
        time.sleep(2 * attempt)


def build_jobs(root, country, max_train, include_val):
    parquet = ensure_metadata(root, country)
    df = pd.read_parquet(parquet)
    selected = []
    train = df[df["split"] == "train"]["aoi_id"].astype(str).tolist()
    selected.extend(train[:max_train] if max_train else train)
    if include_val:
        selected.extend(df[df["split"] == "val"]["aoi_id"].astype(str).tolist())

    jobs = []
    for aoi_id in selected:
        for pattern in REQUIRED_PATTERNS:
            rel = f"{country}/{pattern.format(aoi_id=aoi_id)}"
            jobs.append((f"{BASE_URL}/{rel}", root / rel))
    return jobs, len(selected)


def main():
    parser = argparse.ArgumentParser(description="Download the FTW files needed by this project.")
    parser.add_argument("--root", default="data/ftw")
    parser.add_argument("--country", default="austria")
    parser.add_argument("--max-train", type=int, default=4000)
    parser.add_argument("--include-val", action="store_true", default=True)
    parser.add_argument("--no-val", action="store_false", dest="include_val")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    root = Path(args.root)
    jobs, n_chips = build_jobs(root, args.country, args.max_train, args.include_val)
    print(f"Downloading/checking {len(jobs)} files for {n_chips} {args.country} chips.")

    counts = {"downloaded": 0, "skipped": 0}
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download_file, url, path) for url, path in jobs]
        for index, future in enumerate(as_completed(futures), start=1):
            status, path = future.result()
            if status in counts:
                counts[status] += 1
            else:
                failures.append((status, path))
            if index == 1 or index % 100 == 0 or index == len(jobs):
                print(
                    f"{index}/{len(jobs)} complete "
                    f"({counts['downloaded']} downloaded, {counts['skipped']} skipped, "
                    f"{len(failures)} failed)",
                    flush=True,
                )

    if failures:
        print("Failures:")
        for status, path in failures[:20]:
            print(f"{status}: {path}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
