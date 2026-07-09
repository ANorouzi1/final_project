import argparse
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


BASE_URL = "https://data.source.coop/kerner-lab/fields-of-the-world"
DEFAULT_COUNTRY = "austria"
REQUIRED_PATTERNS = [
    "s2_images/window_a/{aoi_id}.tif",
    "s2_images/window_b/{aoi_id}.tif",
    "label_masks/semantic_2class/{aoi_id}.tif",
    "label_masks/instance/{aoi_id}.tif",
]


def download_file(url, path, retries=3, overwrite=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0 and not overwrite:
        return "skipped", path

    part_path = path.with_suffix(path.suffix + ".part")
    for attempt in range(1, retries + 1):
        result = subprocess.run(
            ["wget", "-q", "-c", "-O", os.fspath(part_path), url],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0 and part_path.exists() and part_path.stat().st_size > 0:
            os.replace(part_path, path)
            return "downloaded", path
        if attempt == retries:
            return f"failed: {result.stderr.strip() or 'wget exited non-zero'}", path
        time.sleep(2 * attempt)


def ensure_metadata(root, country, retries, overwrite=False):
    country_dir = root / country
    parquet = country_dir / f"chips_{country}.parquet"
    if parquet.exists() and parquet.stat().st_size > 0 and not overwrite:
        return parquet

    url = f"{BASE_URL}/{country}/chips_{country}.parquet"
    status, path = download_file(url, parquet, retries=retries, overwrite=overwrite)
    if status not in {"downloaded", "skipped"}:
        raise FileNotFoundError(
            f"Could not download metadata parquet for country '{country}'. "
            f"Expected URL: {url}. {status}"
        )
    return path


def build_jobs(root, country, splits, include_semantic_3class, retries, overwrite):
    parquet = ensure_metadata(root, country, retries=retries, overwrite=False)
    df = pd.read_parquet(parquet)
    if "split" not in df.columns or "aoi_id" not in df.columns:
        raise ValueError(f"Unexpected metadata columns in {parquet}: {list(df.columns)}")

    selected = df[df["split"].isin(splits)]["aoi_id"].astype(str).tolist()
    patterns = list(REQUIRED_PATTERNS)
    if include_semantic_3class:
        patterns.append("label_masks/semantic_3class/{aoi_id}.tif")

    jobs = []
    for aoi_id in selected:
        for pattern in patterns:
            rel = f"{country}/{pattern.format(aoi_id=aoi_id)}"
            jobs.append((f"{BASE_URL}/{rel}", root / rel, retries, overwrite))
    return jobs, len(selected)


def main():
    parser = argparse.ArgumentParser(
        description="Download all required FTW France files, skipping files already present."
    )
    parser.add_argument("--root", default="data/ftw")
    parser.add_argument("--country", default=DEFAULT_COUNTRY)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
        help="Dataset splits to download. Default downloads all France chips.",
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--include-semantic-3class",
        action="store_true",
        help="Also download semantic_3class masks. The current project does not require them.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    jobs, n_chips = build_jobs(
        root=root,
        country=args.country,
        splits=args.splits,
        include_semantic_3class=args.include_semantic_3class,
        retries=args.retries,
        overwrite=args.overwrite,
    )
    print(
        f"Prepared {len(jobs)} file jobs for {n_chips} {args.country} chips "
        f"({', '.join(args.splits)})."
    )

    if args.dry_run:
        for url, path, _, _ in jobs[:20]:
            print(f"{url} -> {path}")
        if len(jobs) > 20:
            print(f"... {len(jobs) - 20} more files")
        return

    counts = {"downloaded": 0, "skipped": 0}
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(download_file, url, path, retries, overwrite)
            for url, path, retries, overwrite in jobs
        ]
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
        for status, path in failures[:30]:
            print(f"{status}: {path}")
        if len(failures) > 30:
            print(f"... {len(failures) - 30} more failures")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
