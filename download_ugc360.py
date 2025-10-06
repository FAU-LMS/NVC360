import argparse
import subprocess
from pathlib import Path
from huggingface_hub import hf_hub_download


FILES = {
    "S": ["UGC360-S.z01", "UGC360-S.z02", "UGC360-S.z03", "UGC360-S.zip"],
    "M": ["UGC360-M.z01", "UGC360-M.z02", "UGC360-M.zip"],
    "L": ["UGC360-L.zip"],
}

REPO = "FAU-LMS/UGC360"


def download_and_unpack(subset: str, outdir: Path):
    print(f"Downloading subset {subset} into {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)

    files = FILES[subset]
    local_files = []

    for fname in files:
        local_path = hf_hub_download(
            repo_id=REPO,
            repo_type="dataset",
            filename=f"{fname}",
            local_dir=outdir
        )
        local_files.append(Path(local_path))

    # multipart archives (S, M)
    if len(files) > 1:
        zipfile = next(f for f in local_files if f.suffix == ".zip")
        allzip = outdir / f"UGC360-{subset}-all.zip"
        print(f"Joining parts into {allzip.name}")
        subprocess.run(
            ["zip", "-q", "-s-", str(zipfile.name), "-O", str(allzip.name)],
            check=True,
            cwd=outdir
        )
        print("Cleaning up split archives")
        for f in local_files:
            f.unlink()
        print(f"Unpacking {allzip.name}")
        subprocess.run(["unzip", "-q", "-o", str(allzip.name)], check=True, cwd=outdir)
        allzip.unlink()
    else:
        # single archive (L)
        zipfile = local_files[0]
        print(f"Unpacking {zipfile.name}")
        subprocess.run(["unzip", "-q", "-o", str(zipfile.name)], check=True, cwd=outdir)
        zipfile.unlink()


def main():
    parser = argparse.ArgumentParser(description="Download UGC360 dataset subsets from HuggingFace")
    parser.add_argument(
        "--subset",
        choices=["S", "M", "L"],
        nargs="+",
        default=["S", "M", "L"],
        help="Which subsets to download (default: S M L).",
    )
    parser.add_argument(
        "--download_dir",
        type=Path,
        default=Path.cwd(),
        help="Directory to download into (default: current directory).",
    )
    args = parser.parse_args()

    for subset in args.subset:
        download_and_unpack(subset, args.download_dir)


if __name__ == "__main__":
    main()
