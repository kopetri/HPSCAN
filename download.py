"""Download the released checkpoints and the datasets.

    python download.py --checkpoints        # -> checkpoints/*.ckpt (GitHub release assets)
    python download.py --improved           # -> checkpoints/hpscan_improved.ckpt (post-publication, improved/)
    python download.py --psc                # -> data/PSC (Hugging Face: kopetri/PSC)
    python download.py --sdr                # -> data/SDR (GitHub release asset)

ClustMe (Table 5) is not redistributed, see README.md for the conversion.
"""
import argparse
import urllib.request
import zipfile
from pathlib import Path

from hpscan.paper import CHECKPOINTS, RELEASE_URL


def fetch(url, target):
    target = Path(target)
    if target.exists():
        print(f"exists: {target}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"download {url}")
    tmp = target.with_suffix(target.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(target)
    return target


def download_checkpoints(folder, names=None):
    for name in names or CHECKPOINTS:
        fetch(f"{RELEASE_URL}/{name}", Path(folder) / name)


def download_psc(folder):
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id="kopetri/PSC", repo_type="dataset", local_dir=folder)


def download_sdr(folder):
    folder = Path(folder)
    archive = fetch(f"{RELEASE_URL}/SDR.zip", folder.parent / "SDR.zip")
    with zipfile.ZipFile(archive) as z:
        z.extractall(folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints", action="store_true")
    parser.add_argument("--improved", action="store_true", help="checkpoint of the improved model (improved/)")
    parser.add_argument("--psc", action="store_true")
    parser.add_argument("--sdr", action="store_true")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--data_root", default="data")
    args = parser.parse_args()
    if not (args.checkpoints or args.improved or args.psc or args.sdr):
        parser.error("choose at least one of --checkpoints, --improved, --psc, --sdr")
    if args.checkpoints:
        download_checkpoints(args.checkpoint_dir)
    if args.improved:
        from improved import CHECKPOINT
        download_checkpoints(args.checkpoint_dir, [CHECKPOINT])
    if args.psc:
        download_psc(Path(args.data_root) / "PSC")
    if args.sdr:
        download_sdr(Path(args.data_root) / "SDR")


if __name__ == "__main__":
    main()
