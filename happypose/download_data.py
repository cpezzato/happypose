"""
Download data and model weights needed to run the tracking example.

Usage:
    # Download everything (models + example data) — recommended first-time setup:
    uv run python examples/download_data.py --all

    # Download individual pieces:
    uv run python examples/download_data.py --cosypose-models
    uv run python examples/download_data.py --megapose-models
    uv run python examples/download_data.py --example-data

    # Custom data directory:
    HAPPYPOSE_DATA_DIR=~/my_data uv run python examples/download_data.py --all
"""

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import gdown

MUSTARD0_FOLDER_ID = "1pRyFmxYXmAnpku7nGRioZaKrVJtIsroP"

COSYPOSE_HOPE_MODELS = [
    "detector-bop-hope-pbr--15246",
    "coarse-bop-hope-pbr--225203",
    "refiner-bop-hope-pbr--955392",
]


def get_data_dir() -> Path:
    env = os.environ.get("HAPPYPOSE_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return Path.home() / "happypose_data"


def run_download_module(*args: str) -> None:
    subprocess.run(
        [sys.executable, "-m", "happypose.toolbox.utils.download", *args],
        check=True,
    )


def download_cosypose_models() -> None:
    print("Downloading CosyPose HOPE model weights ...")
    run_download_module("--cosypose_models", *COSYPOSE_HOPE_MODELS)
    print("CosyPose models done.")


def download_megapose_models() -> None:
    print("Downloading MegaPose model weights ...")
    run_download_module("--megapose_models")
    print("MegaPose models done.")


def download_mustard0(data_dir: Path) -> None:
    dest = data_dir / "mustard0"
    if dest.exists() and any(dest.iterdir()):
        print(f"Already exists: {dest} — skipping download.")
        return

    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading mustard0 example dataset to {data_dir} ...")

    url = f"https://drive.google.com/drive/folders/{MUSTARD0_FOLDER_ID}"
    gdown.download_folder(url, output=str(data_dir), quiet=False, use_cookies=False)

    # Extract only mustard0.zip; skip unrelated or corrupt archives.
    for zf in data_dir.glob("*.zip"):
        if zf.stem != "mustard0":
            print(f"Skipping unrelated archive: {zf.name}")
            continue
        print(f"Extracting {zf} ...")
        try:
            with zipfile.ZipFile(zf) as z:
                z.extractall(data_dir)
            zf.unlink()
        except zipfile.BadZipFile as e:
            print(f"Warning: could not extract {zf.name}: {e}")

    print(f"Example data done. Dataset at: {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download HappyPose data and models.")
    parser.add_argument("--all", action="store_true", help="Download everything (models + example data)")
    parser.add_argument("--cosypose-models", action="store_true", help="Download CosyPose HOPE model weights")
    parser.add_argument("--megapose-models", action="store_true", help="Download MegaPose model weights")
    parser.add_argument("--example-data", action="store_true", help="Download mustard0 example dataset")
    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        sys.exit(1)

    data_dir = get_data_dir()
    print(f"Data directory: {data_dir}\n")

    if args.all or args.cosypose_models:
        download_cosypose_models()
    if args.all or args.megapose_models:
        download_megapose_models()
    if args.all or args.example_data:
        download_mustard0(data_dir)

    print("\nAll downloads complete.")


if __name__ == "__main__":
    main()
