"""
download_hf_datasets.py

Downloads sample turn-taking and duplex datasets from Hugging Face
into the data/hf_downloads/ directory for inspection and integration.
"""

import os
import argparse
from huggingface_hub import hf_hub_download, snapshot_download

HF_TARGET_DATASETS = [
    {
        "repo_id": "ASLP-lab/Easy-Turn-Testset",
        "repo_type": "dataset",
        "description": "Standardized evaluation set for turn-taking (complete, incomplete, backchannel, wait)",
    },
    {
        "repo_id": "besimple-ai/duplex-cue",
        "repo_type": "dataset",
        "description": "In-turn adaptation and cue detection dataset for full-duplex agents",
    }
]


def download_dataset(repo_id: str, repo_type: str = "dataset", target_dir: str = "data/hf_downloads"):
    clean_name = repo_id.replace("/", "_")
    dest_path = os.path.join(target_dir, clean_name)
    os.makedirs(dest_path, exist_ok=True)
    
    print(f"\n[HuggingFace Download] Fetching: {repo_id} -> {dest_path}")
    try:
        # Download repository files ignoring large audio binaries if we only need metadata/transcripts
        downloaded_dir = snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            local_dir=dest_path,
            ignore_patterns=["*.wav", "*.mp3", "*.flac", "*.tar.gz"],  # Keep lightweight metadata & text
            max_workers=4
        )
        print(f" Successfully fetched text/metadata to: {downloaded_dir}")
        return downloaded_dir
    except Exception as e:
        print(f" Warning: Could not complete snapshot download for {repo_id}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Download turn-taking datasets from Hugging Face.")
    parser.add_argument("--dest", default="data/hf_downloads", help="Destination folder")
    parser.add_argument("--repo", default=None, help="Specific repo ID to download (optional)")
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    
    if args.repo:
        download_dataset(args.repo, target_dir=args.dest)
    else:
        for ds in HF_TARGET_DATASETS:
            print(f"Target: {ds['repo_id']} ({ds['description']})")
            download_dataset(ds["repo_id"], ds["repo_type"], target_dir=args.dest)


if __name__ == "__main__":
    main()
