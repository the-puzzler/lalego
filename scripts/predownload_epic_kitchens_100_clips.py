from huggingface_hub import snapshot_download


def main() -> None:
    snapshot_download(
        repo_id="lightly-ai/epic-kitchens-100-clips",
        repo_type="dataset",
        local_dir="data/epic_kitchens_100_clips_full",
    )


if __name__ == "__main__":
    main()
