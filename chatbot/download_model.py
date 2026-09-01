"""Download one of the chatbot's GGUF model files into /Users/vuk/projects/models.

Usage:
    python download_model.py                    # downloads the default model
    python download_model.py --model llama3.1-8b
    python download_model.py --list-models
"""
import argparse

from huggingface_hub import hf_hub_download

import chatbot_config as cfg


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a local GGUF model for the chatbot.")
    parser.add_argument(
        "--model",
        choices=list(cfg.AVAILABLE_MODELS),
        default=cfg.DEFAULT_MODEL,
        help=f"Which model to download (default: {cfg.DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print available --model choices and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.list_models:
        for key, info in cfg.AVAILABLE_MODELS.items():
            marker = " (default)" if key == cfg.DEFAULT_MODEL else ""
            print(f"  {key}{marker} — {info['description']}")
        return

    info = cfg.model_info(args.model)
    if info["provider"] != "local":
        env_var = cfg.PROVIDER_API_KEY_ENV[info["provider"]]
        print(f"'{args.model}' is a cloud model ({info['provider']}) — nothing to download.")
        print(f"Set {env_var} instead, then run: python main.py --model {args.model}")
        return

    cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=info["repo_id"],
        filename=info["filename"],
        local_dir=str(cfg.MODELS_DIR),
    )
    print(f"Downloaded to: {path}")


if __name__ == "__main__":
    main()
