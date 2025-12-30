#!/usr/bin/env python3

import argparse

from lib.multimodal_search import verify_image_embedding


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build the inverted index")

    verify_image_embedding_cmd = subparsers.add_parser(
        "verify_image_embedding",
        help="Verify an image embedding using clip-ViT-B-32 model",
    )
    verify_image_embedding_cmd.add_argument(
        "img_path", type=str, help="The image path to be verified"
    )

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.img_path)
        case _:
            parser.exit(2, parser.format_help())


if __name__ == "__main__":
    main()
