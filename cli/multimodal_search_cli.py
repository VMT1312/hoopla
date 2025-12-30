#!/usr/bin/env python3

import argparse

from lib.multimodal_search import verify_image_embedding, image_search_command


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

    image_search_cmd = subparsers.add_parser(
        "image_search", help="Search the database using an image"
    )
    image_search_cmd.add_argument(
        "img_path", type=str, help="The path to the image to be used for searching"
    )

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.img_path)
        case "image_search":
            results = image_search_command(args.img_path)
            for i, result in enumerate(results):
                print(
                    f"{i + 1}. {result["title"]} (similarity: {result["similarity_score"]})"
                )
                print(f"   {result["description"]}")
                print()
        case _:
            parser.exit(2, parser.format_help())


if __name__ == "__main__":
    main()
