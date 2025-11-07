#!/usr/bin/env python3

import argparse
import numpy as np
from lib.semantic_search import verify_model, embed_text, verify_embeddings


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify the model of semantic search engine")

    embed_text_parser = subparsers.add_parser(
        "embed_text", help="Generate the text embedding of input"
    )
    embed_text_parser.add_argument("text", type=str, help="query to be embedded")

    subparsers.add_parser("verify_embeddings", help="Verify the document embeddings")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()

        case "embed_text":
            embed_text(args.text)

        case "verify_embeddings":
            verify_embeddings()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
