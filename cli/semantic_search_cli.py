#!/usr/bin/env python3

import argparse
import numpy as np
from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    SemanticSearch,
)
import os
import json
import re


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify the model of semantic search engine")

    embed_text_parser = subparsers.add_parser(
        "embed_text", help="Generate the text embedding of input"
    )
    embed_text_parser.add_argument("text", type=str, help="query to be embedded")

    subparsers.add_parser("verify_embeddings", help="Verify the document embeddings")

    embedquery_parser = subparsers.add_parser(
        "embedquery",
        help="Generate the shape and first 5 dimensions of the query",
    )
    embedquery_parser.add_argument("query", type=str, help="query to be embedded")

    search_parser = subparsers.add_parser(
        "search", help="Search the database for similar meaning"
    )
    search_parser.add_argument("query", type=str, help="Query to search for")
    search_parser.add_argument(
        "--limit", type=int, default=5, help="The limit to return the result"
    )

    chunk_parser = subparsers.add_parser(
        "chunk", help="Help breaking down query into chunks"
    )
    chunk_parser.add_argument(
        "query", type=str, help="Query to be broken down into chunks"
    )
    chunk_parser.add_argument(
        "--chunk-size", type=int, help="Chunk size to break down the query", default=200
    )
    chunk_parser.add_argument(
        "--overlap", type=int, help="The overlap words with the following chunk."
    )

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Break down query into semantic chunks"
    )
    semantic_chunk_parser.add_argument(
        "query", type=str, help="Query to be broken down"
    )
    semantic_chunk_parser.add_argument(
        "--max-chunk-size",
        type=int,
        help="Size of the chunks to be broken down",
        default=4,
    )
    semantic_chunk_parser.add_argument(
        "--overlap", type=int, help="The overlapping words of the chunks", default=0
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()

        case "embed_text":
            embed_text(args.text)

        case "verify_embeddings":
            verify_embeddings()

        case "embedquery":
            embed_query_text(args.query)

        case "search":
            semantic_search = SemanticSearch()

            movies_json_path = os.path.abspath(os.path.join("data", "movies.json"))
            with open(movies_json_path, "r") as f:
                movies = json.load(f)["movies"]

            semantic_search.load_or_create_embeddings(movies)
            results = semantic_search.search(args.query, args.limit)

            for i in range(len(results)):
                score, movie = results[i]

                print(f"{i + 1}. {movie["title"]} (score: {score})")
                print(f"{movie["description"]}")

        case "chunk":
            results = []

            if args.overlap > 0:
                step = args.chunk_size - args.overlap
            else:
                step = args.chunk_size

            if args.overlap > args.chunk_size:
                raise ValueError("Overlap should not be large than chunk size.")

            chunks = args.query.split()
            i = 0
            while i < len(chunks):
                results.append(" ".join(chunks[i : i + args.chunk_size]))
                i += step

            print(f"Chunking {len(args.query)} characters")
            for i, result in enumerate(results):
                print(f"{i + 1}. {result}")

        case "semantic_chunk":
            results = []

            if args.overlap > 0:
                step = args.max_chunk_size - args.overlap
            else:
                step = args.max_chunk_size

            sentences = re.split(r"(?<=[.!?])\s+", args.query)
            i = 0
            while i < len(sentences):
                results.append(" ".join(sentences[i : i + args.max_chunk_size]))
                i += step

            print(f"Semantically chunking {len(args.query)} characters")
            for i, result in enumerate(results):
                print(f"{i + 1}. {result}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
