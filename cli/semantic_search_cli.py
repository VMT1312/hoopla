#!/usr/bin/env python3

import argparse
from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    SemanticSearch,
    chunking,
    semantic_chunking,
    embed_chunks,
    search_chunked,
)
import os
import json


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

    subparsers.add_parser("embed_chunks", help="Build or load the chunked embeddings")

    search_chunked_parser = subparsers.add_parser(
        "search_chunked", help="Searching query by semantic chunking"
    )
    search_chunked_parser.add_argument("query", type=str, help="Query to search for")
    search_chunked_parser.add_argument(
        "--limit", type=int, help="The numnber of results to be printed", default=5
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
            chunking(args.query, args.chunk_size, args.overlap)

        case "semantic_chunk":
            results = semantic_chunking(args.query, args.max_chunk_size, args.overlap)
            print(f"Semantically chunking {len(args.query)} characters")
            print(results)
            for i, result in enumerate(results):
                print(f"{i + 1}. {result}")

        case "embed_chunks":
            embed_chunks()
            print("Generated 72909 chunked embeddings")

        case "search_chunked":
            search_chunked(args.query, args.limit)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
