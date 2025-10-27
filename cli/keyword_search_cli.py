#!/usr/bin/env python3

import argparse
import os
import json
from internal.tokenise import search_movies
from internal.inverted_index import InvertedIndex

with open(os.path.join("data", "movies.json"), encoding="utf-8") as f:
    movies = json.load(f)["movies"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    search_parser = subparsers.add_parser("build", help="Build inverted index")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")

            result = search_movies(args.query, movies)

            result = sorted(result, key=lambda x: x["id"], reverse=False)
            result = result[:6]
            for i, movie in enumerate(result):
                print(f"{i + 1}. {movie["title"]}\n")
        case "build":
            index = InvertedIndex(index=dict(), docmap=dict())
            index.build(movies)
            index.save()

            docs = index.get_documents("merida")

            print(f"First document for token 'merida' = {docs[0]}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
