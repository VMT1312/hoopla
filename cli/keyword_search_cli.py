#!/usr/bin/env python3

import argparse
import os
import json
from internal.keyword_search import tokenise, InvertedIndex, search_movies

with open(os.path.join("data", "movies.json"), encoding="utf-8") as f:
    movies = json.load(f)["movies"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build inverted index")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a document")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to get frequency for")

    args = parser.parse_args()

    index = InvertedIndex(index=dict(), docmap=dict(), term_frequencies=dict())

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")

            try:
                index.load()
            except FileNotFoundError:
                print(
                    "Inverted index not found. Please build the index first using the 'build' command."
                )
                return

            results = search_movies(args.query, index)

            for movie in results:
                print(f"movie id: {movie["id"]}; movie title: {movie["title"]}\n")

        case "build":
            index.build(movies)
            index.save()

        case "tf":
            try:
                index.load()
            except FileNotFoundError:
                print(
                    "Inverted index not found. Please build the index first using the 'build' command."
                )
                return

            tf = index.get_tf(args.doc_id, args.term)
            print(tf)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
