#!/usr/bin/env python3

import argparse, json, os, string


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    with open(os.path.join("data", "movies.json"), encoding="utf-8") as f:
        movies = json.load(f)

    result = []

    dictionary = str.maketrans(string.punctuation, " " * len(string.punctuation))

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            query = args.query.translate(dictionary)
            for movie in movies["movies"]:
                title = movie["title"].translate(dictionary)
                title = " ".join(title.split())
                if query.lower() in title.lower():
                    result.append(movie)
            result = result[:6]
            result = sorted(result, key=lambda x: x["id"], reverse=True)
            for i, movie in enumerate(result):
                print(f"{i + 1}. {movie["title"]}\n")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
