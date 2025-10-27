#!/usr/bin/env python3

import argparse, json, os, string
from internal.remove_stop_words import remove_stop_words
from nltk.stem import PorterStemmer


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

    stemmer = PorterStemmer()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            query = args.query.translate(dictionary).lower()
            query = query.split()
            query = remove_stop_words(query)
            for movie in movies["movies"]:
                title = movie["title"].translate(dictionary).lower()
                title = title.split()
                title = remove_stop_words(title)
                for q in query:
                    q = stemmer.stem(q)
                    for token in title:
                        token = stemmer.stem(token)
                        if q in token and movie not in result:
                            result.append(movie)
            result = sorted(result, key=lambda x: x["id"], reverse=False)
            result = result[:6]
            for i, movie in enumerate(result):
                print(f"{i + 1}. {movie["title"]}\n")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
