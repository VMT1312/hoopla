import argparse

from lib.hybrid_search import rrf_search_command
from lib.augmented_generation import augmented_generation


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            rrf_results = rrf_search_command(query)

            rag_response, titles = augmented_generation(query, rrf_results)

            print("Search Results:")
            for title in titles:
                print(f"    - {title}")
            print()
            print("RAG Response:")
            print(f"{rag_response}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
