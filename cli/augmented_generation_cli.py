import argparse

from lib.hybrid_search import rrf_search_command
from lib.augmented_generation import augmented_generation, summarize_command


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarise_parser = subparsers.add_parser(
        "summarize", help="Summarization of the results"
    )
    summarise_parser.add_argument(
        "query", type=str, help="Search query for summarization"
    )
    summarise_parser.add_argument(
        "--limit", type=int, help="The limit of returned results", default=5
    )

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
        case "summarize":
            query = args.query
            limit = args.limit

            summary, titles = summarize_command(query, limit)
            print("Search Results:")
            for title in titles:
                print(f"    - {title}")
            print()
            print("LLM Summary:")
            print(f"{summary}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
