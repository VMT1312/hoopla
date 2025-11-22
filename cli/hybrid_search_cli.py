import argparse
from lib.hybrid_search import (
    normalize,
    hybrid_score_command,
    rrf_score_command,
    enhanced_spell_rrf,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    normalise_parser = subparser.add_parser(
        "normalize", help="Score normalization using min-max"
    )
    normalise_parser.add_argument(
        "scores", nargs="+", type=float, help="List of scores to be normalized"
    )

    weighted_search_parser = subparser.add_parser(
        "weighted-search",
        help="Search a query based on the weighting between keyword and semantic",
    )
    weighted_search_parser.add_argument("query", type=str, help="Query to search for")
    weighted_search_parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Constant to conrol the weighting between keyword and semantic",
    )
    weighted_search_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return"
    )

    rrf_search_parser = subparser.add_parser(
        "rrf-search",
        help="Peciprocal Rank Fusion of keyword and semantic search scores",
    )
    rrf_search_parser.add_argument("query", type=str, help="Query to search for")
    rrf_search_parser.add_argument(
        "--k", type=int, default=60, help="Constant weight k"
    )
    rrf_search_parser.add_argument(
        "--limit", type=int, default=5, help="The returned results limit"
    )
    rrf_search_parser.add_argument(
        "--enhance", type=str, choices=["spell"], help="Query enhancement method"
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize(args.scores)

        case "weighted-search":
            hybrid_score_command(args.query, args.alpha, args.limit)

        case "rrf-search":
            if args.enhance == "spell":
                enhanced_spell_rrf(args.query, args.k, args.limit)
            else:
                rrf_score_command(args.query, args.k, args.limit)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
