import argparse
import mimetypes

from lib.describe_image import describe_image_command


def main():
    parser = argparse.ArgumentParser(description="Multimodal query rewriting cli")
    parser.add_argument("--image", type=str, help="The path to an image file")
    parser.add_argument(
        "--query", type=str, help="A text query to rewrite based on the image"
    )

    args = parser.parse_args()

    image = args.image
    query = args.query

    describe_image_command(image, query)


if __name__ == "__main__":
    main()
