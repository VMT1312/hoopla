import os

with open(os.path.join("data", "stopwords.txt"), encoding="utf-8") as f:
    stopwords = f.read().splitlines()


def remove_stop_words(tokens: list[str]) -> list[str]:
    return [token for token in tokens if token not in stopwords]
