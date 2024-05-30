from typing import Generator, Any
from re import sub

from datasets import load_dataset


def batchify(iterable, batch_size=1000):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i: i + batch_size]["text"]


def concat_iterables(*iterables: list[Generator]) -> Any:
    for iterable in iterables:
        yield from iterable


def get_stopwords_from(filepath: str) -> [str]:
    stopwords = list()
    with open(filepath, "r", encoding="utf8") as file:
        for line in file:
            stopwords.append(line.strip())
    return stopwords


def get_regex_from(stopwords: [str]) -> str:
    return r"\b(" + r"|".join(stopwords) + r")\b"


def get_regex():
    stopwords = get_stopwords_from("./data/stopwords.txt")
    stopwords_regex = get_regex_from(stopwords)
    special_character_regex = (
        r"[^\s\w]+"  # all characters that are no whitespaces or word characters
    )
    digit_regex = r"[^\D]+"  # all digits
    regex = rf"({stopwords_regex}|{special_character_regex}|{digit_regex})"
    return regex


def get_data_iterator_from(huggingface_url: str):
    dataset = load_dataset(huggingface_url, cache_dir=".")
    regex = get_regex()
    for entry in dataset["train"]["markdown"]:
        yield sub(regex, "", entry).lower()