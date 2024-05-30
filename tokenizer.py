import json
import re

import datasets
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from datasets import load_dataset


def batchify(iterable, batch_size=1000):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]["text"]


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


def build_tokenizer():
    tokenizer = Tokenizer(WordLevel(vocab=None, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer


def load_tokenizer():
    return Tokenizer.from_file("../data/tokenizer.json")


def get_data_iterator_from(huggingface_url: str):
    dataset = load_dataset(huggingface_url, cache_dir=".")
    regex = get_regex()
    for entry in dataset["train"]["markdown"]:
        yield re.sub(regex, "", entry).lower()


def train_tokenizer(huggingface_url: str) -> None:
    tokenizer = build_tokenizer()
    trainer = WordLevelTrainer(
        min_frequency=10, special_tokens=["[UNK]"], show_progress=True
    )
    data_iterator = get_data_iterator_from(huggingface_url)

    tokenizer.train_from_iterator(data_iterator, trainer=trainer, length=240000)
    tokenizer.save("./data/tokenizer.json")


if __name__ == "__main__":
    train_tokenizer(huggingface_url="open-phi/textbooks")
