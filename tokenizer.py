import warnings
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

from utils import get_data_iterator_from


tokenizer_save_path: Path = Path(__file__).parent / "./data/tokenizer.json"


def build_tokenizer():
    tokenizer = Tokenizer(WordLevel(vocab=None, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer


def load_tokenizer():
    try:
        return Tokenizer.from_file("./data/tokenizer.json")
    except OSError:
        warnings.warn("Error while loading the tokenizer file. Are you sure this is the right path? Did you train the tokenizer already?")


def train_tokenizer(huggingface_url: str, min_frequency: int = 10, show_progress: bool = True, force_retrain: bool = False) -> None:
    if tokenizer_save_path.exists() and not force_retrain:
        return

    tokenizer = build_tokenizer()
    trainer = WordLevelTrainer(
        min_frequency=min_frequency, special_tokens=["[UNK]"], show_progress=show_progress
    )
    data_iterator = get_data_iterator_from(huggingface_url)

    tokenizer.train_from_iterator(data_iterator, trainer=trainer, length=240000)
    tokenizer.save("./data/tokenizer.json")


if __name__ == "__main__":
    train_tokenizer(huggingface_url="open-phi/textbooks")
