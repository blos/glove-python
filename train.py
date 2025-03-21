import multiprocessing as mp

import torch.cuda
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger

from tokenizer import train_tokenizer
from coocurrence import multiprocess_cooc
from dataset import GloveDataset
from glove import Glove


def main():
    # huggingface dataset url
    huggingface_url = "open-phi/textbooks"

    # tokenizer hyperparameter
    min_frequency = 10
    show_progress = True

    # coocurrence hyperparameter
    window_size = 5
    max_docs = 1000
    worker_count = mp.cpu_count() // 4 - 2

    # model hyperparameters
    device = "gpu" if torch.cuda.is_available() else "cpu"  # automatically use the right device
    batch_size = 1024
    learning_rate = 3e-4  # 5e-2
    embedding_dim = 300
    x_max = 100
    alpha = 3 / 4
    vocab_size = 30000

    train_tokenizer(
        huggingface_url=huggingface_url,
        min_frequency=min_frequency,
        show_progress=show_progress
    )
    multiprocess_cooc(                 # or use: singleprocess_cooc
        huggingface_url=huggingface_url,
        window_size=window_size,
        max_docs=max_docs,
        worker_count=worker_count
    )
    dm = GloveDataset(
        batch_size=batch_size,
    )
    dm.setup("fit")
    limit_train_batches = dm.get_max_batches()

    model = Glove(
        learning_rate=learning_rate,
        embedding_dim=embedding_dim,
        x_max=x_max,
        alpha=alpha,
        vocab_size=vocab_size,
    )
    logger = TensorBoardLogger(save_dir=".", name="lightning_logs")
    trainer = Trainer(
        accelerator=device,
        precision="16-mixed",
        inference_mode=True,
        max_epochs=50,
        log_every_n_steps=100,
        accumulate_grad_batches=1,
        limit_train_batches=limit_train_batches,
        logger=logger,
        fast_dev_run=False,
    )
    trainer.fit(model, dm.train_dataloader())


if __name__ == "__main__":
    main()
