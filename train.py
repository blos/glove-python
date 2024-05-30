from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger

from dataset import GloveDataset
from glove import Glove


def main():
    device = "gpu"
    batch_size = 1024
    learning_rate = 3e-4  # 5e-2
    embedding_dim = 300
    x_max = 100
    alpha = 3 / 4
    vocab_size = 30000

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


if __name__ == '__main__':
    main()
