from torch import where, Tensor
from torch.nn import Embedding
from torch.optim import AdamW, Optimizer
from lightning import LightningModule


class Glove(LightningModule):
    vocab_size: int = None

    def __init__(
        self,
        embedding_dim: int,
        vocab_size: int,
        x_max: int,
        alpha: float,
        learning_rate: float,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.vocab_size = vocab_size

        # weights
        self.input_embedding = Embedding(
            num_embeddings=self.vocab_size, embedding_dim=embedding_dim
        )
        self.output_embedding = Embedding(
            num_embeddings=self.vocab_size, embedding_dim=embedding_dim
        )
        self.input_bias = Embedding(num_embeddings=self.vocab_size, embedding_dim=1)
        self.output_bias = Embedding(num_embeddings=self.vocab_size, embedding_dim=1)

    def _weight_function(self, x) -> Tensor:
        # explicitly clipping the values to 1
        return where(
            x < self.hparams.x_max, (x / self.hparams.x_max) ** self.hparams.alpha, 1.0
        ).clip(max=1.0)

    @staticmethod
    def loss_fn(weights: Tensor, outputs: Tensor, targets: Tensor) -> Tensor:
        loss = weights * (outputs - targets) ** 2
        return loss.mean()

    def forward(
        self, center: str, context: str
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        center_embeds = self.input_embedding(center)
        center_biases = self.input_bias(center)
        context_embeds = self.output_embedding(context)
        context_biases = self.output_bias(context)
        return center_embeds, center_biases, context_embeds, context_biases

    def _shared_step(self, batch: dict[str, str], batch_idx: int | Tensor, stage: str) -> Tensor:
        center = batch["center"]
        context = batch["context"]
        cooccurrences = batch["cooccurrence"]
        center_embed, center_bias, context_embed, context_bias = self.forward(
            center, context
        )

        outputs = (center_embed * context_embed).sum(dim=1) + center_bias + context_bias
        weights = self._weight_function(cooccurrences)
        targets = cooccurrences.log()
        loss = self.loss_fn(weights=weights, outputs=outputs, targets=targets)

        self.log(f"{stage}/loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch: dict[str, str], batch_idx: int | Tensor) -> Tensor:
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch: dict[str, str], batch_idx: int | Tensor) -> Tensor:
        return self._shared_step(batch, batch_idx, "valid")

    def configure_optimizers(self) -> Optimizer:
        return AdamW(self.parameters(), lr=self.hparams.learning_rate)
