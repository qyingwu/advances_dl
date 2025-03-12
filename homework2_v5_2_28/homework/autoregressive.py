import abc
import torch
import torch.nn as nn
import torch.nn.functional as F


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


class Autoregressive(abc.ABC):
    """
    Base class for all autoregressive models.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Take a tensor x (B, h, w) of integers as input.
        Produce a probability over the next token as output (B, h, w, n_token).
        """

    @abc.abstractmethod
    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:
        """
        Generate B new token images of size (B, h, w) of type int/long.
        """


class AutoregressiveModel(nn.Module, Autoregressive):
    """
    Implements an auto-regressive model using a Transformer decoder-only architecture.
    - The input is a set of patch tokens (integers).
    - The output is a probability distribution over the next token.
    - Uses a causal mask to prevent future information leakage.

    Hint: `torch.nn.Embedding` for token representation.
    Hint: `torch.nn.TransformerEncoderLayer` with a causal mask.
    Hint: Input must be shifted by one position in the forward pass.
    """

    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10, num_layers: int = 6):
        super().__init__()
        self.d_latent = d_latent
        self.n_tokens = n_tokens
        self.token_embedding = nn.Embedding(n_tokens, d_latent)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_latent,
            nhead=8,
            dim_feedforward=4 * d_latent,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_latent, n_tokens)

    def _generate_causal_mask(self, seq_len: int, device) -> torch.Tensor:
        """
        Generate a causal mask so that each token can only attend to previous tokens.
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Forward pass for training.
        - x: (B, h, w) or (B, 1, h, w) of token indices.
        - Returns (B, h, w, n_tokens) probability logits for the next token.
        """
        if len(x.shape) == 4:
            x = x.squeeze(1)
        
        B, h, w = x.shape
        x = x.float()
        x = x.view(B, h * w)
        x = self.token_embedding(x.long())

        causal_mask = self._generate_causal_mask(h * w, x.device)
        x = self.transformer(x, mask=causal_mask)

        logits = self.output_proj(x) 
        logits = logits.view(B, h, w, self.n_tokens)

        return logits, {}

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:
        """Generate new images by sequentially sampling from the model."""
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval()

        tokens = torch.zeros(B, h * w, dtype=torch.long, device=device)

        for i in range(h * w):
            with torch.no_grad():
                tokens_reshaped = tokens.view(B, h, w) 
                logits, _ = self.forward(tokens_reshaped)
                logits = logits.view(B, -1, self.n_tokens)
                
                next_token = torch.argmax(logits[:, i], dim=-1) 
                tokens[:, i] = next_token

        return tokens.view(B, h, w)

