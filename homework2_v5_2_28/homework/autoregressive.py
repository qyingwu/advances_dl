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
    Implement a specific model below.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Take a tensor x (B, h, w) if integers as input.
        Produce a probability over the next token as an output (B, h, w, n_token).
        Make sure the model is auto-regressive:
          - The first output result[:, 0, 0] does not depend on any input
          - The second output result[:, 0, 1] depends only on x[:, 0, 0]
          - etc.

        Hint 1: Flatten the tensor into a sequence.
        Hint 2: A positional embedding can help, but is not required.
        Hint 3: You need to shift the input sequence by 1 position. Do this after embedding the
                values, and before passing them through your model. (torch.concat or
                torch.nn.ConstantPad1d both work)
        """

    @abc.abstractmethod
    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:
        """
        Use your generative model to produce B new token images of size (B, h, w) and type (int/long).
        """


class AutoregressiveModel(nn.Module, Autoregressive):
    """
    Implement an auto-regressive model.
    The input is a set of patch tokens (integers), the output is an image of probability.
    You need to implicitly shift your inputs by one position in the forward pass.
    Make sure n_tokens matches your BSQ dimension (2**codebook_bits_).

    Hint: You will need the torch.nn.Embedding function
    Hint: You can use torch.nn.TransformerEncoderLayer if you'd like
    Hint: You can complete this homework without using positional embeddings
    """

    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10, num_layers: int = 6, use_pos_emb=True):
        super().__init__()
        self.d_latent = d_latent
        self.n_tokens = n_tokens
        self.use_pos_emb = use_pos_emb
        self.token_embedding = nn.Embedding(n_tokens, d_latent)

        # Positional Encoding
        if use_pos_emb:
            self.pos_embedding = nn.Parameter(torch.randn(1, 100 * 150, d_latent) * 0.02)

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
        """causal mask so that each token can only attend to prev tokens."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Forward pass for training.
        - x: (B, h, w) or (B, 1, h, w) of token indices.
        - Returns (B, h, w, n_tokens) probability logits for the next token.
        """
        # Handle extra channel dimension, encountered this error while generating
        if len(x.shape) == 4:  # (B, 1, h, w) =>(B, h, w)
            x = x.squeeze(1) 

        if len(x.shape) != 3:
            raise ValueError(f"Expected input shape (B, h, w) or (B, 1, h, w), but got {x.shape}")

        B, h, w = x.shape
        seq_len = h * w
        # flatten image into a sequence first. 
        x = x.view(B, seq_len)  

        # Token embedding
        x = self.token_embedding(x)  # (B, seq_len, d_latent)

        # Add positional encoding if enabled
        if self.use_pos_emb:
            x += self.pos_embedding[:, :seq_len, :]

        causal_mask = self._generate_causal_mask(seq_len, x.device)

        x = self.transformer(x, mask=causal_mask)

        # Project to token space
        logits = self.output_proj(x)  # (B, seq_len, n_tokens)
        logits = logits.view(B, h, w, self.n_tokens)  # Reshape 

        return logits, {}


    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:
        """
        Generate new images by sequentially sampling from the model.
        
        Args:
            B: Batch size
            h: Height of generated image in tokens
            w: Width of generated image in tokens
            device: Device to generate on
        Returns:
            Tensor of shape (B, h, w) containing generated token indices
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval()

        # Init rand tokens
        tokens = torch.randint(0, self.n_tokens, (B, h * w), dtype=torch.long, device=device)

        for i in range(h * w):
            with torch.no_grad():
      
                tokens_reshaped = tokens.view(B, h, w)
                logits, _ = self.forward(tokens_reshaped)
                logits = logits.view(B, -1, self.n_tokens)

                # sampling strategy (Top-K, Temperature)
                next_token = self._sample(logits[:, i], temperature=0.9, top_k=10)
                tokens[:, i] = next_token

        return tokens.view(B, h, w)

    def _sample(self, logits, temperature=1.0, top_k=10):
        """
        Sample using temperature scaling and top-k filtering.
        """
        if temperature > 0:
            logits = logits / temperature
        if top_k > 0:
            values, _ = torch.topk(logits, top_k)
            min_value = values[:, -1].unsqueeze(-1)
            logits[logits < min_value] = float('-inf')
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
