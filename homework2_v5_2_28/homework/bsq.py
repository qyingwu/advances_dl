import abc

import torch

from .ae import PatchAutoEncoder


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "BSQPatchAutoEncoder"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


def diff_sign(x: torch.Tensor) -> torch.Tensor:
    """
    A differentiable sign function using the straight-through estimator.
    Returns -1 for negative values and 1 for non-negative values.
    """
    sign = 2 * (x >= 0).float() - 1
    return x + (sign - x).detach()


class Tokenizer(abc.ABC):
    """
    Base class for all tokenizers.
    Implement a specific tokenizer below.
    """

    @abc.abstractmethod
    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tokenize an image tensor of shape (B, H, W, C) into
        an integer tensor of shape (B, h, w) where h * patch_size = H and w * patch_size = W
        """

    @abc.abstractmethod
    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a tokenized image into an image tensor.
        """


class BSQ(torch.nn.Module):
    def __init__(self, codebook_bits: int, embedding_dim: int):
        super().__init__()
        self._codebook_bits = codebook_bits
        self.down_proj = torch.nn.Linear(embedding_dim, codebook_bits)
        self.up_proj = torch.nn.Linear(codebook_bits, embedding_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement the BSQ encoder:
        - A linear down-projection into codebook_bits dimensions
        - L2 normalization
        - differentiable sign
        """
        x = self.down_proj(x)
        # L2 Normalization
        x = x / (torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-6)
        x = diff_sign(x)
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement the BSQ decoder:
        - A linear up-projection into embedding_dim should suffice
        """
        return self.up_proj(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run BQS and encode the input tensor x into a set of integer tokens
        """
        return self._code_to_index(self.encode(x))

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a set of integer tokens into an image.
        """
        return self.decode(self._index_to_code(x))

    def _code_to_index(self, x: torch.Tensor) -> torch.Tensor:
        x = (x >= 0).int()
        return (x * (2 ** torch.arange(x.size(-1), device=x.device))).sum(dim=-1)
    
    def _index_to_code(self, x: torch.Tensor) -> torch.Tensor:
        return (2 * ((x[..., None] & (2 ** torch.arange(self._codebook_bits).to(x.device))) > 0).float() - 1)


class BSQPatchAutoEncoder(PatchAutoEncoder, Tokenizer):
    """
    Combine your PatchAutoEncoder with BSQ to form a Tokenizer.

    Hint: The hyper-parameters below should work fine, no need to change them
          Changing the patch-size of codebook-size will complicate later parts of the assignment.
    """

    def __init__(self, patch_size: int = 5, latent_dim: int = 128, codebook_bits: int = 10):
        super().__init__(patch_size=patch_size, latent_dim=latent_dim)
        self.codebook_bits = codebook_bits
        self.bsq = BSQ(codebook_bits, latent_dim)

    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        """Convert image to tokens"""
        features = self.encoder(x)
        quantized = self.bsq.encode(features)
        return self.bsq._code_to_index(quantized)

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        """Convert tokens back to image"""
        quantized = self.bsq._index_to_code(x)
        features = self.bsq.decode(quantized)
        return self.decoder(features)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to binary features"""
        features = self.encoder(x)
        return self.bsq.encode(features)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode binary features to image"""
        features = self.bsq.decode(x)
        return self.decoder(features)


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Forward pass with monitoring of codebook usage
        """
        encoded = self.encode(x)
        decoded = self.decode(encoded)

        # Monitor codebook usage for debug  issues, 
        tokens = self.encode_index(x)
        cnt = torch.bincount(tokens.flatten(), minlength=2**self.codebook_bits)
        token_probs = cnt / (cnt.sum() + 1e-6)
        entropy_loss = -torch.sum(token_probs * torch.log(token_probs + 1e-6))  # Shannon entropy

        return decoded, {
            "cb0": (cnt == 0).float().mean(),  # Unused tokens
            "cb2": (cnt <= 2).float().mean(),  # Rarely used
            "cb10": (cnt <= 10).float().mean(), # Very rarely used
            "entropy_loss": entropy_loss
        }