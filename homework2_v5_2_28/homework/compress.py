from pathlib import Path
from typing import cast
import numpy as np
import torch
from PIL import Image

from .autoregressive import Autoregressive
from .bsq import Tokenizer


class Compressor:
    def __init__(self, tokenizer: Tokenizer, autoregressive: Autoregressive):
        super().__init__()
        self.tokenizer = tokenizer
        self.autoregressive = autoregressive

    def compress(self, x: torch.Tensor) -> bytes:
        """
        Compress the image into bytes using tokenization.
        Simple implementation: convert tokens to bytes directly.
        """
        # Ensure input is in correct format (B, C, H, W)
        if len(x.shape) == 3:  # (H, W, C)
            x = x.permute(2, 0, 1).unsqueeze(0)  # -> (1, C, H, W)
        
        # Tokenize the image
        with torch.no_grad():
            tokens = self.tokenizer.encode_index(x)  # (1, h, w)
            
        # Convert to bytes
        return tokens.cpu().numpy().tobytes()

    def decompress(self, x: bytes) -> torch.Tensor:
        """
        Decompress bytes back into an image.
        """
        # Convert bytes back to tokens
        tokens = np.frombuffer(x, dtype=np.int64)
        h, w = 20, 30  # Fixed dimensions for this dataset
        tokens = torch.tensor(tokens, dtype=torch.long).reshape(1, h, w)
        
        # Convert tokens back to image
        with torch.no_grad():
            x = self.tokenizer.decode_index(tokens)  # (1, C, H, W)
            x = x.squeeze(0)  # Remove batch dimension -> (C, H, W)
            x = x.permute(1, 2, 0)  # -> (H, W, C)
            return x


def compress(tokenizer: Path, autoregressive: Path, image: Path, compressed_image: Path):
    """Compress an image using a pre-trained tokenizer and autoregressive model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tk_model = cast(Tokenizer, torch.load(tokenizer, weights_only=False).to(device))
    ar_model = cast(Autoregressive, torch.load(autoregressive, weights_only=False).to(device))
    cmp = Compressor(tk_model, ar_model)

    x = torch.tensor(np.array(Image.open(image)), dtype=torch.uint8, device=device)
    cmp_img = cmp.compress(x.float() / 255.0 - 0.5)
    
    with open(compressed_image, "wb") as f:
        f.write(cmp_img)


def decompress(tokenizer: Path, autoregressive: Path, compressed_image: Path, image: Path):
    """Decompress an image using a pre-trained tokenizer and autoregressive model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tk_model = cast(Tokenizer, torch.load(tokenizer, weights_only=False).to(device))
    ar_model = cast(Autoregressive, torch.load(autoregressive, weights_only=False).to(device))
    cmp = Compressor(tk_model, ar_model)

    with open(compressed_image, "rb") as f:
        cmp_img = f.read()

    x = cmp.decompress(cmp_img)  # Now returns (H, W, C)
    img = Image.fromarray(((x + 0.5) * 255.0).clamp(0, 255).byte().cpu().numpy())
    img.save(image)


if __name__ == "__main__":
    from fire import Fire
    Fire({"compress": compress, "decompress": decompress})
