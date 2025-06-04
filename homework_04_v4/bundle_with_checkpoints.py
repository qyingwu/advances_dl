import argparse
import zipfile
from pathlib import Path

BLACKLIST = ["__pycache__", ".pyc", ".ipynb"]
MAXSIZE_MB = 50  # Increased to 50MB as per README


def bundle_with_checkpoints(homework_dir: str, utid: str):
    """
    Usage: python3 bundle_with_checkpoints.py homework <utid>
    """
    homework_dir = Path(homework_dir).resolve()
    output_path = Path(__file__).parent / f"{utid}.zip"

    # Get the files from the homework directory
    files = []

    for f in homework_dir.rglob("*"):
        if all(b not in str(f) for b in BLACKLIST):
            files.append(f)

    # Add checkpoint files
    checkpoint_files = []
    for f in Path(".").rglob("adapter_config.json"):
        checkpoint_files.append(f)
    for f in Path(".").rglob("adapter_model.safetensors"):
        checkpoint_files.append(f)

    print("\n".join(str(f.relative_to(homework_dir)) for f in files))
    print("\nIncluding checkpoint files:")
    print("\n".join(str(f) for f in checkpoint_files))

    # Zip all files, keeping the directory structure
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Add homework files
        for f in files:
            zf.write(f, homework_dir.stem / f.relative_to(homework_dir))
        
        # Add checkpoint files to the homework directory
        for f in checkpoint_files:
            # Create a path that puts the checkpoint files under the homework directory
            arcname = homework_dir.stem / f
            zf.write(f, arcname)

    output_size_mb = output_path.stat().st_size / 1024 / 1024

    if output_size_mb > MAXSIZE_MB:
        print(f"Warning: The created zip file ({output_size_mb:.2f} MB) is larger than the maximum allowed size ({MAXSIZE_MB} MB)!")

    print(f"Submission created: {output_path.resolve()!s} {output_size_mb:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("homework")
    parser.add_argument("utid")

    args = parser.parse_args()

    bundle_with_checkpoints(args.homework, args.utid) 