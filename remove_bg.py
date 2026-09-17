#!/usr/bin/env python3
"""remove image backgrounds using BRIA RMBG v2.0 (BiRefNet). https://huggingface.co/briaai/RMBG-2.0

usage:
    rmbg photo.jpg                    # → photo_no_bg.png (same folder)
    rmbg photo.jpg output.png         # → custom output path
    rmbg photo.jpg -o ~/Desktop/      # → redirect output folder
    rmbg photo.jpg --open             # → auto-open result in Preview
    rmbg photos/                      # → batch process folder

requires:
    pip install torch torchvision pillow kornia timm transformers huggingface_hub
    huggingface-cli login  (RMBG 2.0 is a gated model)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

MODEL_ID = "briaai/RMBG-2.0"
IMAGE_SIZE = (1024, 1024)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".heic"}

TRANSFORM_IMAGE = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# hardware
def select_device(requested: str | None = None) -> torch.device:
    """resolve compute device. explicit *requested* overrides auto-detect."""
    if requested:
        name = requested.lower()
        if name == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if name == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if name == "cpu":
            return torch.device("cpu")
        print(f"warning: requested device '{requested}' unavailable, auto-detecting.")

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# model
def load_model(device: torch.device) -> torch.nn.Module:
    """download (cached after first run) and load RMBG 2.0 onto *device*."""
    torch.set_float32_matmul_precision("high")
    model = AutoModelForImageSegmentation.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )
    model.to(device).eval()
    return model

# core pipeline
def predict_mask(
    model: torch.nn.Module,
    image: Image.Image,
    device: torch.device,
) -> Image.Image:
    """return an alpha-mask PIL image at the original resolution."""
    input_tensor = TRANSFORM_IMAGE(image).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(input_tensor)[-1].sigmoid().cpu()

    pred = preds[0].squeeze()
    mask_pil = transforms.ToPILImage()(pred)
    return mask_pil.resize(image.size, Image.BILINEAR)


def apply_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    """composite *mask* as the alpha channel of *image* (returns RGBA)."""
    rgba = image.convert("RGBA")
    rgba.putalpha(mask)
    return rgba

# path resolution
def resolve_output_path(
    input_path: Path,
    output_arg: Path | None,
    output_dir: Path | None,
    suffix: str,
    overwrite: bool,
) -> Path:
    """determine where to write the result PNG."""
    if overwrite:
        return input_path.with_suffix(".png")
    if output_arg:
        return output_arg
    parent = output_dir or input_path.parent
    return parent / f"{input_path.stem}{suffix}.png"


def resolve_image_paths(source: Path) -> list[Path]:
    """return image paths from a file or directory."""
    if source.is_file():
        return [source]
    if source.is_dir():
        return sorted(
            p for p in source.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
    return []

# process
def process_single(
    input_path: Path,
    output_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    quiet: bool,
) -> float:
    """remove background from one image. Returns elapsed seconds."""
    image = Image.open(input_path).convert("RGB")
    t0 = time.perf_counter()
    mask = predict_mask(model, image, device)
    result = apply_mask(image, mask)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path, "PNG")
    elapsed = time.perf_counter() - t0
    if not quiet:
        print(f"  {input_path.name} → {output_path.name}  ({elapsed:.1f}s)")
    return elapsed


def process_batch(
    source: Path,
    output_dir: Path,
    model: torch.nn.Module,
    device: torch.device,
    suffix: str,
    quiet: bool,
) -> None:
    """process every image in *source* (file or folder)."""
    paths = resolve_image_paths(source)
    if not paths:
        print(f"no images found in {source}")
        sys.exit(1)

    total = len(paths)
    if not quiet:
        print(f"processing {total} image(s)…\n")

    for idx, img_path in enumerate(paths, 1):
        out = resolve_output_path(img_path, None, output_dir, suffix, False)
        elapsed = process_single(img_path, out, model, device, quiet)
        if not quiet:
            print(f"  [{idx}/{total}] {img_path.name} → {out.name}  ({elapsed:.1f}s)")

    if not quiet:
        print(f"\ndone — {total} image(s) saved to {output_dir}")


def open_file(path: Path) -> None:
    """Open *path* with the default macOS application."""
    subprocess.run(["open", str(path)], check=False)

# cli
SUPPORTED_FORMATS = ", ".join(sorted(IMAGE_EXTENSIONS))

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rmbg",
        description="remove image backgrounds with BRIA RMBG v2.0.",
        epilog=f"supported formats: {SUPPORTED_FORMATS}",
    )
    parser.add_argument(
        "source",
        nargs="?",
        type=Path,
        default=None,
        help="input image or folder.",
    )
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=None,
        help="output path (single image) or output folder (batch).",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="output directory (overrides default).",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        dest="auto_open",
        help="open result in Preview after processing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace the original file (converts to .png).",
    )
    parser.add_argument(
        "--suffix",
        default="_no_bg",
        metavar="SFX",
        help="suffix appended to filename (default: _no_bg).",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="suppress all output except errors.",
    )
    parser.add_argument(
        "--device",
        choices=["mps", "cuda", "cpu"],
        default=None,
        help="force compute device (default: auto-detect).",
    )
    parser.add_argument(
        "--formats",
        action="store_true",
        help="print supported formats and exit.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.formats:
        print(SUPPORTED_FORMATS)
        return

    if args.source is None:
        parser.error("source is required (unless --formats is used)")

    if not args.source.exists():
        print(f"error: {args.source} does not exist.")
        sys.exit(1)

    device = select_device(args.device)
    if not args.quiet:
        print(f"device: {device.type.upper()}")

    if not args.quiet:
        print(f"loading {MODEL_ID}…")
    model = load_model(device)
    if not args.quiet:
        print("model loaded.\n")

    # single file with explicit output path
    if args.source.is_file() and args.output:
        elapsed = process_single(args.source, args.output, model, device, args.quiet)
        if not args.quiet:
            print(f"saved to {args.output} ({elapsed:.1f}s)")
        if args.auto_open:
            open_file(args.output)
        return

    # single file, auto output name
    if args.source.is_file():
        out = resolve_output_path(args.source, None, args.output_dir, args.suffix, args.overwrite)
        elapsed = process_single(args.source, out, model, device, args.quiet)
        if not args.quiet:
            print(f"saved to {out} ({elapsed:.1f}s)")
        if args.auto_open:
            open_file(out)
        return

    # Batch mode
    out_dir = args.output_dir or args.source / "output"
    process_batch(args.source, out_dir, model, device, args.suffix, args.quiet)


if __name__ == "__main__":
    main()
