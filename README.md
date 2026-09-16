# rmbg

remove backgrounds from images. locally. no uploads. no accounts. just vibes.

<table>
  <tr>
    <td><strong>original</strong></td>
    <td><strong>background removed</strong></td>
  </tr>
  <tr>
    <td><img src="preview_original.jpg" width="400" /></td>
    <td><img src="preview_no_bg.png" width="400" /></td>
  </tr>
</table>

uses [bria rmbg v2.0](https://huggingface.co/briaai/RMBG-2.0) (birefnet) running on apple silicon mps.

## setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision pillow kornia timm transformers huggingface_hub
huggingface-cli login
```

## usage

```bash
rmbg photo.jpg                    # → photo_no_bg.png
rmbg photo.jpg output.png         # → custom path
rmbg photo.jpg -o ~/Desktop/      # → output folder
rmbg photo.jpg --open             # → open in preview
rmbg photo.jpg --overwrite        # → replace original
rmbg photos/                      # → batch mode
rmbg --help                       # → all flags
```

## requirements

- python 3.10+
- apple silicon mac (mps) or cuda gpu
- ~900mb disk for model weights (auto-cached after first run)

## how it works

1. loads bria rmbg 2.0 (birefnet architecture)
2. preprocesses image to 1024x1024
3. runs inference on mps/cuda/cpu
4. generates alpha mask
5. composites transparent png

takes ~1-3 seconds per image on mps.
