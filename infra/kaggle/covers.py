"""Cover images for Bulwark's Kaggle datasets: infra/kaggle/covers/<slug>.png.

    python infra/kaggle/covers.py            # render every cover
    python infra/kaggle/covers.py --show     # also write each 280x280 thumbnail crop to /tmp

Kaggle crops a dataset's cover at fixed rectangles (kaggle_api_extended
`_upload_dataset_image_file`): the header is the whole 560x280 image and the thumbnail is
x 140..420, so every cover is exactly 560x280 and keeps its text inside that middle square.
The motif is abstract (a row or grid of frame tiles standing for a 16-frame clip); no
frame of any dataset and no image of a person is ever drawn. The committed PNGs are what
the cards use; rerun this script only to change them. Fonts: DejaVu Sans from
matplotlib's bundled fonts when it is installed, else Pillow's default font.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent / "covers"
SIZE = (560, 280)
THUMB = (140, 0, 420, 280)  # Kaggle's thumbnail crop: left, top, right, bottom
SCALE = 4  # drawn at 4x and downsampled, for smooth edges

INK = "#000000"
PAPER = "#F5F4F0"
SIGNAL = "#FF5A1F"  # the one accent
GRAPHITE = "#575757"
COBALT = "#2F6BFF"
MIST = "#B6C9CF"

Font = ImageFont.FreeTypeFont | ImageFont.ImageFont


def font(size: int, bold: bool = False) -> Font:
    try:
        import matplotlib

        name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / name
        return ImageFont.truetype(str(path), size * SCALE)
    except (ImportError, OSError):
        return ImageFont.load_default(size * SCALE)


# slug -> (dataset name, descriptor, motif)
COVERS: dict[str, tuple[str, str, str]] = {
    "bulwark-drive-and-act": ("Drive&Act", "training shards", "views"),
    "bulwark-dmd": ("DMD", "training shards", "clip"),
    "bulwark-mrl": ("MRL Eye", "training shards", "stills"),
    "bulwark-auc": ("AUC v2", "training shards", "stills"),
    "bulwark-meta": ("Manifest", "clip manifest & checkpoints", "table"),
}


def _box(draw: ImageDraw.ImageDraw, x: float, y: float, w: float, h: float, fill: str) -> None:
    draw.rectangle([x * SCALE, y * SCALE, (x + w) * SCALE - 1, (y + h) * SCALE - 1], fill=fill)


def _strip(
    draw: ImageDraw.ImageDraw, y: float, h: float, colours: list[str], gap: float = 3
) -> None:
    """One tile per colour (16: a clip) across the full width, with 24 px margins."""
    left, right, n = 24, SIZE[0] - 24, len(colours)
    w = (right - left - gap * (n - 1)) / n
    for i, colour in enumerate(colours):
        _box(draw, left + i * (w + gap), y, w, h, colour)


def _motif(draw: ImageDraw.ImageDraw, kind: str) -> None:
    if kind == "clip":  # one face camera: a single 16-frame clip, the 12th frame marked
        _strip(
            draw,
            200,
            44,
            [SIGNAL if i == 11 else (COBALT if i % 4 == 3 else MIST) for i in range(16)],
        )
    elif kind == "views":  # synchronised cabin cameras: three rows, one moment across them
        for row, y in enumerate((196, 216, 236)):
            base = (MIST, COBALT, GRAPHITE)[row]
            _strip(draw, y, 16, [SIGNAL if i == 7 else base for i in range(16)])
    elif kind == "stills":  # one frame per image: 16 separate tiles, each a single frame
        left, n, gap = 24, 16, 6
        w = (SIZE[0] - 2 * left - gap * (n - 1)) / n
        for i in range(n):
            x = left + i * (w + gap)
            fill = SIGNAL if i == 4 else (COBALT if i % 5 == 2 else MIST)
            _box(draw, x, 206, w, w, fill)
    elif kind == "table":  # the manifest: rows of clips, one column of labels marked
        for row, y in enumerate(range(196, 252, 11)):
            colours = [SIGNAL if i == 9 else (GRAPHITE if row == 0 else MIST) for i in range(16)]
            _strip(draw, y, 7, colours, gap=2)
    else:
        raise ValueError(kind)


def _centred(
    draw: ImageDraw.ImageDraw, y: float, text: str, face: Font, fill: str, spacing: float = 0
) -> None:
    """Text centred on x = 280 (the thumbnail's centre); `spacing` adds tracking in px."""
    if not spacing:
        width = draw.textlength(text, font=face)
        draw.text(((SIZE[0] * SCALE - width) / 2, y * SCALE), text, font=face, fill=fill)
        return
    widths = [draw.textlength(c, font=face) for c in text]
    total = sum(widths) + spacing * SCALE * (len(text) - 1)
    x = (SIZE[0] * SCALE - total) / 2
    for c, w in zip(text, widths, strict=True):
        draw.text((x, y * SCALE), c, font=face, fill=fill)
        x += w + spacing * SCALE


def render(slug: str) -> Image.Image:
    name, descriptor, motif = COVERS[slug]
    image = Image.new("RGB", (SIZE[0] * SCALE, SIZE[1] * SCALE), PAPER)
    draw = ImageDraw.Draw(image)
    _box(draw, 0, 0, SIZE[0], 6, INK)  # top rule
    # Wordmark: a signal square and the letter-spaced name.
    label = font(13, bold=True)
    _centred(draw, 38, "BULWARK", label, INK, spacing=4)
    _box(draw, 274, 26, 12, 4, SIGNAL)
    title = font(44 if len(name) <= 8 else 40, bold=True)
    _centred(draw, 74, name, title, INK)
    _centred(draw, 136, descriptor, font(16), GRAPHITE)
    _motif(draw, motif)
    return image.resize(SIZE, Image.Resampling.LANCZOS)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--show", action="store_true", help="also write each thumbnail crop to /tmp"
    )
    args = parser.parse_args(argv)
    OUT.mkdir(exist_ok=True)
    for slug in COVERS:
        image = render(slug)
        image.save(OUT / f"{slug}.png", optimize=True)
        print(OUT / f"{slug}.png")
        if args.show:
            image.crop(THUMB).save(f"/tmp/{slug}-thumb.png")


if __name__ == "__main__":
    main()
