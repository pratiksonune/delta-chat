from __future__ import annotations

import os

from PIL import Image, ImageDraw, ImageFont
from pdf2image import convert_from_path

from src.canonical.model import CanonicalDocument
from src.delta.engine import ChangeType, DeltaEntry

COLOR = {
    ChangeType.ADDED: (30, 180, 60),
    ChangeType.REMOVED: (200, 30, 30),
    ChangeType.MODIFIED: (230, 140, 20),
}

DPI = 200
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def render_markup(pid_b_path: str, entries: list[DeltaEntry], out_dir: str) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    scale = DPI / 72.0
    pages = convert_from_path(pid_b_path, dpi=DPI)
    out_paths = []

    by_page: dict[int, list[DeltaEntry]] = {}
    for e in entries:
        by_page.setdefault(e.page, []).append(e)

    try:
        font = ImageFont.truetype(FONT_PATH, 14)
    except Exception:
        font = ImageFont.load_default()

    for page_idx, img in enumerate(pages):
        img = img.convert("RGB")
        draw = ImageDraw.Draw(img)
        for e in by_page.get(page_idx, []):
            color = COLOR[e.change_type]
            box = (
                e.bbox.x0 * scale - 3,
                e.bbox.top * scale - 3,
                e.bbox.x1 * scale + 3,
                e.bbox.bottom * scale + 3,
            )
            draw.rectangle(box, outline=color, width=3)
            label = f"{e.entry_id} {e.change_type.value[:3].upper()}"
            draw.text((box[0], max(0, box[1] - 16)), label, fill=color, font=font)

        legend_y = 10
        for ct, color in COLOR.items():
            draw.rectangle((10, legend_y, 30, legend_y + 14), fill=color)
            draw.text((36, legend_y), ct.value.capitalize(), fill=(0, 0, 0), font=font)
            legend_y += 20

        out_path = os.path.join(out_dir, f"markup_page{page_idx + 1}.png")
        img.save(out_path)
        out_paths.append(out_path)

    if out_paths:
        images = [Image.open(p).convert("RGB") for p in out_paths]
        pdf_path = os.path.join(out_dir, "markup.pdf")
        images[0].save(pdf_path, save_all=True, append_images=images[1:])
        out_paths.append(pdf_path)

    return out_paths
