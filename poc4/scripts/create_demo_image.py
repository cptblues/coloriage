"""Génère l'image de démonstration synthétique incluse dans le projet."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "examples" / "demo.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    width, height = 960, 720
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            sky = y < 430
            if sky:
                ratio = y / 430
                pixels[x, y] = (
                    int(125 + 90 * ratio),
                    int(185 + 45 * ratio),
                    int(235 - 15 * ratio),
                )
            else:
                ratio = (y - 430) / (height - 430)
                pixels[x, y] = (
                    int(95 - 35 * ratio),
                    int(165 - 40 * ratio),
                    int(85 - 25 * ratio),
                )

    draw = ImageDraw.Draw(image)
    draw.ellipse((710, 65, 850, 205), fill=(255, 220, 98))
    draw.polygon([(0, 450), (185, 220), (380, 450)], fill=(84, 120, 143))
    draw.polygon([(190, 450), (430, 170), (700, 450)], fill=(74, 104, 129))
    draw.polygon([(320, 450), (430, 170), (512, 450)], fill=(228, 229, 221))
    draw.polygon([(560, 450), (760, 245), (960, 430), (960, 510)], fill=(91, 125, 134))

    # Sujet central stylisé : un chat, utile pour observer la fidélité de forme.
    draw.ellipse((355, 335, 625, 650), fill=(206, 137, 73), outline=(87, 55, 42), width=8)
    draw.ellipse((360, 245, 620, 500), fill=(218, 151, 81), outline=(87, 55, 42), width=8)
    draw.polygon([(382, 300), (398, 178), (478, 268)], fill=(218, 151, 81), outline=(87, 55, 42))
    draw.polygon([(522, 268), (600, 178), (607, 313)], fill=(218, 151, 81), outline=(87, 55, 42))
    draw.polygon([(405, 273), (414, 214), (455, 263)], fill=(190, 105, 102))
    draw.polygon([(548, 263), (589, 214), (593, 279)], fill=(190, 105, 102))
    draw.ellipse((420, 330, 467, 378), fill=(86, 145, 92), outline=(35, 52, 37), width=4)
    draw.ellipse((513, 330, 560, 378), fill=(86, 145, 92), outline=(35, 52, 37), width=4)
    draw.ellipse((438, 342, 451, 371), fill=(20, 26, 20))
    draw.ellipse((529, 342, 542, 371), fill=(20, 26, 20))
    draw.polygon([(474, 392), (506, 392), (490, 414)], fill=(113, 61, 61))
    draw.arc((455, 400, 490, 435), 10, 100, fill=(70, 40, 38), width=3)
    draw.arc((490, 400, 525, 435), 80, 170, fill=(70, 40, 38), width=3)
    for offset in (-15, 0, 15):
        draw.line((455, 411 + offset, 340, 395 + offset), fill=(84, 61, 50), width=3)
        draw.line((525, 411 + offset, 640, 395 + offset), fill=(84, 61, 50), width=3)

    # Rayures et texture légère pour créer quelques micro-régions mesurables.
    for index in range(8):
        x = 400 + index * 25
        y = 470 + int(15 * math.sin(index))
        draw.arc((x, y, x + 70, y + 105), 80, 230, fill=(135, 82, 50), width=7)

    image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
    image.save(output)
    print(output)


if __name__ == "__main__":
    main()

