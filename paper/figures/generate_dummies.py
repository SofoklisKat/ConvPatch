"""Generate dummy figures for patch-embedding concepts. Run: python generate_dummies.py"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).parent
W, H = 640, 480
BG = (248, 249, 252)
GRID = (180, 190, 210)
ACCENT = (59, 130, 246)
ACCENT2 = (16, 185, 129)
TEXT = (30, 41, 59)
MUTED = (100, 116, 139)


def font(size: int):
    for name in ("DejaVuSans.ttf", "LiberationSans-Regular.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def new_canvas(title: str, subtitle: str = "") -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    draw.text((24, 16), title, fill=TEXT, font=font(20))
    if subtitle:
        draw.text((24, 44), subtitle, fill=MUTED, font=font(13))
    return img, draw


def draw_image_grid(draw, x0, y0, size, n_patches, patch_px, colors, overlap=False):
    """Draw n_patches x n_patches patch grid on a size x size canvas area."""
    cell = size // n_patches
    for r in range(n_patches):
        for c in range(n_patches):
            if overlap:
                # overlapping receptive fields: larger box, centered on cell
                pad = patch_px // 4
                x1 = x0 + c * cell - pad
                y1 = y0 + r * cell - pad
                x2 = x1 + patch_px
                y2 = y1 + patch_px
            else:
                x1 = x0 + c * cell
                y1 = y0 + r * cell
                x2 = x1 + cell
                y2 = y1 + cell
            col = colors[(r + c) % len(colors)]
            draw.rectangle([x1, y1, x2, y2], fill=col, outline=GRID, width=2)


def fig1_patch_grid():
    """Non-overlapping P×P patches on a 32×32-style image."""
    img, draw = new_canvas(
        "1. Linear patchify: non-overlapping patches",
        "32×32 image, P=4 → 8×8 grid = 64 tokens",
    )
    x0, y0, display = 120, 100, 256
    n = 8  # 32/4
    colors = [(255, 220, 200), (200, 230, 255), (220, 255, 210), (255, 240, 200)]
    draw_image_grid(draw, x0, y0, display, n, 32, colors, overlap=False)
    # highlight one patch
    cell = display // n
    draw.rectangle([x0 + 3 * cell, y0 + 2 * cell, x0 + 4 * cell, y0 + 3 * cell], outline=ACCENT, width=4)
    draw.text((x0 + 3 * cell + 4, y0 + 2 * cell + 4), "1 patch", fill=ACCENT, font=font(11))
    draw.text((x0, y0 + display + 16), "Each cell → flatten → linear proj → 1 token (dim D)", fill=MUTED, font=font(12))
    draw.text((400, 120), "ViT default:", fill=TEXT, font=font(14))
    draw.text((400, 148), "Conv k=P, s=P", fill=ACCENT, font=font(13))
    draw.text((400, 175), "no overlap", fill=MUTED, font=font(12))
    draw.text((400, 200), "N = 64", fill=TEXT, font=font(13))
    img.save(OUT / "01_linear_patch_grid.png")


def fig2_one_patch_to_token():
    """Zoom: one patch flattened to token vector."""
    img, draw = new_canvas(
        "2. One patch → one token (linear embedding)",
        "P×P×3 pixels reshaped and projected to dimension D",
    )
    # patch block
    px, py, ps = 80, 120, 120
    for r in range(4):
        for c in range(4):
            shade = 200 + (r + c) * 8
            draw.rectangle(
                [px + c * 30, py + r * 30, px + (c + 1) * 30, py + (r + 1) * 30],
                fill=(shade, 180, 160),
                outline=GRID,
            )
    draw.text((px, py - 24), "P×P patch (e.g. 4×4)", fill=TEXT, font=font(13))
    draw.text((px + 140, py + 50), "flatten", fill=MUTED, font=font(14))
    draw.text((px + 195, py + 45), "→", fill=TEXT, font=font(24))
    # vector bars
    bx, by = 320, 110
    draw.text((bx, by - 24), "token vector (D)", fill=TEXT, font=font(13))
    for i in range(12):
        h = 30 + (i % 5) * 18
        draw.rectangle([bx, by + i * 22, bx + h, by + i * 22 + 14], fill=ACCENT)
    draw.text((bx + 100, by + 80), "...", fill=MUTED, font=font(20))
    draw.text((80, 400), "Same op as: Conv2d(3 → D, kernel=P, stride=P)", fill=MUTED, font=font(12))
    img.save(OUT / "02_patch_to_token.png")


def fig3_conv_stem():
    """Progressive downsampling conv stem."""
    img, draw = new_canvas(
        "3. Conv stem: stacked stride-2 convolutions",
        "Total stride = P → same N tokens as linear baseline",
    )
    stages = [
        ("32×32×3", 100, 140, 80, ACCENT2),
        ("16×16", 220, 150, 60, (96, 165, 250)),
        ("8×8", 330, 165, 45, ACCENT),
        ("64×D", 430, 175, 35, (99, 102, 241)),
    ]
    for i, (label, x, y, s, col) in enumerate(stages):
        draw.rectangle([x, y, x + s, y + s], fill=col, outline=TEXT, width=2)
        draw.text((x, y + s + 8), label, fill=TEXT, font=font(11))
        if i < len(stages) - 1:
            nx = stages[i + 1][1]
            draw.text((x + s + 8, y + s // 2 - 8), "3×3 s2", fill=MUTED, font=font(11))
            draw.line([x + s, y + s // 2, nx, y + s // 2], fill=MUTED, width=2)
    draw.text((100, 320), "BN + GELU between stages", fill=MUTED, font=font(12))
    draw.text((100, 345), "Then flatten 8×8 grid → N=64 tokens", fill=MUTED, font=font(12))
    draw.text((100, 380), "Local 3×3 filters build hierarchy before attention", fill=ACCENT2, font=font(12))
    img.save(OUT / "03_conv_stem_downsample.png")


def fig4_overlap():
    """Overlapping vs non-overlapping."""
    img, draw = new_canvas(
        "4. Non-overlap vs overlapping patch embed",
        "Variant C: kernel=2P, stride=P (receptive fields overlap)",
    )
    y1, size, n = 110, 180, 6
    draw.text((70, y1 - 28), "A: Linear (s=P)", fill=TEXT, font=font(13))
    colors = [(255, 210, 190), (190, 220, 255)]
    draw_image_grid(draw, 70, y1, size, n, 32, colors, overlap=False)
    y2 = 300
    draw.text((70, y2 - 28), "C: Overlapping (k=2P)", fill=TEXT, font=font(13))
    cell = size // n
    for r in range(n):
        for c in range(n):
            pad = cell // 2
            x1 = 70 + c * cell - pad
            y1b = y2 + r * cell - pad
            x2 = x1 + cell + pad
            y2b = y1b + cell + pad
            col = colors[(r + c) % 2]
            draw.rectangle([x1, y1b, x2, y2b], fill=(*col, 80) if False else col, outline=ACCENT2, width=2)
    draw.text((320, 130), "No overlap:", fill=MUTED, font=font(12))
    draw.text((320, 152), "each pixel in one patch", fill=MUTED, font=font(12))
    draw.text((320, 310), "Overlap:", fill=MUTED, font=font(12))
    draw.text((320, 332), "each token sees", fill=MUTED, font=font(12))
    draw.text((320, 354), "neighboring context", fill=ACCENT2, font=font(12))
    draw.text((320, 390), "Same N if stride=P", fill=TEXT, font=font(12))
    img.save(OUT / "04_overlap_vs_linear.png")


def fig5_full_pipeline():
    """End-to-end: image → patch embed → tokens → ViT encoder."""
    img, draw = new_canvas(
        "5. Where patch embedding sits in ViT",
        "ConvPatch changes only the orange box; encoder is fixed",
    )
    boxes = [
        ("Image\n32×32×3", 40, 200, 90, 70, (226, 232, 240)),
        ("Patch embed\n(variant A–E)", 160, 190, 120, 90, (254, 215, 170)),
        ("+ CLS + pos", 320, 200, 90, 70, (226, 232, 240)),
        ("Transformer\n(L blocks)", 450, 185, 110, 100, (191, 219, 254)),
        ("Class\nlogits", 590, 200, 80, 70, (226, 232, 240)),
    ]
    for label, x, y, w, h, col in boxes:
        draw.rounded_rectangle([x, y, x + w, y + h], radius=8, fill=col, outline=TEXT, width=2)
        for i, line in enumerate(label.split("\n")):
            draw.text((x + 10, y + 12 + i * 18), line, fill=TEXT, font=font(11))
        if x < 560:
            draw.line([x + w, y + h // 2, x + w + 28, y + h // 2], fill=MUTED, width=2)
            draw.polygon([(x + w + 28, y + h // 2), (x + w + 20, y + h // 2 - 5), (x + w + 20, y + h // 2 + 5)], fill=MUTED)
    draw.text((160, 300), "← independent variable (this study)", fill=ACCENT, font=font(12))
    draw.text((450, 300), "held constant", fill=MUTED, font=font(12))
    img.save(OUT / "05_vit_pipeline.png")


def fig6_token_count_invariant():
    """All variants → same N."""
    img, draw = new_canvas(
        "6. Token-count invariant (controlled study)",
        "All stems: total stride = P → N = (H/P)²",
    )
    variants = ["Linear", "Conv stem", "Overlap", "Hierarchical", "DW-sep"]
    for i, v in enumerate(variants):
        y = 100 + i * 62
        draw.rounded_rectangle([60, y, 260, y + 44], radius=6, fill=(230, 240, 255), outline=GRID)
        draw.text((80, y + 12), v, fill=TEXT, font=font(13))
        draw.text((300, y + 12), "→", fill=MUTED, font=font(16))
        draw.rounded_rectangle([340, y, 520, y + 44], radius=6, fill=ACCENT2, outline=TEXT)
        draw.text((380, y + 12), "N = 64 tokens", fill=(255, 255, 255), font=font(13))
    draw.text((60, 420), "CIFAR: H=32, P=4  |  ImageNet-100: H=224, P=16 → N=196", fill=MUTED, font=font(12))
    img.save(OUT / "06_same_token_count.png")


if __name__ == "__main__":
    fig1_patch_grid()
    fig2_one_patch_to_token()
    fig3_conv_stem()
    fig4_overlap()
    fig5_full_pipeline()
    fig6_token_count_invariant()
    print(f"Saved 6 figures to {OUT}")
