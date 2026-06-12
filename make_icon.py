"""Generate the AirSketch app icon (airsketch.ico) + a PNG preview.

Theme: air-drawing — a freehand aqua stroke ending in a bright fingertip dot,
on a dark rounded panel (matches the app's UI palette). Rendered supersampled
for crisp downscaling to the standard .ico sizes.

Run:  python make_icon.py
Out:  airsketch.ico  +  airsketch_icon_preview.png
"""
import math

from PIL import Image, ImageDraw, ImageFilter

S = 1024                      # supersample master size
M = int(S * 0.085)            # margin
BG_TOP = (28, 31, 44)         # dark navy
BG_BOT = (16, 18, 26)
ACCENT = (44, 226, 198)       # aqua/teal (app UI_ACCENT-ish)
ACCENT_DK = (24, 150, 150)
TIP = (240, 248, 255)         # fingertip core


def rounded_mask(size, radius):
    m = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(m)
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=radius, fill=255)
    return m


def vertical_gradient(size, top, bot):
    base = Image.new("RGB", (size, size), top)
    top = list(top); bot = list(bot)
    px = base.load()
    for y in range(size):
        t = y / (size - 1)
        r = int(top[0] + (bot[0] - top[0]) * t)
        g = int(top[1] + (bot[1] - top[1]) * t)
        b = int(top[2] + (bot[2] - top[2]) * t)
        for x in range(size):
            px[x, y] = (r, g, b)
    return base


def stroke_points():
    """A smooth freehand 'air-drawn' flourish across the icon."""
    pts = []
    for i in range(101):
        t = i / 100.0
        # left-low -> rises with a gentle S wave -> upper-right
        x = M * 1.3 + t * (S - 2.6 * M)
        y = (S * 0.62
             - math.sin(t * math.pi * 1.15) * S * 0.26
             + math.sin(t * math.pi * 3.0) * S * 0.05)
        pts.append((x, y))
    return pts


def build_master():
    # background panel
    grad = vertical_gradient(S, BG_TOP, BG_BOT)
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    img.paste(grad, (0, 0), rounded_mask(S, int(S * 0.22)))

    draw = ImageDraw.Draw(img)

    # subtle accent border
    draw.rounded_rectangle(
        [int(S * 0.018)] * 1 + [int(S * 0.018), S - int(S * 0.018), S - int(S * 0.018)],
        radius=int(S * 0.2), outline=(ACCENT_DK[0], ACCENT_DK[1], ACCENT_DK[2], 90),
        width=max(2, S // 180))

    pts = stroke_points()

    # glow pass (blurred wide stroke)
    glow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    gdraw = ImageDraw.Draw(glow)
    gdraw.line(pts, fill=(ACCENT[0], ACCENT[1], ACCENT[2], 160),
               width=int(S * 0.11), joint="curve")
    glow = glow.filter(ImageFilter.GaussianBlur(S * 0.03))
    img.alpha_composite(glow)

    # main stroke (rounded caps via overlapping circles + polyline)
    draw.line(pts, fill=(ACCENT[0], ACCENT[1], ACCENT[2], 255),
              width=int(S * 0.06), joint="curve")
    r_cap = int(S * 0.03)
    for cx, cy in (pts[0], pts[-1]):
        draw.ellipse([cx - r_cap, cy - r_cap, cx + r_cap, cy + r_cap],
                     fill=(ACCENT[0], ACCENT[1], ACCENT[2], 255))

    # fingertip dot at the leading (end) point: aqua ring + white core + halo
    ex, ey = pts[-1]
    halo = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(halo).ellipse(
        [ex - S * 0.14, ey - S * 0.14, ex + S * 0.14, ey + S * 0.14],
        fill=(ACCENT[0], ACCENT[1], ACCENT[2], 120))
    img.alpha_composite(halo.filter(ImageFilter.GaussianBlur(S * 0.02)))
    r1 = int(S * 0.085)
    draw.ellipse([ex - r1, ey - r1, ex + r1, ey + r1], fill=(0, 0, 0, 255))
    draw.ellipse([ex - r1, ey - r1, ex + r1, ey + r1],
                 outline=(ACCENT[0], ACCENT[1], ACCENT[2], 255), width=int(S * 0.018))
    r2 = int(S * 0.038)
    draw.ellipse([ex - r2, ey - r2, ex + r2, ey + r2], fill=(*TIP, 255))

    # re-apply rounded mask so glow/halo don't bleed past the panel corners
    out = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    out.paste(img, (0, 0), rounded_mask(S, int(S * 0.22)))
    return out


def main():
    master = build_master()
    base = master.resize((256, 256), Image.LANCZOS)
    base.save("airsketch_icon_preview.png")
    sizes = [(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)]
    base.save("airsketch.ico", format="ICO", sizes=sizes)
    print("wrote airsketch.ico (sizes: %s) + airsketch_icon_preview.png"
          % ", ".join(str(s[0]) for s in sizes))


if __name__ == "__main__":
    main()
