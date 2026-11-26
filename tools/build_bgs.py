"""Build smooth phone backgrounds for the purple echobots.gg theme.

shield: traces the pulsing shield GIF into clean, symmetric vector outlines
        once, then redraws every frame at full phone resolution with
        anti-aliased edges, sized from each original frame so the pulse matches
esu:    places the ESU banner GIF on a tall black canvas with a soft
        oval fade so the glow melts into black
"""
import sys, math, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2
from trace_shield import trace, symmetric
from PIL import Image, ImageDraw

W, H = 1080, 2340            # phone-shaped canvas (iPhone ratio ~0.46)
OUTER = (123, 60, 200)       # shield outer purple
SS = 3                       # supersampling for anti-aliasing


# ---------- shield ----------
def masks(frame):
    a = np.array(frame.convert('RGBA'))
    shape = a[..., 3] > 0
    inner = shape & (a[..., 0] < 90) & (a[..., 1] < 45)
    inner_col = tuple(int(np.median(a[..., k][inner])) for k in range(3))
    return shape, inner, inner_col


def shield(src_path, out_path, shield_px):
    src = Image.open(src_path)
    # reference = biggest frame of the pulse
    areas = []
    for i in range(src.n_frames):
        src.seek(i)
        areas.append(masks(src)[0].sum())
    ref = int(np.argmax(areas))
    src.seek(ref)
    shape, inner, inner_col = masks(src)
    ys, xs = np.nonzero(shape)
    cx, cy = (xs.min() + xs.max() + 1) / 2, (ys.min() + ys.max() + 1) / 2
    ref_w = xs.max() + 1 - xs.min()
    ax2 = xs.min() + xs.max() + 1                    # symmetry axis x 2
    shape, inner = symmetric(shape, ax2), symmetric(inner, ax2)
    kw = dict(sig_src=5, thresh=25, axis=ax2 / 2)
    outer_polys, inner_polys = trace(shape, **kw), trace(inner, **kw)
    print('ref frame', ref, 'outer parts', len(outer_polys), 'inner parts', len(inner_polys), 'inner colour', inner_col)

    base = shield_px / ref_w                          # px per source px at full size
    frames = []
    for i in range(src.n_frames):
        s = base * math.sqrt(areas[i] / areas[ref])   # smooth pulse from the real frame sizes
        canvas = Image.new('RGB', (W * SS, H * SS), (0, 0, 0))
        d = ImageDraw.Draw(canvas)
        def draw(polys, col):
            for pts, hole, _ in sorted(polys, key=lambda p: p[1]):   # fills first, holes after
                xy = [((x - cx) * s * SS + W * SS / 2, (y - cy) * s * SS + H * SS / 2) for x, y in pts]
                d.polygon(xy, fill=(0, 0, 0) if hole else col)
        draw(outer_polys, OUTER)
        draw(inner_polys, inner_col)
        frames.append(canvas.resize((W, H), Image.LANCZOS))
    save(frames, out_path, src.info.get('duration', 50))


# ---------- ESU banner ----------
def esu(src_path, out_path, banner_w):
    src = Image.open(src_path)
    sw, sh = src.size
    bw, bh = banner_w, round(sh * banner_w / sw)
    content_cx = 0.505 * bw                           # logo + ESU text centre in the banner
    ox, oy = round(W / 2 - content_cx), round((H - bh) / 2)
    # oval fade centred on the logo
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    rx, ry = W * 0.62, bh * 0.62
    dist = np.sqrt(((xx - W / 2) / rx) ** 2 + ((yy - H / 2) / ry) ** 2)
    t = np.clip((1.0 - dist) / 0.55, 0, 1)
    fade = Image.fromarray((255 * t * t * (3 - 2 * t)).astype(np.uint8))
    frames = []
    for i in range(src.n_frames):
        src.seek(i)
        b = src.convert('RGB').resize((bw, bh), Image.LANCZOS)
        full = Image.new('RGB', (W, H), (0, 0, 0))
        full.paste(b, (ox, oy))
        canvas = Image.new('RGB', (W, H), (0, 0, 0))
        canvas.paste(full, (0, 0), fade)
        frames.append(canvas)
    save(frames, out_path, src.info.get('duration', 50))


def save(frames, out_path, duration):
    # one shared palette for every frame -> no colour flicker between frames
    sample = Image.new('RGB', (W, H * 3))
    for k, idx in enumerate([0, len(frames) // 3, 2 * len(frames) // 3]):
        sample.paste(frames[idx], (0, H * k))
    pal = sample.quantize(colors=255, method=Image.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q[0].save(out_path, save_all=True, append_images=q[1:], duration=duration, loop=0, optimize=True, disposal=1)


if __name__ == '__main__':
    kind, src_path, out_path, size = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    {'shield': shield, 'esu': esu}[kind](src_path, out_path, size)
