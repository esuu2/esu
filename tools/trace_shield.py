"""Trace the shield into smooth outlines that keep the real corners sharp."""
import math
import numpy as np
import cv2

UP = 8


def resample(c, step):
    c = np.vstack([c, c[:1]])
    seg = np.hypot(*np.diff(c, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    n = max(int(s[-1] / step), 16)
    t = np.linspace(0, s[-1], n, endpoint=False)
    return np.stack([np.interp(t, s, c[:, 0]), np.interp(t, s, c[:, 1])], 1)


def gauss_closed(p, sig):
    if sig <= 0:
        return p.copy()
    r = int(3 * sig) + 1
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sig) ** 2); k /= k.sum()
    pad = np.vstack([p[-r:], p, p[:r]])
    return np.stack([np.convolve(pad[:, j], k, 'valid') for j in range(2)], 1)


def gauss_open(p, sig):
    """Smooth an open segment, keeping both end points (and their direction) fixed."""
    r = int(3 * sig) + 1
    if len(p) < 3:
        return p
    r = min(r, len(p) - 1)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sig) ** 2); k /= k.sum()
    head = 2 * p[0] - p[r:0:-1]
    tail = 2 * p[-1] - p[-2:-r - 2:-1]
    pad = np.vstack([head, p, tail])
    out = np.stack([np.convolve(pad[:, j], k, 'valid') for j in range(2)], 1)
    out[0], out[-1] = p[0], p[-1]
    return out


def corners(p, win, thresh_deg, min_sep):
    n = len(p)
    turn = np.zeros(n)
    for i in range(n):
        a, b, c = p[(i - win) % n], p[i], p[(i + win) % n]
        v1, v2 = a - b, c - b
        cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        turn[i] = 180 - math.degrees(math.acos(np.clip(cosang, -1, 1)))
    idx = []
    for i in np.argsort(-turn):
        if turn[i] < thresh_deg:
            break
        if all(min((i - j) % n, (j - i) % n) >= min_sep for j in idx):
            idx.append(i)
    return sorted(idx), turn


def smooth_contour(c_src, step=0.25, sig_src=1.6, win_src=2.5, thresh=40, sep_src=3.0, force=()):
    """c_src: contour in source pixels. Returns smoothed polygon (source px)."""
    p = resample(c_src, step)
    ps = gauss_closed(p, 1.0 / step)                         # light smoothing for corner finding
    cs, _ = corners(ps, int(win_src / step), thresh, int(sep_src / step))
    for fx, fy in force:                                     # e.g. the top point of the shield
        i = int(np.argmin(np.hypot(p[:, 0] - fx, p[:, 1] - fy)))
        if np.hypot(*(p[i] - (fx, fy))) < 1.0 and all(min((i - j) % len(p), (j - i) % len(p)) > 4 for j in cs):
            cs.append(i)
    cs = sorted(cs)
    sig = sig_src / step
    if not cs:
        return gauss_closed(p, sig), []
    # real corner positions: take them from the lightly smoothed curve
    out = []
    for a, b in zip(cs, cs[1:] + [cs[0] + len(p)]):
        seg = np.array([ps[i % len(p)] if i in (a, b) else p[i % len(p)] for i in range(a, b + 1)])
        out.append(gauss_open(seg, sig)[:-1])
    return np.vstack(out), [ps[i] for i in cs]


def trace(mask, **kw):
    big = cv2.resize(mask.astype(np.uint8) * 255, None, fx=UP, fy=UP, interpolation=cv2.INTER_NEAREST)
    cs, hier = cv2.findContours(big, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    polys = []
    for c, h in zip(cs, hier[0]):
        if cv2.contourArea(c) < (UP * 3) ** 2:
            continue
        pts = (c[:, 0, :].astype(float) + 0.5) / UP
        axis = kw.pop('axis', None) if 'axis' in kw else None
        force = []
        if axis is not None:                                  # top-most point on the symmetry axis
            on = pts[np.abs(pts[:, 0] - axis) < 0.6]
            if len(on):
                force.append((axis, on[:, 1].min()))
        sm, cn = smooth_contour(pts, force=force, **{k: v for k, v in kw.items() if k != 'axis'})
        if axis is not None:
            kw['axis'] = axis
        polys.append((sm, h[3] != -1, cn))
    return polys


def symmetric(mask, axis2):
    """Mirror the right half onto the left so the logo is perfectly symmetric.
    axis2 = 2 * axis x (in pixel-edge units), e.g. 250 for an axis at x = 125."""
    m = mask.copy()
    w = m.shape[1]
    for c in range(w):
        mc = axis2 - 1 - c
        if c < axis2 / 2 and 0 <= mc < w:
            m[:, c] = mask[:, mc]
    return m
