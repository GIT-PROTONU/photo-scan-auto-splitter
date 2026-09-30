#!/usr/bin/env python3
import argparse
import math
import os
import queue
import shutil
import subprocess
import sys
import threading

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image, ImageDraw, ImageFilter, ImageOps

Image.MAX_IMAGE_PIXELS = None

EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

APP_NAME = "photo-scan-auto-splitter"
APP_VERSION = "1.6.0"


def _runs(b):
    idx = np.flatnonzero(b)
    if idx.size == 0:
        return []
    sp = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[sp + 1]]
    ends = np.r_[idx[sp], idx[-1]]
    return list(zip(starts.tolist(), ends.tolist()))


def _majority(m):
    mm = m.astype(np.uint8)
    acc = np.zeros_like(mm)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            acc += np.roll(np.roll(mm, dy, 0), dx, 1)
    return (mm + acc) >= 5


def _label(mask):
    h, w = mask.shape
    lab = np.zeros((h, w), np.int32)
    parent = [0]

    def find(x):
        r = x
        while parent[r] != r:
            r = parent[r]
        while parent[x] != r:
            parent[x], x = r, parent[x]
        return r

    nl = 1
    prev = []
    for y in range(h):
        idx = np.flatnonzero(mask[y])
        cur = []
        if idx.size:
            sp = np.flatnonzero(np.diff(idx) > 1)
            starts = np.r_[idx[0], idx[sp + 1]]
            ends = np.r_[idx[sp], idx[-1]]
            for s, e in zip(starts.tolist(), ends.tolist()):
                l = 0
                for ps, pe, pl in prev:
                    if s <= pe + 1 and ps <= e + 1:
                        if l == 0:
                            l = pl
                        elif l != pl:
                            ra, rb = find(l), find(pl)
                            if ra != rb:
                                parent[max(ra, rb)] = min(ra, rb)
                if l == 0:
                    l = nl
                    parent.append(nl)
                    nl += 1
                cur.append((s, e, l))
                lab[y, s:e + 1] = l
        prev = cur
    for l in np.unique(lab):
        if l == 0:
            continue
        r = find(int(l))
        if r != l:
            lab[lab == l] = r
    return lab


def _split(m, ox, oy):
    if not m.any():
        return []
    occ_r = m.any(axis=1)
    occ_c = m.any(axis=0)
    runs = _runs(occ_r)
    if len(runs) > 1:
        parts = []
        for a, b in runs:
            parts += _split(m[a:b + 1, :], ox, oy + a)
        return parts
    runs = _runs(occ_c)
    if len(runs) > 1:
        parts = []
        for a, b in runs:
            parts += _split(m[:, a:b + 1], ox + a, oy)
        return parts
    return [(m, ox, oy)]


def _erode(m):
    er = m.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            er &= np.roll(np.roll(m, dy, 0), dx, 1)
    return er


def _dist_tf(m):
    h, w = m.shape
    BIG = np.float32(3.0 * (h + w + 1))
    a = np.where(m, BIG, np.float32(0.0))
    xr = np.arange(w, dtype=np.float32) * 3.0
    for flip in (False, True):
        rows = range(h) if not flip else range(h - 1, -1, -1)
        for y in rows:
            row = a[y]
            if (not flip and y > 0) or (flip and y < h - 1):
                prev = a[y - 1] if not flip else a[y + 1]
                pl = np.empty(w, np.float32)
                pr = np.empty(w, np.float32)
                pl[1:] = prev[:-1]
                pl[0] = BIG
                pr[:-1] = prev[1:]
                pr[-1] = BIG
                row = np.minimum(row, prev + 3.0)
                row = np.minimum(row, pl + 4.0)
                row = np.minimum(row, pr + 4.0)
            if not flip:
                v = np.minimum.accumulate(row - xr)
                row = v + xr
            else:
                v = np.minimum.accumulate((row + xr)[::-1])[::-1]
                row = v - xr
            a[y] = row
    return a * (1.0 / 3.0)


def _dt_seeds(d, max_seeds=9):
    h, w = d.shape
    k = 4
    H = -(-h // k) * k
    W = -(-w // k) * k
    pad = np.zeros((H, W), np.float32)
    pad[:h, :w] = d
    dc = pad.reshape(H // k, k, W // k, k).mean(axis=(1, 3))
    ch, cw = dc.shape
    top = float(dc.max())
    if top <= 0:
        return []
    seeds = []
    for step in range(15):
        thr = top * (0.88 ** step)
        m = dc >= thr
        lab = _label(m)
        for l in np.unique(lab):
            if l == 0:
                continue
            has = False
            for fx, fy in seeds:
                cy = min(fy // k, ch - 1)
                cx = min(fx // k, cw - 1)
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        yy = min(max(cy + dy, 0), ch - 1)
                        xx = min(max(cx + dx, 0), cw - 1)
                        if lab[yy, xx] == l:
                            has = True
                            break
                    if has:
                        break
                if has:
                    break
            if has:
                continue
            ys, xs = np.nonzero(lab == l)
            if xs.size < 3:
                continue
            vals = dc[ys, xs]
            i = int(np.argmax(vals))
            base_y = int(ys[i]) * k
            base_x = int(xs[i]) * k
            y0 = max(0, base_y - 2)
            y1 = min(h, base_y + k + 2)
            x0 = max(0, base_x - 2)
            x1 = min(w, base_x + k + 2)
            patch = d[y0:y1, x0:x1]
            py, px = np.unravel_index(int(np.argmax(patch)), patch.shape)
            seeds.append((x0 + int(px), y0 + int(py)))
            if len(seeds) >= max_seeds:
                return seeds
        if len(seeds) >= max_seeds:
            break
    if len(seeds) > 1:
        sorted_s = sorted(seeds, key=lambda s: -d[s[1], s[0]])
        filtered = []
        for s in sorted_s:
            sx, sy = s
            sd = d[sy, sx]
            if any(math.hypot(sx - fx, sy - fy) < 0.75 * max(sd, d[fy, fx]) for fx, fy in filtered):
                continue
            filtered.append(s)
        seeds = filtered
    return seeds


def _voronoi_parts(m, seeds):
    ys, xs = np.nonzero(m)
    d = np.stack([(xs - sx) ** 2.0 + (ys - sy) ** 2.0 for sx, sy in seeds])
    lab = np.argmin(d, axis=0).astype(np.int32) + 1
    assign = np.zeros(m.shape, np.int32)
    assign[ys, xs] = lab
    return _parts_from_assign(m, assign, len(seeds))


def _geodesic_dt(seed, barrier):
    h, w = barrier.shape
    BIG = np.float32(1e9)
    d = np.full((h, w), BIG, np.float32)
    d[seed[1], seed[0]] = 0.0
    xr3 = np.arange(w, dtype=np.float32) * 3.0
    xr4 = np.arange(w, dtype=np.float32) * 4.0
    for flip in (False, True):
        rows = range(h) if not flip else range(h - 1, -1, -1)
        for y in rows:
            row = d[y]
            if (not flip and y > 0) or (flip and y < h - 1):
                prev = d[y - 1] if not flip else d[y + 1]
                pl = np.empty(w, np.float32)
                pr = np.empty(w, np.float32)
                pl[1:] = prev[:-1]
                pl[0] = BIG
                pr[:-1] = prev[1:]
                pr[-1] = BIG
                row = np.minimum(row, prev + 3.0)
                row = np.minimum(row, pl + 4.0)
                row = np.minimum(row, pr + 4.0)
            if not flip:
                v = np.minimum.accumulate(row - xr3)
                row = np.minimum(v + xr3, row)
            else:
                v = np.minimum.accumulate((row + xr3)[::-1])[::-1]
                row = np.minimum(v - xr3, row)
            d[y] = np.where(barrier[y], BIG, row)
    return d


def _geodesic_parts(m, seeds):
    h, w = m.shape
    barrier = ~m
    maps = [_geodesic_dt(s, barrier) for s in seeds]
    stack = np.stack(maps)
    arg = np.argmin(stack, axis=0).astype(np.int32) + 1
    reach = stack.min(axis=0) < 1e8
    assign = np.zeros((h, w), np.int32)
    assign[reach] = arg[reach]
    return _parts_from_assign(m, assign, len(seeds))


def _parts_from_assign(m, assign, k):
    total = float(m.sum())
    out = []
    boxes = []
    for i in range(1, k + 1):
        pm = m & (assign == i)
        a = float(pm.sum())
        if a < max(0.05 * total, 300.0):
            continue
        pys, pxs = np.nonzero(pm)
        boxes.append((pxs.min(), pys.min(), pxs.max(), pys.max(), pm))
    for i, b1 in enumerate(boxes):
        good = True
        for j, b2 in enumerate(boxes):
            if i == j:
                continue
            ix = min(b1[2], b2[2]) - max(b1[0], b2[0])
            iy = min(b1[3], b2[3]) - max(b1[1], b2[1])
            if ix > 0 and iy > 0:
                a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                if ix * iy / min(a1, a2) > 0.5:
                    good = False
                    break
        if good:
            out.append((b1[4], 0, 0))
    if len(out) < 2:
        return []
    return out


def _fill_holes(m):
    inv = ~m
    if not inv.any():
        return m.copy()
    lab = _label(inv)
    top = int(lab.max())
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    outside = np.zeros(top + 1, bool)
    outside[border] = True
    return m | (inv & ~outside[lab])


def _watershed(m, ox, oy):
    mf = _fill_holes(m)
    seeds = _dt_seeds(_dist_tf(mf))
    if len(seeds) < 2:
        return []
    parts = _geodesic_parts(mf, seeds)
    if not parts:
        parts = _voronoi_parts(mf, seeds)
    return [((pm & m), ox, oy) for pm, _, _ in parts]


_SEAM_SPAN_FRAC = 0.10
_SEAM_EV_EDGE_FRAC = 0.12
_MAX_PIECE_AR = 6.0
_MIN_PIECE_FRAC = 0.01
_QUAD_LOCK_FILL = 0.92
_QUAD_LOCK_AREA = 0.33
# Real flatbed placement tilt is <~2.5 deg, but a genuinely tilted single print
# can reach ~5 deg and should still be deskewed. Fitted quad angles beyond this
# are silhouette-fit artifacts (ragged / white-merged masks), never a true
# rotation, so the piece is cropped axis-aligned instead of being spuriously
# deskewed (which tilted it and clipped its corners).
_MAX_DESKEW_DEG = 6.0


def _quad_lock(m, page_area):
    """True when a component is geometrically a single photo: near-perfect
    rectangular silhouette, 4 external corners, photo-like aspect ratio, and
    too small to be a butted grid. Only ever fires where the depth-0 elong/area
    gate would skip seam search anyway, so it can never block a real split."""
    total = float(m.sum())
    if total >= _QUAD_LOCK_AREA * page_area:
        return False
    nc = _ext_corners(m)
    if not 4 <= nc <= 8:
        return False
    if total / float(m.size) < _QUAD_LOCK_FILL:
        return False
    bh, bw = m.shape
    ar = max(bh, bw) / max(1.0, min(bh, bw))
    return ar < 2.2


def _find_seam(m, lum, lo=35.0):
    """Best near-horizontal seam line inside comp mask m (image coords aligned).
    Returns cut[x] row indices, or None."""
    h, w = m.shape
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    span = x1 - x0 + 1
    vext = y1 - y0 + 1
    if span < 80 or vext < 90:
        return None
    g = np.zeros((h, w), np.float32)
    if h > 5:
        g[2:h - 2] = np.abs(lum[4:h] - lum[:h - 4])
    mg = max(12, int(0.08 * vext))
    ya, yb = y0 + mg, y1 - mg
    if yb - ya < 40:
        return None
    win, half = 9, 4
    pxs, pys = [], []
    for x in range(x0, x1 + 1):
        col = g[ya:yb + 1, x]
        if col.size < win:
            continue
        sw = sliding_window_view(col, win)
        mx = sw.max(axis=1)
        c = col[half:half + sw.shape[0]]
        for i in np.flatnonzero((c >= lo) & (c >= mx)):
            pxs.append(x)
            pys.append(i + ya + half)
    if len(pxs) < 0.75 * span:
        return None
    pxs = np.asarray(pxs, np.float64)
    pys = np.asarray(pys, np.float64)
    edges = np.arange(ya - 8, yb + 9, 4.0)
    best = None
    for b in np.arange(-0.06, 0.061, 0.01):
        hist, _ = np.histogram(pys - b * pxs, bins=edges)
        sm = np.convolve(hist, np.ones(5, np.float64), "same")
        i = int(np.argmax(sm))
        a = 0.5 * (edges[i] + edges[i + 1])
        dd = np.abs(pys - (a + b * pxs))
        cols = np.unique(pxs[dd <= 9.0].astype(np.int64))
        if best is None or len(cols) > best[0]:
            best = (len(cols), b, a, cols.min(), cols.max())
    cov, b, a, c_lo, c_hi = best
    if cov < 0.68 * span or (c_hi - c_lo) < 0.7 * span:
        return None
    for _ in range(2):
        d = np.abs(pys - (a + b * pxs))
        inl = d <= 12.0
        if inl.sum() < 0.6 * span:
            break
        A = np.stack([np.ones(int(inl.sum())), pxs[inl]], 1)
        sol, *_ = np.linalg.lstsq(A, pys[inl], rcond=None)
        na, nb = float(sol[0]), float(sol[1])
        cols = np.unique(pxs[np.abs(pys - (na + nb * pxs)) <= 9.0].astype(np.int64))
        nc = len(cols)
        if nc >= cov:
            a, b, cov, c_lo, c_hi = na, nb, nc, cols.min(), cols.max()
        else:
            break
    if cov < 0.68 * span or (c_hi - c_lo) < 0.7 * span:
        return None
    ev_tol = max(4, int(round(_SEAM_EV_EDGE_FRAC * span)))
    if c_lo > x0 + ev_tol or c_hi < x1 - ev_tol:
        return None
    lo_y, hi_y = a + b * x0, a + b * x1
    lim = 0.06 * vext
    if not (y0 + lim <= min(lo_y, hi_y) and max(lo_y, hi_y) <= y1 - lim):
        return None
    cut = np.clip(np.round(a + b * np.arange(w, dtype=np.float64)), 0, h).astype(np.int32)
    xi = np.arange(x0, x1 + 1)
    if m[cut[xi], xi].mean() < _SEAM_SPAN_FRAC:
        return None
    return cut


def _crop_sub(m, ox, oy):
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return None
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    return m[y0:y1 + 1, x0:x1 + 1], ox + x0, oy + y0


def _crop_subs(m, ox, oy, page_area):
    lab_m = _label(m)
    res = []
    for l in np.unique(lab_m):
        if l == 0:
            continue
        ys, xs = np.nonzero(lab_m == l)
        area = xs.size
        if area < max(400, _MIN_PIECE_FRAC * page_area):
            continue
        sub_m = lab_m[ys.min():ys.max() + 1, xs.min():xs.max() + 1] == l
        res.append((sub_m, ox + int(xs.min()), oy + int(ys.min())))
    return res


def _band_like(pa, pb, lc, axis):
    if pa.sum() < 3000:
        return False
    if axis == 0:
        ea = int(pa.any(axis=1).sum())
        eb = int(pb.any(axis=1).sum())
    else:
        ea = int(pa.any(axis=0).sum())
        eb = int(pb.any(axis=0).sum())
    if ea >= 0.85 * eb:
        return False
    return float(np.median(lc[pa])) < 0.62 * float(np.median(lc[pb]))


def _seg_dist(p, a, b):
    ab = b - a
    t = float(np.dot(p - a, ab)) / max(float(np.dot(ab, ab)), 1e-9)
    t = min(1.0, max(0.0, t))
    return float(math.hypot(*(p - (a + t * ab))))


def _hull(pts):
    p = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def half(q):
        o = []
        for pt in q:
            while len(o) >= 2:
                a, b = o[-2], o[-1]
                if (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0]) <= 0:
                    o.pop()
                else:
                    break
            o.append(pt)
        return o

    hpts = half(p)[:-1] + half(p[::-1])[:-1]
    if len(hpts) < 3:
        return None
    return np.array(hpts, np.float64)


def _ext_corners(m, maxside=720.0):
    """Harris corner count on the convex hull of a candidate piece mask."""
    h, w = m.shape
    if h < 24 or w < 24:
        return 0
    s = maxside / max(h, w)
    if s < 1.0:
        img = Image.fromarray(m.astype(np.uint8) * 255).resize(
            (max(2, int(round(w * s))), max(2, int(round(h * s)))), Image.BILINEAR)
        a = np.asarray(img, np.float32) / 255.0
    else:
        a = m.astype(np.float32)
    hh, ww = a.shape
    diag = math.hypot(hh, ww)
    sm = max(2, int(round(0.004 * diag)))

    def box(arr, rb):
        ii = np.zeros((hh + 1, ww + 1), np.float64)
        ii[1:, 1:] = arr.cumsum(0, dtype=np.float64).cumsum(1)
        ya = np.minimum(np.arange(hh) + rb + 1, hh)
        yb = np.maximum(np.arange(hh) - rb, 0)
        xa = np.minimum(np.arange(ww) + rb + 1, ww)
        xb = np.maximum(np.arange(ww) - rb, 0)
        return ii[np.ix_(ya, xa)] - ii[np.ix_(yb, xa)] - ii[np.ix_(ya, xb)] + ii[np.ix_(yb, xb)]

    a = box(a, sm) / float((2 * sm + 1) ** 2)
    gx = np.zeros_like(a)
    gy = np.zeros_like(a)
    gx[:, 1:-1] = a[:, 2:] - a[:, :-2]
    gy[1:-1, :] = a[2:, :] - a[:-2, :]
    r = max(3, int(round(0.012 * diag)))
    sxx, syy, sxy = box(gx * gx, r), box(gy * gy, r), box(gx * gy, r)
    det = sxx * syy - sxy * sxy
    tr = sxx + syy
    disc = np.maximum(tr * tr - 4.0 * det, 0.0)
    R = 0.5 * (tr - np.sqrt(disc))
    top = float(R.max())
    if top <= 0.0:
        return 0
    pk = 2
    sw = sliding_window_view(np.pad(R, pk, constant_values=-1e30), (2 * pk + 1, 2 * pk + 1))
    cand = (R >= 0.05 * top) & (R >= sw.max(axis=(2, 3)))
    ys, xs = np.nonzero(cand)
    if ys.size == 0:
        return 0
    o = np.argsort(R[ys, xs])[::-1][:200]
    dmin2 = max(36.0, (0.045 * diag) ** 2)
    pts = []
    for i in o.tolist():
        y, x = float(ys[i]), float(xs[i])
        if all((y - q[0]) ** 2 + (x - q[1]) ** 2 >= dmin2 for q in pts):
            pts.append((y, x))
            if len(pts) >= 24:
                break
    bm = a >= 0.5
    by, bx = np.nonzero(bm & ~_erode(bm))
    hp = _hull(np.stack([bx, by], 1).astype(np.float64))
    if hp is None or len(hp) < 4:
        return 0
    dtol = max(2.5, 0.02 * diag)
    n = 0
    for py, px in pts:
        p = np.array([px, py])
        d = min(_seg_dist(p, hp[i], hp[(i + 1) % len(hp)]) for i in range(len(hp)))
        if d <= dtol:
            n += 1
    return n


def _seam_split(m, ox, oy, lum, page_area, depth=0):
    total = float(m.sum())
    h, w = m.shape
    if depth >= 5 or total < 4000.0 or h < 60 or w < 60:
        return [(m, ox, oy)]
    ys, xs = np.nonzero(m)
    bh = int(ys.max()) - int(ys.min()) + 1
    bw = int(xs.max()) - int(xs.min()) + 1
    elong = max(bh, bw) / max(1.0, min(bh, bw))
    if _quad_lock(m, page_area):
        return [(m, ox, oy)]
    if elong < 2.2 and total < 0.25 * page_area:
        return [(m, ox, oy)]
    lc = lum[oy:oy + h, ox:ox + w]
    for axis in (0, 1):
        mm = m if axis == 0 else m.T
        ll = lc if axis == 0 else lc.T
        cut = _find_seam(np.ascontiguousarray(mm), np.ascontiguousarray(ll), lo=35.0)
        if cut is None:
            cut = _find_seam(np.ascontiguousarray(mm), np.ascontiguousarray(ll), lo=25.0)
        if cut is None:
            continue
        if axis == 0:
            rr = np.arange(h)[:, None]
            pa = m & (rr < cut[None, :])
            pb = m & (rr >= cut[None, :])
        else:
            cc = np.arange(w)[None, :]
            pa = m & (cc < cut[:, None])
            pb = m & (cc >= cut[:, None])
        sa, sb = float(pa.sum()), float(pb.sum())
        need = max(0.15 * total, 400.0)
        ba = _band_like(pa, pb, lc, axis)
        bb = _band_like(pb, pa, lc, axis)
        if min(sa, sb) < need:
            ok = (sa < sb and ba and sb >= need) or (sb < sa and bb and sa >= need)
            if not ok:
                continue
        if ba and not bb and sb > sa:
            sides = ((pb, ox, oy),)
        elif bb and not ba and sa > sb:
            sides = ((pa, ox, oy),)
        else:
            sides = ((pa, ox, oy), (pb, ox, oy))
        out = []
        for pm, px, py in sides:
            cs_list = _crop_subs(pm, px, py, page_area)
            for cs in cs_list:
                nc = _ext_corners(cs[0])
                fill = float(cs[0].sum()) / float(cs[0].size)
                bh, bw = cs[0].shape
                ar = max(bh, bw) / max(1.0, min(bh, bw))
                if not ((4 <= nc <= 10 and fill >= 0.80) or (nc < 4 and fill >= 0.90)):
                    # A low-fill side with photo-like corner count is usually a
                    # still-butted multi-photo composite (e.g. half of a 2x2 grid
                    # whose own pair still has an internal gap). Recurse instead of
                    # rejecting the whole cut; the depth cap bounds recursion and
                    # the pre-extract sanity gate drops composites that refuse to
                    # sub-split.
                    if nc <= 14 and ar <= _MAX_PIECE_AR:
                        out += _seam_split(cs[0], cs[1], cs[2], lum, page_area, depth + 1)
                        continue
                    out = None
                    break
                if ar > _MAX_PIECE_AR or float(cs[0].sum()) < max(400.0, _MIN_PIECE_FRAC * total):
                    out = None
                    break
                out.append(cs)
            if out is None:
                break
        if out is None:
            continue
        parts = []
        for cm, cx, cy in out:
            parts += _seam_split(cm, cx, cy, lum, page_area, depth + 1)
        return parts or [(m, ox, oy)]
    return [(m, ox, oy)]


def _quad_corners(pts):
    s1 = pts[:, 0] + pts[:, 1]
    s2 = pts[:, 0] - pts[:, 1]
    tl = pts[np.argmin(s1)]
    br = pts[np.argmax(s1)]
    tr = pts[np.argmax(s2)]
    bl = pts[np.argmin(s2)]
    return np.array([tl, tr, br, bl], np.float64)


def _isect(n1, d1, n2, d2):
    try:
        return np.linalg.solve(np.array([n1, n2], np.float64), np.array([d1, d2], np.float64))
    except np.linalg.LinAlgError:
        return None


def _refine(pts, q, inset):
    q = q.astype(np.float64).copy()
    for _ in range(3):
        c = q.mean(axis=0)
        lines = []
        for i in range(4):
            a, b = q[i], q[(i + 1) % 4]
            e = b - a
            L = math.hypot(e[0], e[1])
            if L < 1e-6:
                break
            e /= L
            n = np.array([e[1], -e[0]])
            mid = (a + b) / 2
            if float(np.dot(n, mid - c)) < 0:
                n = -n
            vals = pts @ n
            mx = float(vals.max())
            span = max(10.0, 0.04 * L)
            band = vals >= mx - span
            d = None
            if int(band.sum()) >= 40:
                bv = vals[band]
                hist, edges = np.histogram(bv, bins=24)
                kbin = int(np.argmax(hist))
                c0 = 0.5 * (edges[kbin] + edges[kbin + 1])
                sel2 = np.abs(bv - c0) <= 2.0
                if int(sel2.sum()) >= 40:
                    d = float(np.median(bv[sel2])) - inset
            if d is None:
                sel = vals >= mx - 3.0
                if int(sel.sum()) < 40:
                    sel = vals >= np.percentile(vals, 99.5) - 1.0
                d = float(np.median(vals[sel])) - inset
            lines.append((n, d))
        if len(lines) < 4:
            return q
        nq = []
        ok = True
        for i in range(4):
            n1, d1 = lines[(i - 1) % 4]
            n2, d2 = lines[i]
            p = _isect(n1, d1, n2, d2)
            if p is None:
                ok = False
                break
            nq.append(p)
        if not ok:
            return q
        q = np.array(nq)
    return q


def _axis_crop(full, up, wsc, fx0, fy0, fx1, fy1, pts):
    """Group-5 fallback: a piece whose fitted quad angle was rejected as a
    silhouette artifact is cropped as a plain axis-aligned rectangle of its own
    mask extent. Never rotates (which is what produced tilted crops with white
    corner triangles and clipped content) and never masks by the ragged mask
    (which punched holes into photos that contain large white regions)."""
    bx0 = int(max(0, math.floor(pts[:, 0].min() / wsc)) + fx0)
    by0 = int(max(0, math.floor(pts[:, 1].min() / wsc)) + fy0)
    bx1 = int(min(full.width, math.ceil(pts[:, 0].max() / wsc)) + fx0)
    by1 = int(min(full.height, math.ceil(pts[:, 1].max() / wsc)) + fy0)
    if bx1 - bx0 < 100 or by1 - by0 < 100:
        return None
    return full.crop((bx0, by0, bx1, by1))


def _extract(full, part, inv_scale, bg, tol, inset_full):
    m, ox, oy = part
    ys, xs = np.nonzero(m)
    if len(ys) < 50:
        return None
    sx0, sx1 = int(xs.min()) + ox, int(xs.max()) + ox
    sy0, sy1 = int(ys.min()) + oy, int(ys.max()) + oy
    fx0 = max(0, int(math.floor((sx0 - 4) * inv_scale)))
    fy0 = max(0, int(math.floor((sy0 - 4) * inv_scale)))
    fx1 = min(full.width, int(math.ceil((sx1 + 5) * inv_scale)))
    fy1 = min(full.height, int(math.ceil((sy1 + 5) * inv_scale)))
    if fx1 - fx0 < 40 or fy1 - fy0 < 40:
        return None
    crop = full.crop((fx0, fy0, fx1, fy1))
    wsc = min(1.0, 4200.0 / max(crop.size))
    if wsc < 1.0:
        ww = max(1, round(crop.width * wsc))
        wh = max(1, round(crop.height * wsc))
        work = crop.resize((ww, wh), Image.BILINEAR)
    else:
        ww, wh = crop.size
        work = crop
    sub = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    up = np.asarray(Image.fromarray(sub.astype(np.uint8) * 255).resize((ww, wh), Image.BILINEAR)) > 100
    arr = np.asarray(work, np.float32)
    dist = np.sqrt(((arr - bg) ** 2).sum(axis=2))
    fmask = (dist > tol) & up
    pts = np.argwhere(fmask)[:, ::-1].astype(np.float64)
    if len(pts) < 300:
        return None
    if len(pts) > 250000:
        pts = pts[:: len(pts) // 250000 + 1]
    inset = max(1.0, inset_full * wsc)
    q = _refine(pts, _quad_corners(pts), inset)
    xs_work = wsc
    q[:, 0] = q[:, 0] / xs_work + fx0
    q[:, 1] = q[:, 1] / xs_work + fy0
    # Flatbed scans are flat: no perspective. Deskew to the fitted rectangle's
    # angle (undoes placement rotation) and crop an axis-aligned rectangle.
    a = math.atan2(q[1][1] - q[0][1], q[1][0] - q[0][0])
    if abs(math.degrees(a)) > _MAX_DESKEW_DEG:
        return _axis_crop(full, up, wsc, fx0, fy0, fx1, fy1, pts)
    ca, sa = math.cos(a), math.sin(a)
    cx = float(q[:, 0].mean())
    cy = float(q[:, 1].mean())
    rot = np.empty_like(q)
    dx = q[:, 0] - cx
    dy = q[:, 1] - cy
    rot[:, 0] = cx + dx * ca + dy * sa
    rot[:, 1] = cy - dx * sa + dy * ca
    rx0 = max(0, int(math.floor(rot[:, 0].min())))
    ry0 = max(0, int(math.floor(rot[:, 1].min())))
    rx1 = min(full.width, int(math.ceil(rot[:, 0].max())))
    ry1 = min(full.height, int(math.ceil(rot[:, 1].max())))
    if rx1 - rx0 < 100 or ry1 - ry0 < 100:
        return None
    box = (rx0, ry0, rx1, ry1)
    # Mask everything outside the photo quad to white so neighbouring photos /
    # background caught by the axis-aligned bbox of a deskewed (rotated) crop
    # never bleed into the output.
    qm = Image.new("L", full.size, 0)
    qd = ImageDraw.Draw(qm)
    qd.polygon([(float(p[0]), float(p[1])) for p in q], fill=255)
    if abs(a) >= 1e-4:
        qm = qm.rotate(math.degrees(a), resample=Image.NEAREST, center=(cx, cy))
    qm = qm.crop(box)
    out = full.crop(box) if abs(a) < 1e-4 else \
        full.rotate(math.degrees(a), resample=Image.BICUBIC, center=(cx, cy)).crop(box)
    white = Image.new("RGB", out.size, (255, 255, 255))
    return Image.composite(out, white, qm)


def _auto_upright(img):
    small = img.copy()
    small.thumbnail((500, 500))
    g = np.asarray(small.convert("L"), np.float32)
    gx = float(np.abs(np.diff(g, axis=1)).mean())
    gy = float(np.abs(np.diff(g, axis=0)).mean())
    if gx > gy * 1.3:
        best = None
        best_score = -1e9
        for rot in (90, 270):
            cand = img.transpose(Image.ROTATE_90 if rot == 90 else Image.ROTATE_270)
            cs = cand.copy()
            cs.thumbnail((500, 500))
            cg = np.asarray(cs.convert("L"), np.float32)
            h = cg.shape[0]
            band = max(1, h // 8)
            score = float(cg[:band].mean()) - float(cg[-band:].mean())
            if score > best_score:
                best_score = score
                best = cand
        return best
    return img


def analyze_scan(path, tol=16.0, maxdim=3200, inset_full=4.0):
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    if im.mode != "RGB":
        im = im.convert("RGB")
    scale = maxdim / max(im.size)
    if scale < 1.0:
        small = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.BILINEAR)
    else:
        scale = 1.0
        small = im
    arr = np.asarray(small, np.float32)
    b = max(4, arr.shape[0] // 50)
    ring = np.concatenate([arr[:b].reshape(-1, 3), arr[-b:].reshape(-1, 3),
                           arr[:, :b].reshape(-1, 3), arr[:, -b:].reshape(-1, 3)])
    bg = np.median(ring, axis=0)
    dist = np.sqrt(((arr - bg) ** 2).sum(axis=2))
    mimg = Image.fromarray((dist > tol).astype(np.uint8) * 255)
    padded = ImageOps.expand(mimg, border=4, fill=0)
    opened = padded.filter(ImageFilter.MinFilter(5)).filter(ImageFilter.MaxFilter(5))
    opened = ImageOps.crop(opened, border=4)
    mask = np.asarray(opened) > 127
    mask = _majority(mask)
    return {"path": path, "im": im, "small": small, "scale": scale, "bg": bg,
            "tol": tol, "inset_full": inset_full, "mask": mask}


def _piece_ok(qm, page_area):
    """Pre-extract sanity gate for a candidate piece mask (B1)."""
    ys, xs = np.nonzero(qm)
    if len(ys) == 0:
        return False
    bh = int(ys.max()) - int(ys.min()) + 1
    bw = int(xs.max()) - int(xs.min()) + 1
    ar = max(bh, bw) / max(1.0, min(bh, bw))
    fill = float(qm.sum()) / float(bh * bw)
    if ar > _MAX_PIECE_AR:
        return False
    if qm.sum() < max(400.0, _MIN_PIECE_FRAC * page_area):
        return False
    # low fill usually means a still-butted multi-photo composite; allow the
    # exotic large-butted case only for page-dominating components where the
    # seam split already tried its best (B1 keeps smaller low-fill shards out)
    if fill < 0.55 and qm.sum() < 0.5 * page_area:
        return False
    return True


def _dedupe_parts(parts, page_area, frac=0.5):
    """Drop pieces whose mask is mostly covered by already-kept pieces (A3).

    Sorts by area descending; a piece is a duplicate when > frac of its own
    mask overlaps the union of kept pieces. Guarantees every kept piece
    contributes mostly-unique pixels — no photo is emitted twice.
    """
    kept = []
    kept_union = None
    for qm, qx, qy in sorted(parts, key=lambda p: -float(p[0].sum())):
        ys, xs = np.nonzero(qm)
        if len(ys) == 0:
            continue
        cand = np.zeros((int(ys.max() - ys.min()) + 1, int(xs.max() - xs.min()) + 1), bool)
        cand[ys - ys.min(), xs - xs.min()] = True
        cx = qx + int(xs.min())
        cy = qy + int(ys.min())
        overlap = 0
        if kept_union is not None:
            ox0 = max(cx, kept_union[1])
            oy0 = max(cy, kept_union[2])
            ox1 = min(cx + cand.shape[1], kept_union[1] + kept_union[0].shape[1])
            oy1 = min(cy + cand.shape[0], kept_union[2] + kept_union[0].shape[0])
            if ox1 > ox0 and oy1 > oy0:
                ov = kept_union[0][oy0 - kept_union[2]:oy1 - kept_union[2],
                                   ox0 - kept_union[1]:ox1 - kept_union[1]]
                cc = cand[oy0 - cy:oy1 - cy, ox0 - cx:ox1 - cx]
                overlap = int((ov & cc).sum())
        if overlap > frac * float(qm.sum()):
            continue
        kept.append((qm, qx, qy))
        if kept_union is None:
            kept_union = [cand.copy(), cx, cy]
        else:
            # widen union canvas
            ux0, uy0 = min(cx, kept_union[1]), min(cy, kept_union[2])
            ux1 = max(cx + cand.shape[1], kept_union[1] + kept_union[0].shape[1])
            uy1 = max(cy + cand.shape[0], kept_union[2] + kept_union[0].shape[0])
            nu = np.zeros((uy1 - uy0, ux1 - ux0), bool)
            nu[kept_union[2] - uy0:kept_union[2] - uy0 + kept_union[0].shape[0],
               kept_union[1] - ux0:kept_union[1] - ux0 + kept_union[0].shape[1]] |= kept_union[0]
            nu[cy - uy0:cy - uy0 + cand.shape[0], cx - ux0:cx - ux0 + cand.shape[1]] |= cand
            kept_union = [nu, ux0, uy0]
    return kept


def _extract_core(st, mask, rotate):
    """Component split + watershed + seam split + extraction for one mask."""
    im = st["im"]
    lab = _label(mask)
    min_area = 0.003 * mask.size
    comps = []
    for l in np.unique(lab):
        if l == 0:
            continue
        ys, xs = np.nonzero(lab == l)
        area = int(xs.size)
        if area < min_area:
            continue
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        if area / max(1.0, (x1 - x0 + 1) * (y1 - y0 + 1)) < 0.35:
            continue
        cm = lab[y0:y1 + 1, x0:x1 + 1] == l
        comps.append((cm, x0, y0, area))
    parts = []
    page_area = float(mask.size)
    med = float(np.median([c[3] for c in comps]) if comps else 0.0)
    arr = np.asarray(st["small"], np.float32)
    lum = arr @ np.array([0.299, 0.587, 0.114], np.float32)
    for cm, x0, y0, area in comps:
        subs = _split(cm, x0, y0)
        if len(subs) == 1 and area >= 0.04 * page_area and area > 1.15 * med:
            subs = _watershed(*subs[0]) or subs
        for pm, px, py in subs:
            for qm, qx, qy in _seam_split(pm, px, py, lum, page_area):
                if qm.sum() >= min_area * 0.5:
                    parts.append((qm, qx, qy))
    parts = _dedupe_parts([p for p in parts if _piece_ok(p[0], page_area)], page_area)
    inv = 1.0 / st["scale"]
    out = []
    for part in parts:
        img = _extract(im, part, inv, st["bg"], st["tol"], st["inset_full"])
        if img is None:
            continue
        if _mostly_blank(img):
            continue
        out.append(_auto_upright(img) if rotate else img)
    return out, parts


def _mostly_blank(img, frac=0.95):
    """B2: drop extractions that are ~all background/white."""
    t = img.convert("RGB")
    t.thumbnail((200, 200))
    a = np.asarray(t, np.float32)
    # near-white OR near-gray-uniform background: count pixels close to white
    white = (a > 235).all(axis=2)
    return float(white.mean()) > frac


def extract_parts(st, mask=None, rotate=False):
    if mask is None:
        mask = st["mask"]
    out, parts = _extract_core(st, mask, rotate)
    # B3: coverage check + one bounded retry. Coverage = kept-piece mask area
    # vs total photo-area in the original mask. If a big chunk was dropped or
    # left merged, retry with a lower tolerance (rescues white-on-white photos
    # whose mask collapses at the user tolerance) and keep the better result.
    page_area = float(mask.size)
    cov = sum(float(p[0].sum()) for p in parts) / page_area if parts else 0.0
    if cov < 0.75 and mask is st["mask"] and st["tol"] > 10.0:
        try:
            st2 = analyze_scan(st["path"], tol=max(10.0, st["tol"] - 4.0),
                               maxdim=3200, inset_full=st["inset_full"])
            out2, parts2 = _extract_core(st2, st2["mask"], rotate)
            cov2 = sum(float(p[0].sum()) for p in parts2) / page_area if parts2 else 0.0
            if cov2 > cov and len(out2) > len(out):
                out = out2
        except Exception:
            pass
    return out


def split_scan(path, tol=16.0, rotate=False, maxdim=3200, inset_full=4.0):
    st = analyze_scan(path, tol=tol, maxdim=maxdim, inset_full=inset_full)
    return extract_parts(st, rotate=rotate)


def process_state(st, mask, outdir, fmt, quality, rotate):
    imgs = extract_parts(st, mask, rotate=rotate)
    stem = os.path.splitext(os.path.basename(st["path"]))[0]
    written = []
    for i, img in enumerate(imgs, 1):
        name = "%s_%02d.%s" % (stem, i, "png" if fmt == "png" else "jpg")
        p = os.path.join(outdir, name)
        if fmt == "png":
            img.save(p)
        else:
            img.save(p, quality=quality, subsampling=0)
        written.append(p)
    return written


def process_file(path, outdir, fmt, quality, tol, rotate):
    st = analyze_scan(path, tol=tol)
    return process_state(st, st["mask"], outdir, fmt, quality, rotate)


def open_folder(path):
    if sys.platform.startswith("win"):
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def _label_components(m):
    """Label 4-connected True regions of a boolean mask. Returns (labels, n)."""
    h, w = m.shape
    parent = []

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    runs = []
    prev = []
    for y in range(h):
        idx = np.flatnonzero(m[y])
        if idx.size == 0:
            prev = []
            continue
        brk = np.flatnonzero(np.diff(idx) > 1)
        starts = np.r_[idx[0], idx[brk + 1]]
        ends = np.r_[idx[brk], idx[-1]] + 1
        cur = []
        for s, e in zip(starts.tolist(), ends.tolist()):
            rid = len(parent)
            parent.append(rid)
            runs.append((y, s, e, rid))
            cur.append((s, e, rid))
        for ps, pe, prid in prev:
            for s, e, rid in cur:
                if s < pe and ps < e:
                    ra, rb = find(prid), find(rid)
                    if ra != rb:
                        parent[rb] = ra
        prev = cur
    lbl = np.zeros((h, w), np.int32)
    ids = {}
    for y, x0, x1, rid in runs:
        r = find(rid)
        if r not in ids:
            ids[r] = len(ids) + 1
        lbl[y, x0:x1] = ids[r]
    return lbl, len(ids)


def mask_editor(parent, st):
    """Modal mask review dialog. Returns (action, mask) with action in {"ok", "skip", "cancel"}."""
    import base64
    import io
    import tkinter as tk
    from tkinter import ttk, messagebox

    try:
        from PIL import ImageTk
    except ImportError:
        ImageTk = None

    small = st["small"]
    w0, h0 = small.size
    sw = max(320, parent.winfo_screenwidth() - 140)
    sh = max(240, parent.winfo_screenheight() - 260)
    dscale = min(sw / w0, sh / h0, 1.0)
    dw, dh = max(1, int(w0 * dscale)), max(1, int(h0 * dscale))

    base = np.asarray(small.resize((dw, dh), Image.BILINEAR), np.float32)
    orig = st["mask"].copy()
    mask = st["mask"].copy()
    red = np.array([255.0, 48.0, 48.0], np.float32)

    top = tk.Toplevel(parent)
    top.title("Review mask - %s" % os.path.basename(st["path"]))
    top.transient(parent)
    res = {"action": "cancel", "mask": None}

    tool = tk.StringVar(value="paint")
    brush = tk.IntVar(value=16)
    comps = []
    lbl = {"a": None}
    drag = {"i": None, "mode": None, "orig": None, "start": None,
            "rect": None, "ghost": None}

    photo = {"im": None}

    def make_photo(img):
        if ImageTk is not None:
            return ImageTk.PhotoImage(img)
        b = io.BytesIO()
        img.save(b, "PNG", compress_level=1)
        try:
            return tk.PhotoImage(data=base64.b64encode(b.getvalue()))
        except tk.TclError:
            b = io.BytesIO()
            img.save(b, "GIF")
            return tk.PhotoImage(data=base64.b64encode(b.getvalue()))

    def show(img):
        photo["im"] = make_photo(img)

    mid = ttk.Frame(top)
    mid.pack(fill="both", expand=True)
    canvas = tk.Canvas(mid, width=dw, height=dh, cursor="crosshair",
                       highlightthickness=0)
    canvas.pack(side="left", fill="both", expand=True, padx=(8, 4), pady=(8, 2))

    tw = 150
    side = tk.Canvas(mid, width=tw + 24, highlightthickness=0)
    sb = ttk.Scrollbar(mid, orient="vertical", command=side.yview)
    side.config(yscrollcommand=sb.set)
    sb.pack(side="right", fill="y", pady=(8, 2))
    side.pack(side="right", fill="y", pady=(8, 2))
    thumbs = []

    def set_cursor():
        canvas.config(cursor="fleur" if tool.get() == "move" else "crosshair")
        draw(compute=True)

    rect_id = {"id": None}

    def recompute_comps():
        comps.clear()
        lbl["a"] = None
        if not mask.any():
            return
        L, n = _label_components(mask)
        lbl["a"] = L
        for k in range(1, n + 1):
            ys, xs = np.nonzero(L == k)
            comps.append({"root": k,
                          "rect": [float(xs.min()) * dscale, float(ys.min()) * dscale,
                                   float(xs.max() + 1) * dscale,
                                   float(ys.max() + 1) * dscale]})

    def update_previews():
        side.delete("all")
        thumbs.clear()
        y = 6
        for i, c in enumerate(comps):
            xa = int(round(c["rect"][0] / dscale))
            ya = int(round(c["rect"][1] / dscale))
            xb = int(round(c["rect"][2] / dscale))
            yb = int(round(c["rect"][3] / dscale))
            pad = max(2, int(0.05 * max(xb - xa, yb - ya)))
            crop = small.crop((max(0, xa - pad), max(0, ya - pad),
                               min(w0, xb + pad), min(h0, yb + pad)))
            th = max(1, int(round(tw * crop.height / crop.width)))
            im = crop.resize((tw, th), Image.BILINEAR)
            pi = make_photo(im)
            thumbs.append(pi)
            side.create_image(12, y, image=pi, anchor="nw")
            side.create_text(tw + 12, y + th + 4, text=str(i + 1), anchor="n",
                             fill="#22d3ee", font="TkDefaultFont 10 bold")
            y += th + 26
        side.config(scrollregion=(0, 0, tw + 24, max(y, 1)))

    def draw(compute=False):
        md = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255)
                        .resize((dw, dh), Image.BILINEAR)) > 100
        out = base.copy()
        out[md] = out[md] * 0.45 + red * 0.55
        show(Image.fromarray(out.astype(np.uint8)))
        canvas.delete("all")
        canvas.create_image(0, 0, image=photo["im"], anchor="nw")
        if compute:
            recompute_comps()
            update_previews()
        for i, c in enumerate(comps):
            r = c["rect"]
            t = "c%d" % i
            canvas.create_rectangle(*r, outline="#22d3ee", width=2,
                                    dash=(5, 4), tags=t)
            if r[1] >= 16:
                canvas.create_text(r[0] + 2, r[1] - 3, text=str(i + 1),
                                   anchor="sw", fill="#22d3ee",
                                   font="TkDefaultFont 10 bold", tags=t)
            else:
                canvas.create_text(r[0] + 2, r[1] + 3, text=str(i + 1),
                                   anchor="nw", fill="#22d3ee",
                                   font="TkDefaultFont 10 bold", tags=t)
            if tool.get() == "move":
                for _, hx, hy in rect_handles(r):
                    canvas.create_rectangle(hx - 4, hy - 4, hx + 4, hy + 4,
                                            fill="#22d3ee", outline="", tags=t)

    undo = []

    def push_undo():
        undo.append(mask.copy())
        if len(undo) > 8:
            undo.pop(0)

    cur = {"x": 0, "y": 0, "down": False}
    rstart = {"xy": None}

    def mxy(x, y):
        return (int(min(max(x / dscale, 0), w0 - 1)),
                int(min(max(y / dscale, 0), h0 - 1)))

    def rect_handles(r):
        x0, y0, x1, y1 = r
        return (("nw", x0, y0), ("n", (x0 + x1) / 2.0, y0), ("ne", x1, y0),
                ("w", x0, (y0 + y1) / 2.0), ("e", x1, (y0 + y1) / 2.0),
                ("sw", x0, y1), ("s", (x0 + x1) / 2.0, y1), ("se", x1, y1))

    def handle_at(r, x, y):
        for name, hx, hy in rect_handles(r):
            if abs(x - hx) <= 5 and abs(y - hy) <= 5:
                return name
        return None

    def hit_comp(x, y):
        best = None
        for i, c in enumerate(comps):
            r = c["rect"]
            h = handle_at(r, x, y)
            if h is not None:
                key = (0, (r[2] - r[0]) * (r[3] - r[1]))
            elif r[0] <= x <= r[2] and r[1] <= y <= r[3]:
                key = (1, (r[2] - r[0]) * (r[3] - r[1]))
            else:
                continue
            if best is None or key < best[0]:
                best = (key, i, h if h is not None else "move")
        if best is None:
            return None
        return best[1], best[2]

    def stamp(mx, my, val):
        r = max(1, int(round(brush.get() / dscale)))
        x0, x1 = max(0, mx - r), min(w0, mx + r + 1)
        y0, y1 = max(0, my - r), min(h0, my + r + 1)
        if x1 <= x0 or y1 <= y0:
            return
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask[y0:y1, x0:x1][(yy - my) ** 2 + (xx - mx) ** 2 <= r * r] = val

    def stroke(a, b, val):
        n = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
        if n == 0:
            stamp(a[0], a[1], val)
            return
        for t in range(n + 1):
            stamp(int(round(a[0] + (b[0] - a[0]) * t / n)),
                  int(round(a[1] + (b[1] - a[1]) * t / n)), val)

    def painting():
        return tool.get() in ("paint", "erase")

    def on_press(ev):
        if tool.get() == "move":
            hit = hit_comp(ev.x, ev.y)
            if hit is None:
                return
            i, m = hit
            drag["i"] = i
            drag["mode"] = m
            drag["orig"] = list(comps[i]["rect"])
            drag["start"] = (ev.x, ev.y)
            canvas.itemconfigure("c%d" % i, state="hidden")
            return
        if painting():
            push_undo()
            cur["down"] = True
            cur["x"], cur["y"] = ev.x, ev.y
            stamp(*mxy(ev.x, ev.y), 1 if tool.get() == "paint" else 0)
            draw()
        else:
            rstart["xy"] = (ev.x, ev.y)

    def shape_rect(ev):
        ox0, oy0, ox1, oy1 = drag["orig"]
        m = drag["mode"]
        if m == "move":
            dx = min(max(ev.x - drag["start"][0], -ox0), dw - ox1)
            dy = min(max(ev.y - drag["start"][1], -oy0), dh - oy1)
            return [ox0 + dx, oy0 + dy, ox1 + dx, oy1 + dy]
        sep = max(4.0, 2 * dscale)
        r = [ox0, oy0, ox1, oy1]
        if "w" in m:
            r[0] = min(max(ev.x, 0), ox1 - sep)
        if "e" in m:
            r[2] = max(min(ev.x, dw), ox0 + sep)
        if "n" in m:
            r[1] = min(max(ev.y, 0), oy1 - sep)
        if "s" in m:
            r[3] = max(min(ev.y, dh), oy0 + sep)
        return r

    def update_ghost(r):
        if drag["ghost"] is None:
            items = [canvas.create_rectangle(*r, outline="#22d3ee", width=2,
                                             tags="ghost")]
            for _, hx, hy in rect_handles(r):
                items.append(canvas.create_rectangle(hx - 4, hy - 4, hx + 4, hy + 4,
                                                     fill="#22d3ee", outline="",
                                                     tags="ghost"))
            items.append(canvas.create_text(r[0] + 2, r[1], text=str(drag["i"] + 1),
                                            anchor="sw", fill="#22d3ee",
                                            font="TkDefaultFont 10 bold",
                                            tags="ghost"))
            drag["ghost"] = items
            return
        canvas.coords(drag["ghost"][0], *r)
        for i, (_, hx, hy) in enumerate(rect_handles(r), 1):
            canvas.coords(drag["ghost"][i], hx - 4, hy - 4, hx + 4, hy + 4)
        canvas.coords(drag["ghost"][-1], r[0] + 2, r[1])

    def on_drag(ev):
        if tool.get() == "move":
            if drag["start"] is None:
                return
            drag["rect"] = shape_rect(ev)
            update_ghost(drag["rect"])
            return
        if painting():
            if not cur["down"]:
                return
            stroke(mxy(cur["x"], cur["y"]), mxy(ev.x, ev.y),
                   1 if tool.get() == "paint" else 0)
            cur["x"], cur["y"] = ev.x, ev.y
            draw()
        elif rstart["xy"] is not None:
            x0, y0 = rstart["xy"]
            if rect_id["id"] is None:
                rect_id["id"] = canvas.create_rectangle(x0, y0, ev.x, ev.y,
                                                        outline="#22d3ee", width=2)
            else:
                canvas.coords(rect_id["id"], x0, y0, ev.x, ev.y)

    def on_release(ev):
        if tool.get() == "move":
            if drag["start"] is None:
                return
            i, orig = drag["i"], drag["orig"]
            drag["i"] = drag["mode"] = drag["orig"] = drag["start"] = None
            if drag["ghost"] is not None:
                canvas.delete("ghost")
                drag["ghost"] = None
            r = drag["rect"]
            drag["rect"] = None
            if r is None or r == orig:
                draw(compute=True)
                return
            xa, xb = sorted((int(round(r[0] / dscale)), int(round(r[2] / dscale))))
            ya, yb = sorted((int(round(r[1] / dscale)), int(round(r[3] / dscale))))
            xa, xb = max(0, min(xa, w0)), max(0, min(xb, w0))
            ya, yb = max(0, min(ya, h0)), max(0, min(yb, h0))
            if xb - xa < 2 or yb - ya < 2:
                draw(compute=True)
                return
            push_undo()
            mask[lbl["a"] == comps[i]["root"]] = 0
            mask[ya:yb, xa:xb] = 1
            draw(compute=True)
            return
        if painting():
            cur["down"] = False
            draw(compute=True)
            return
        if rstart["xy"] is None:
            return
        x0, y0 = rstart["xy"]
        rstart["xy"] = None
        if rect_id["id"] is not None:
            canvas.delete(rect_id["id"])
            rect_id["id"] = None
        ax, ay = mxy(x0, y0)
        bx, by = mxy(ev.x, ev.y)
        xa, xb = sorted((ax, bx))
        ya, yb = sorted((ay, by))
        if xb - xa < 2 or yb - ya < 2:
            return
        push_undo()
        mask[ya:yb + 1, xa:xb + 1] = 1 if tool.get() == "rectadd" else 0
        draw(compute=True)

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)

    def undo_last():
        if undo:
            mask[:] = undo.pop()
            draw(compute=True)

    def reset():
        push_undo()
        mask[:] = orig
        draw(compute=True)

    bar = ttk.Frame(top, padding=(8, 2))
    bar.pack(fill="x")
    ttk.Radiobutton(bar, text="Paint", value="paint", variable=tool,
                    command=set_cursor).pack(side="left")
    ttk.Radiobutton(bar, text="Erase", value="erase", variable=tool,
                    command=set_cursor).pack(side="left", padx=(6, 0))
    ttk.Radiobutton(bar, text="Add rectangle", value="rectadd", variable=tool,
                    command=set_cursor).pack(side="left", padx=(12, 0))
    ttk.Radiobutton(bar, text="Remove rectangle", value="rectdel", variable=tool,
                    command=set_cursor).pack(side="left", padx=(6, 0))
    ttk.Radiobutton(bar, text="Shape mask", value="move", variable=tool,
                    command=set_cursor).pack(side="left", padx=(12, 0))
    ttk.Label(bar, text="Brush size:").pack(side="left", padx=(12, 2))
    ttk.Spinbox(bar, from_=2, to=120, textvariable=brush, width=5).pack(side="left")
    ttk.Button(bar, text="Undo", command=undo_last).pack(side="left", padx=(12, 0))
    ttk.Button(bar, text="Reset mask", command=reset).pack(side="left", padx=(6, 0))

    ttk.Label(top, text="Red = photo areas; one dashed numbered box per detected picture "
                        "(previews on the right). Paint or erase with the brush, add/remove "
                        "whole rectangles, or use Shape mask to drag each box and its "
                        "edge/corner handles so the red covers exactly that picture. "
                        "Click OK to split this scan with the mask shown.",
              wraplength=max(320, dw), justify="left", padding=(8, 0)).pack(fill="x")

    bot = ttk.Frame(top, padding=(8, 6))
    bot.pack(fill="x")

    def ok():
        if not mask.any():
            messagebox.showwarning(APP_NAME, "The mask is empty. Paint or add a rectangle first, "
                                                 "or use Skip to leave this scan unsplit.", parent=top)
            return
        res["action"] = "ok"
        res["mask"] = mask.copy()
        top.destroy()

    def skip():
        res["action"] = "skip"
        top.destroy()

    def cancel():
        res["action"] = "cancel"
        top.destroy()

    ttk.Button(bot, text="Cancel batch", command=cancel).pack(side="right")
    ttk.Button(bot, text="Skip this scan", command=skip).pack(side="right", padx=6)
    ttk.Button(bot, text="OK - split with this mask", command=ok).pack(side="right", padx=6)

    top.protocol("WM_DELETE_WINDOW", cancel)
    top.bind("<Return>", lambda e: ok())
    top.bind("<Escape>", lambda e: cancel())
    set_cursor()
    top.update_idletasks()
    top.geometry("+%d+%d" % (max(0, parent.winfo_rootx() + (parent.winfo_width() - top.winfo_width()) // 2),
                             max(0, parent.winfo_rooty() + 60)))

    def grab(n=0):
        if not top.winfo_exists():
            return
        try:
            top.grab_set()
        except tk.TclError:
            if n < 40:
                top.after(50, lambda: grab(n + 1))

    grab()
    try:
        top.wait_window()
    except tk.TclError:
        pass
    return res["action"], res["mask"]


def collect_inputs(inputs):
    files = []
    for x in inputs:
        if os.path.isdir(x):
            files += [os.path.join(x, f) for f in sorted(os.listdir(x))
                      if os.path.splitext(f)[1].lower() in EXTS]
        elif os.path.isfile(x):
            files.append(x)
    return files


def run_cli(a):
    files = collect_inputs(a.inputs)
    if not files:
        print("No image files found.")
        return 1
    total = 0
    for f in files:
        outdir = a.out or os.path.join(os.path.dirname(os.path.abspath(f)), "split")
        os.makedirs(outdir, exist_ok=True)
        try:
            written = process_file(f, outdir, a.format, a.quality, a.tolerance, a.rotate)
        except Exception as e:
            print("%s -> ERROR: %s" % (f, e))
            continue
        total += len(written)
        print("%s -> %d photos -> %s" % (os.path.basename(f), len(written), outdir))
    print("Done: %d photos." % total)
    return 0


def run_gui():
    try:
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox
    except ImportError:
        print("tkinter is not available.\n"
              "  Debian/Ubuntu: sudo apt install python3-tk\n"
              "  Or use the command line:  python photosplit.py <files-or-folder>")
        return 1
    root = tk.Tk()
    root.title("%s v%s" % (APP_NAME, APP_VERSION))
    root.minsize(560, 480)

    files = []

    top = ttk.Frame(root, padding=8)
    top.pack(fill="x")
    lb = tk.Listbox(top, height=10, selectmode="extended", activestyle="none")
    sb = ttk.Scrollbar(top, orient="vertical", command=lb.yview)
    lb.configure(yscrollcommand=sb.set)
    lb.pack(side="left", fill="both", expand=True)
    sb.pack(side="left", fill="y")

    btns = ttk.Frame(root, padding=(8, 0))
    btns.pack(fill="x")
    b_add = ttk.Button(btns, text="Add images...")
    b_rem = ttk.Button(btns, text="Remove selected")
    b_clr = ttk.Button(btns, text="Clear")
    b_add.pack(side="left")
    b_rem.pack(side="left", padx=6)
    b_clr.pack(side="left")

    opts = ttk.Frame(root, padding=8)
    opts.pack(fill="x")
    ttk.Label(opts, text="Output folder:").grid(row=0, column=0, sticky="w")
    out_var = tk.StringVar()
    e_out = ttk.Entry(opts, textvariable=out_var)
    e_out.grid(row=0, column=1, sticky="ew", padx=6)
    b_out = ttk.Button(opts, text="Browse...")
    b_out.grid(row=0, column=2)
    opts.columnconfigure(1, weight=1)
    ttk.Label(opts, text="Format:").grid(row=1, column=0, sticky="w", pady=(6, 0))
    fmt = tk.StringVar(value="PNG (lossless)")
    cb = ttk.Combobox(opts, textvariable=fmt, state="readonly", width=22,
                      values=["PNG (lossless)", "JPEG (quality 95)", "JPEG (quality 88)"])
    cb.grid(row=1, column=1, sticky="w", padx=6, pady=(6, 0))
    rot = tk.BooleanVar(value=False)
    ttk.Checkbutton(opts, text="Auto-rotate sideways photos (experimental)", variable=rot)\
        .grid(row=2, column=0, columnspan=3, sticky="w", pady=(6, 0))
    ttk.Label(opts, text="Background tolerance:").grid(row=3, column=0, sticky="w", pady=(6, 0))
    tol = tk.IntVar(value=20)
    ttk.Spinbox(opts, from_=8, to=60, textvariable=tol, width=6).grid(row=3, column=1, sticky="w", padx=6, pady=(6, 0))
    step = tk.BooleanVar(value=False)
    ttk.Checkbutton(opts, text="Pause per scan (review and correct each mask before splitting)",
                    variable=step).grid(row=4, column=0, columnspan=3, sticky="w", pady=(6, 0))
    sub = tk.BooleanVar(value=False)
    ttk.Checkbutton(opts, text="One subfolder per scan", variable=sub)\
        .grid(row=5, column=0, columnspan=3, sticky="w", pady=(6, 0))
    orig = tk.BooleanVar(value=False)
    ttk.Checkbutton(opts, text="Also copy the original scan", variable=orig)\
        .grid(row=6, column=0, columnspan=3, sticky="w", pady=(6, 0))

    act = ttk.Frame(root, padding=8)
    act.pack(fill="x")
    b_go = ttk.Button(act, text="Split photos")
    b_go.pack(side="left")
    b_pause = ttk.Button(act, text="Pause", state="disabled")
    b_pause.pack(side="left", padx=(6, 0))
    pb = ttk.Progressbar(act, length=260, maximum=1)
    pb.pack(side="left", padx=10)
    b_open = ttk.Button(act, text="Open output folder", state="disabled")
    b_open.pack(side="left")

    status_var = tk.StringVar(value="Add scanned images, choose an output folder, then Split.")
    ttk.Label(root, textvariable=status_var, padding=(8, 4), wraplength=540, justify="left")\
        .pack(fill="x")

    q = queue.Queue()
    busy = [False]
    pause_flag = threading.Event()
    abort = threading.Event()
    review = {"st": None, "result": None, "ev": threading.Event()}
    flags = {"step": False, "tol": 20, "rot": False, "sub": False, "orig": False}

    def refresh():
        lb.delete(0, "end")
        for f, n in files:
            lb.insert("end", f + ("   -> %d photos" % n if n is not None else ""))

    def set_busy(v):
        busy[0] = v
        for b in (b_add, b_rem, b_clr, b_go, b_out):
            b.configure(state="disabled" if v else "normal")
        b_pause.configure(state="normal" if v else "disabled", text="Pause")
        if not v:
            pause_flag.clear()
        if not v:
            b_open.configure(state="normal" if files and any(n is not None for _, n in files) else "disabled")

    def add():
        p = filedialog.askopenfilenames(parent=root, title="Choose scans",
                                        filetypes=[("Images", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp"),
                                                   ("All files", "*.*")])
        for x in p:
            files.append([x, None])
        if files and not out_var.get():
            out_var.set(os.path.join(os.path.dirname(files[0][0]), "split"))
        refresh()
        b_open.configure(state="disabled")

    def rem():
        sel = set(lb.curselection())
        for i in sorted(sel, reverse=True):
            del files[i]
        refresh()

    def clr():
        files.clear()
        refresh()
        b_open.configure(state="disabled")

    def browse():
        d = filedialog.askdirectory(parent=root)
        if d:
            out_var.set(d)

    def worker(f, qly, outdir):
        total = 0
        errs = []
        cancelled = False
        for i, (path, _) in enumerate(list(files)):
            if abort.is_set():
                cancelled = True
                break
            name = os.path.basename(path)
            try:
                st = analyze_scan(path, tol=flags["tol"])
            except Exception as e:
                errs.append("%s: %s" % (name, e))
                q.put(("file", i, 0, None))
                continue
            mask = st["mask"]
            if flags["step"] or pause_flag.is_set():
                review["st"] = st
                review["result"] = None
                review["ev"].clear()
                q.put(("review", i, name, None))
                review["ev"].wait()
                act, m = review["result"] or ("cancel", None)
                review["st"] = None
                pause_flag.clear()
                if act == "skip":
                    q.put(("status", "Skipped %s." % name))
                    q.put(("file", i, 0, None))
                    continue
                if act == "cancel":
                    cancelled = True
                    q.put(("file", i, 0, None))
                    break
                mask = m
            try:
                q.put(("status", "Splitting %s..." % name))
                destdir = outdir
                if flags["sub"]:
                    destdir = os.path.join(outdir, os.path.splitext(name)[0])
                    os.makedirs(destdir, exist_ok=True)
                written = process_state(st, mask, destdir, f, qly or 95, flags["rot"])
                if flags["orig"]:
                    shutil.copy2(path, os.path.join(destdir, name))
                total += len(written)
                q.put(("file", i, len(written), None))
            except Exception as e:
                errs.append("%s: %s" % (name, e))
                q.put(("file", i, 0, None))
        if cancelled:
            msg = "Cancelled: %d photos saved to %s." % (total, outdir) if total else "Cancelled."
        else:
            msg = "Done: %d photos saved to %s" % (total, outdir)
        if errs:
            msg += "\nErrors: " + "; ".join(errs)
        q.put(("done", total, outdir if (total and not cancelled) else None, msg))

    def go():
        if not files:
            messagebox.showinfo(APP_NAME, "Add some images first.")
            return
        fmtmap = {"PNG (lossless)": ("png", None), "JPEG (quality 95)": ("jpeg", 95),
                  "JPEG (quality 88)": ("jpeg", 88)}
        f, qly = fmtmap[fmt.get()]
        outdir = os.path.abspath(out_var.get() or os.path.join(os.path.dirname(files[0][0]), "split"))
        try:
            os.makedirs(outdir, exist_ok=True)
        except OSError as e:
            status_var.set("Cannot create output folder: %s" % e)
            return
        snap()
        set_busy(True)
        abort.clear()
        status_var.set("Working...")
        pb.configure(maximum=len(files), value=0)
        threading.Thread(target=worker, args=(f, qly, outdir), daemon=True).start()

    def toggle_pause():
        if pause_flag.is_set():
            pause_flag.clear()
            b_pause.configure(text="Pause")
            status_var.set("Resumed - continuing automatically.")
        else:
            pause_flag.set()
            b_pause.configure(text="Resume")
            status_var.set("Pause armed: the next scan will open for review.")

    def snap():
        flags["step"] = bool(step.get())
        flags["tol"] = int(tol.get())
        flags["rot"] = bool(rot.get())
        flags["sub"] = bool(sub.get())
        flags["orig"] = bool(orig.get())

    def poll():
        try:
            snap()
            while True:
                item = q.get_nowait()
                if item[0] == "file":
                    _, i, n, _ = item
                    files[i][1] = n
                    pb.configure(value=i + 1)
                    refresh()
                elif item[0] == "status":
                    status_var.set(item[1])
                elif item[0] == "review":
                    _, i, name, _ = item
                    b_pause.configure(text="Pause")
                    status_var.set("Reviewing %s - correct the mask, then click OK to split." % name)
                    pb.configure(value=i)
                    act, m = mask_editor(root, review["st"])
                    review["result"] = (act, m)
                    review["ev"].set()
                elif item[0] == "done":
                    _, total, outdir, msg = item
                    status_var.set(msg)
                    pb.configure(value=pb["maximum"])
                    set_busy(False)
                    if outdir:
                        b_open.configure(state="normal")
                    root.title("%s v%s - %d photos" % (APP_NAME, APP_VERSION, total))
        except queue.Empty:
            pass
        root.after(120, poll)

    def on_close():
        if busy[0]:
            if not messagebox.askyesno(APP_NAME, "A batch is running. Abort and quit?"):
                return
            abort.set()
            if review["st"] is not None:
                review["result"] = ("cancel", None)
                review["ev"].set()
        root.destroy()

    b_add.configure(command=add)
    b_rem.configure(command=rem)
    b_clr.configure(command=clr)
    b_out.configure(command=browse)
    b_go.configure(command=go)
    b_pause.configure(command=toggle_pause)
    b_open.configure(command=lambda: open_folder(out_var.get() or "."))
    root.protocol("WM_DELETE_WINDOW", on_close)
    poll()
    root.mainloop()
    return 0


def main():
    ap = argparse.ArgumentParser(prog="photosplit",
                                 description="Split scanned pages containing one or more photos into individual photos.")
    ap.add_argument("inputs", nargs="*", help="image files and/or folders (no args = open the GUI)")
    ap.add_argument("-o", "--out", help="output folder (default: <folder of each image>/split)")
    ap.add_argument("--format", choices=["png", "jpeg"], default="png")
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality (default 95)")
    ap.add_argument("--tolerance", type=float, default=16.0,
                    help="background color tolerance, higher = more aggressive (default 16)")
    ap.add_argument("--rotate", action="store_true", help="try to auto-rotate sideways photos (experimental)")
    a = ap.parse_args()
    if a.inputs:
        sys.exit(run_cli(a))
    sys.exit(run_gui())


if __name__ == "__main__":
    main()
