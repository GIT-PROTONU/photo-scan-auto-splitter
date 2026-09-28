#!/usr/bin/env python3
import argparse
import math
import os
import queue
import subprocess
import sys
import threading

import numpy as np
from PIL import Image, ImageFilter, ImageOps

Image.MAX_IMAGE_PIXELS = None

EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


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


def _watershed(m, ox, oy):
    seeds = _dt_seeds(_dist_tf(m))
    if len(seeds) < 2:
        return []
    parts = _geodesic_parts(m, seeds)
    if not parts:
        parts = _voronoi_parts(m, seeds)
    return [(pm, ox, oy) for pm, _, _ in parts]


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
    wtop = math.hypot(*(q[1] - q[0]))
    wbot = math.hypot(*(q[2] - q[3]))
    hleft = math.hypot(*(q[3] - q[0]))
    hright = math.hypot(*(q[2] - q[1]))
    W = int(round(max(wtop, wbot)))
    H = int(round(max(hleft, hright)))
    if W < 100 or H < 100:
        return None
    data = tuple(v for p in (q[0], q[3], q[2], q[1]) for v in p)
    return full.transform((W, H), Image.QUAD, data, Image.BICUBIC)


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


def split_scan(path, tol=16.0, rotate=False, maxdim=3200, inset_full=4.0):
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
    opened = mimg.filter(ImageFilter.MinFilter(5)).filter(ImageFilter.MaxFilter(5))
    mask = np.asarray(opened) > 127
    mask = _majority(mask)
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
    med = float(np.median([c[3] for c in comps])) if comps else 0.0
    for cm, x0, y0, area in comps:
        subs = _split(cm, x0, y0)
        if len(subs) == 1 and area >= 0.04 * page_area and area > 1.15 * med:
            subs = _watershed(*subs[0]) or subs
        for pm, px, py in subs:
            if pm.sum() >= min_area * 0.5:
                parts.append((pm, px, py))
    inv = 1.0 / scale
    out = []
    for part in parts:
        img = _extract(im, part, inv, bg, tol, inset_full)
        if img is not None:
            out.append(_auto_upright(img) if rotate else img)
    return out


def process_file(path, outdir, fmt, quality, tol, rotate):
    imgs = split_scan(path, tol=tol, rotate=rotate)
    stem = os.path.splitext(os.path.basename(path))[0]
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


def open_folder(path):
    if sys.platform.startswith("win"):
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


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
    root.title("PhotoSplit")
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

    act = ttk.Frame(root, padding=8)
    act.pack(fill="x")
    b_go = ttk.Button(act, text="Split photos")
    b_go.pack(side="left")
    pb = ttk.Progressbar(act, length=260, maximum=1)
    pb.pack(side="left", padx=10)
    b_open = ttk.Button(act, text="Open output folder", state="disabled")
    b_open.pack(side="left")

    status_var = tk.StringVar(value="Add scanned images, choose an output folder, then Split.")
    ttk.Label(root, textvariable=status_var, padding=(8, 4), wraplength=540, justify="left")\
        .pack(fill="x")

    q = queue.Queue()
    busy = []

    def refresh():
        lb.delete(0, "end")
        for f, n in files:
            lb.insert("end", f + ("   -> %d photos" % n if n is not None else ""))

    def set_busy(v):
        for b in (b_add, b_rem, b_clr, b_go, b_out):
            b.configure(state="disabled" if v else "normal")
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

    def worker():
        fmtmap = {"PNG (lossless)": ("png", None), "JPEG (quality 95)": ("jpeg", 95),
                  "JPEG (quality 88)": ("jpeg", 88)}
        f, qly = fmtmap[fmt.get()]
        outdir = out_var.get() or os.path.join(os.path.dirname(files[0][0]), "split")
        outdir = os.path.abspath(outdir)
        try:
            os.makedirs(outdir, exist_ok=True)
        except OSError as e:
            q.put(("done", 0, None, "Cannot create output folder: %s" % e))
            return
        total = 0
        errs = []
        for i, (path, _) in enumerate(list(files)):
            try:
                written = process_file(path, outdir, f, qly or 95, int(tol.get()), bool(rot.get()))
                total += len(written)
                q.put(("file", i, len(written), None))
            except Exception as e:
                errs.append("%s: %s" % (os.path.basename(path), e))
                q.put(("file", i, 0, None))
        msg = "Done: %d photos saved to %s" % (total, outdir)
        if errs:
            msg += "\nErrors: " + "; ".join(errs)
        q.put(("done", total, outdir, msg))

    def go():
        if not files:
            messagebox.showinfo("PhotoSplit", "Add some images first.")
            return
        set_busy(True)
        status_var.set("Working...")
        pb.configure(maximum=len(files), value=0)
        threading.Thread(target=worker, daemon=True).start()

    def poll():
        try:
            while True:
                item = q.get_nowait()
                if item[0] == "file":
                    _, i, n, _ = item
                    files[i][1] = n
                    pb.configure(value=i + 1)
                    refresh()
                elif item[0] == "done":
                    _, total, outdir, msg = item
                    status_var.set(msg)
                    pb.configure(value=pb["maximum"])
                    set_busy(False)
                    if outdir:
                        b_open.configure(state="normal")
                    root.title("PhotoSplit - %d photos" % total)
        except queue.Empty:
            pass
        root.after(120, poll)

    b_add.configure(command=add)
    b_rem.configure(command=rem)
    b_clr.configure(command=clr)
    b_out.configure(command=browse)
    b_go.configure(command=go)
    b_open.configure(command=lambda: open_folder(out_var.get() or "."))
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
