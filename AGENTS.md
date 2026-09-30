# AGENTS.md

Single-module app: `photosplit.py` contains everything — split algorithms, CLI, tkinter GUI, and the mask review dialog. Only 4 files are tracked: `photosplit.py`, `README.md`, `LICENSE`, `PhotoSplit.spec`.

## Commands

```bash
python3 -m py_compile photosplit.py                  # only static check available (no lint/test config, no CI)
python3 photosplit.py                                # GUI (no args = GUI)
python3 photosplit.py "Scan_20260918 (37).jpg" -o /tmp/out --tolerance 20   # CLI smoke test
python3 photosplit.py "Scan_20260918 (37).jpg" -o split --tolerance 20      # regenerate the gitignored split/ folder
```

`Scan_*.jpg` files in the repo root are gitignored personal scans — use them as smoke-test inputs, never commit them or anything in `split/`. `PhotoSplit.exe` and the portable zip are gitignored release artifacts.

## Architecture notes (non-obvious)

- Pipeline is intentionally split for the GUI review feature: `analyze_scan()` returns a state dict (`im`, `small`, `scale`, `bg`, `mask`, ...) and `extract_parts(st, mask)` does component splitting/watershed/extraction. The mask lives at the downscaled resolution (`maxdim=3200`); mask edits and their screen mapping are all at that scale. CLI path (`split_scan` → `process_file`) must stay behavior-identical.
- Photos butted edge-to-edge (no background gap, e.g. 2x2 grids or photobooth strips) are handled by `_seam_split()`: it finds near-axis-parallel gradient seam lines (slope ≤ ±0.06, column-coverage ≥ 0.68 of the span) inside a merged component and cuts along them recursively. Gates matter — they were tuned against real scans: dark text/design bands flanking a busy side are dropped (`_band_like`), depth-0 pieces must be elongated (≥2.2) or ≥25% of page, sub-pieces ≥15%, and the cut line must be ≥60% in-mask. Accepted sides must also pass `_ext_corners()` (photosplit.py): Harris-family (Shi–Tomasi) corner detection on the downscaled piece silhouette, counting distinct corners on its convex hull — real photos form ~4 external corners, so pieces with >10 corners (ragged/irregular) or <4 corners without a rectangular silhouette (bbox fill ≥0.90 rescues rounded-corner photos; smooth blobs like ellipses have fill <0.80) reject the whole cut. This is why tol=28 on 1.jpg yields 8 parts, not base's junk 9th slice. Raising coverage gates back to 0.85 misses seams with bg gaps; loosening the slope range splits shirt collars/brick mortar lines. Three more gates (tuned Sept 2026 against the same real scans, all 18 baseline cases byte-identical): (1) seam evidence must terminate boundary-to-boundary — detected gradient columns must start/end within `_SEAM_EV_EDGE_FRAC` (0.12) of the blob span; 0.02/0.04 rejects real seams (measured real ev-gaps go up to 0.110) and gating on the *fitted line* being in-mask fails too because real seams curve (one real seam's line crosses bg for 28% of its span while its evidence touches both ends) — in-mask line fraction stays at the old 0.60 (`_SEAM_SPAN_FRAC`). (2) Post-cut sanity: each piece's bbox AR ≤ `_MAX_PIECE_AR` (6.0 — covers 5-photo square rows; a 4-stacked-portrait sub-piece is the exotic casualty) and piece area ≥ max(400, `_MIN_PIECE_FRAC`*parent) — redundant with the ≥15% side gate today, kept as an invariant. (3) `_quad_lock()` refuses seam search for clean single-photo quads (4–8 corners, fill ≥0.92, AR <2.2, <25% page) — by construction a subset of the depth-0 elong/area gate, so behavior-neutral on any input; it exists as an explicit, tunable pre-filter knob, NOT a behavior change (loosening its AR to 3.0 would block stacked-wallet strips from splitting).
- GUI worker thread must NEVER read tkinter Variables (`var.get()` cross-thread silently returns wrong values — this caused real bugs). All option values are snapshotted in the main thread by `snap()` in `poll()` into the plain `flags` dict; worker reads only `flags`, `pause_flag`/`abort` (threading.Event), and the `review` dict with queue + `threading.Event` handshake (`q.put(("review", ...))` → main thread opens `mask_editor` → sets `review["result"]` + event).
- Defaults differ on purpose: CLI `--tolerance` default is 16, GUI spinbox default is 20. Keep both in sync conceptually when changing behavior.
- `mask_editor()` returns `("ok"|"skip"|"cancel", mask)`; "ok" with an empty mask is blocked. GUI messages flow through the same `q` queue as progress (item types: `file`, `status`, `review`, `done`).

## Environment quirks

- GUI needs tkinter, which pip cannot install (Debian/Ubuntu: `sudo apt install python3-tk`). Pillow alone is not enough.
- Debian's `python3-pil` ships without `ImageTk`: `mask_editor.show()` has a base64-PNG `tk.PhotoImage` fallback (needs Tk 8.6+). Do not remove it — it is what makes the dialog work here. PPM via base64 does NOT work ("couldn't recognize image data"); PNG does.
- No xvfb on this machine; GUI automation must run against the real display (`:0`). Gotchas learned the hard way: `wait_visibility()` can hang and `wait_window()` can raise if the window/app dies — `mask_editor` uses a grab-retry loop and a try/except around `wait_window`; keep that pattern.
- Synthetic `event_generate` only reaches widgets of *mapped* windows: transient Toplevels of a withdrawn root never map. For GUI tests make the root visible and wait for `winfo_viewable()` before generating events.

## Release

Windows `.exe` is built with PyInstaller (`pyinstaller PhotoSplit.spec`) on Windows only — no cross-compile. The `.spec` file is the source of truth for release config.
