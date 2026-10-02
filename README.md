# Photo Scan Auto Splitter

**Photo Scan Auto Splitter** takes a scanned page that contains one or more photos and
automatically cuts it into the individual photos, deskewing/straightening each one and
saving it as its own image file.

It detects the page background color, finds the photos on the page (even when several
overlap or touch), and perspective-corrects each photo to a clean rectangle. Built for
people digitizing old photo albums, binder pages, or scanner beds with multiple prints.

## Download

Grab the ready-to-run Windows executable from the
[**v1.8.2 release page**](https://github.com/GIT-PROTONU/photo-scan-auto-splitter/releases/tag/v1.8.2):

- `PhotoSplit.exe` — fully self-contained (Windows 10+, 64-bit).
  No Python, no installation: download, double-click, done.

Or run from source with Python 3.8+ (see *Running from source* below).

## Usage

### GUI

1. Run `PhotoSplit.exe` (no console window opens).
2. **Add images...** — pick one or more scans, or a whole folder of scans.
3. Choose the output folder, format (PNG lossless or JPEG), and background tolerance.
4. Click **Split photos** — progress is shown per file, then open the output folder.

Options:

| Option | Meaning |
| --- | --- |
| **Output folder** | Where the split photos are written (default: `<image folder>/split`) |
| **Format** | PNG (lossless) or JPEG quality 95/88 |
| **Auto-rotate sideways photos** | Experimental: rotates landscape-oriented scans upright |
| **Background tolerance** | 8–60 (default 20). Higher = more aggressive background removal |
| **Pause per scan** | Step mode: pause before each scan to review the detected mask |
| **One subfolder per scan** | Writes each scan's photos into `<output folder>/<scan name>/` |
| **Also copy the original scan** | Puts a copy of the original scan next to its extracted photos |

### Reviewing and correcting masks

Enable **Pause per scan** to stop before every scan: the detected mask is shown in red
over the image, with one dashed, numbered box per detected picture (1, 2, ...) and a
numbered preview of each picture in the panel on the right. Edits:

- **Paint** / **Erase** with the brush for fine-tuning.
- **Add rectangle** / **Remove rectangle** to fill or clear whole areas.
- **Shape mask** to drag a picture's box or its edge/corner handles so the red covers
  exactly that picture.
- **Undo** / **Reset mask** to step back or start over.

Then click **OK - split with this mask**. **Skip this scan** leaves the scan unsplit and
moves on; **Cancel batch** stops the run.

In automatic mode you can still click **Pause** at any time — the next scan then opens
for the same review, and the batch continues automatically after you click OK.

Output files are named `<scan name>_01.png`, `<scan name>_02.png`, ...
With **One subfolder per scan** they are grouped into one folder per scan
(e.g. `split/Scan_001/Scan_001_01.png`).

### Command line

```
photosplit.py [-h] [-o OUT] [--format {png,jpeg}] [--quality N]
              [--tolerance T] [--rotate] [inputs ...]
```

- `inputs` — image files and/or folders. With no arguments, the GUI opens.
- `-o, --out` — output folder (default: a `split` folder next to each image).
- `--format` — `png` (default) or `jpeg`.
- `--quality` — JPEG quality (default 95).
- `--tolerance` — background color tolerance, higher = more aggressive (default 16).
- `--rotate` — try to auto-rotate sideways photos (experimental).

Examples:

```
python photosplit.py "album page 1.jpg" "album page 2.jpg"
python photosplit.py ~/Scans --format jpeg --quality 95
PhotoSplit.exe C:\Scans\Holiday --tolerance 24
```

Supported input formats: JPG/JPEG, PNG, TIFF, BMP, WebP.

## How it works

1. Estimates the scanner background color from the image border and masks everything
   that differs from it by more than the tolerance.
2. Cleans the mask (morphological open + majority filter) and labels connected regions.
3. Regions that touch or overlap are separated with a geodesic distance-transform
   watershed seeded from distance-transform peaks. Photos butted edge-to-edge with no
   background gap (2x2 grids, photobooth strips) are cut apart by a seam detector that
   only accepts near-axis-parallel seam lines whose gradient evidence runs
   boundary-to-boundary across the region, so internal picture content (horizons, roof
   lines) is never mistaken for a seam — even when the photos only touch across part of
   the seam and a white gap runs along the rest of it. Every resulting piece must pass
   corner-count, fill, aspect-ratio and area sanity bounds, so small prints (passport
   photos, photobooth frames) are kept while slivers and scanner-edge lines are dropped.
4. Each region's quadrilateral is refined against the full-resolution image and the
   photo is deskewed to the rectangle's angle and cropped (flatbed scans are flat,
   so no perspective correction is applied — output is always orthographic).

## Running from source

Requires Python 3.8+ with `numpy` and `pillow`, plus `tkinter` for the GUI:

```
pip install numpy pillow
python photosplit.py            # GUI
python photosplit.py scans.jpg  # CLI
```

## Building the Windows .exe

The release executable is built with PyInstaller (onefile, windowed):

```
pyinstaller --onefile --windowed --name PhotoSplit photosplit.py
```

Build it on Windows (PyInstaller cannot cross-compile). The `PhotoSplit.spec` in this
repo reflects the exact configuration used for the v1.0 release.

## License

[MIT](LICENSE)