# Photo Scan Auto Splitter

**Photo Scan Auto Splitter** takes a scanned page that contains one or more photos and
automatically cuts it into the individual photos, deskewing/straightening each one and
saving it as its own image file.

It detects the page background color, finds the photos on the page (even when several
overlap or touch), and perspective-corrects each photo to a clean rectangle. Built for
people digitizing old photo albums, binder pages, or scanner beds with multiple prints.

## Download

Grab the ready-to-run Windows executable from the
[**v1.0 release page**](https://github.com/GIT-PROTONU/photo-scan-auto-splitter/releases/tag/v1.0):

- `PhotoScanAutoSplitter-1.0-Windows-x64.exe` — fully self-contained (Windows 10+, 64-bit).
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

Output files are named `<scan name>_01.png`, `<scan name>_02.png`, ...

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
   watershed seeded from distance-transform peaks.
4. Each region's quadrilateral is refined against the full-resolution image and the
   photo is perspective-transformed to a rectangle (deskewed).

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