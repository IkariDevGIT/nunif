# Animation Unpack/Pack Tool Specification

This toolset provides a way to decompose animated images (WebP, GIF, APNG) into individual frames and repack them back into an animation while preserving metadata.

## 1. Commands

### anim_unpack

Decomposes an animated image into a sequence of PNG frames and generates a configuration file.

**Usage:**
```bash
python -m nunif.cli.anim_unpack -i <input_file> -o <output_directory>
```

**Arguments:**
- `-i, --input`: Path to the input animated image (WebP, GIF, or APNG/PNG).
- `-o, --output`: Directory where frames and `config.yml` will be saved.

**Output:**
- Individual frames saved as `000000.png`, `000001.png`, etc.
- `config.yml` containing global and frame-specific metadata.

---

### anim_pack

Combines an image sequence into an animated image.

**Usage:**
```bash
python -m nunif.cli.anim_pack -i <input_directory> -o <output_file> [options]
```

**Arguments:**
- `-i, --input`: Directory containing image frames.
- `-o, --output`: Path to the output animated image. The format is determined by the file extension (`.webp`, `.gif`, `.png` for APNG).
- `-c, --config`: (Optional) Path to `config.yml`. If omitted, it looks for `config.yml` in the input directory.
- `--loop`: (Optional) Override the number of loops (0 for infinite).
- `--fps`: (Optional) Override the frame rate (frames per second).
- `--duration`: (Optional) Override the duration of each frame in milliseconds.
- `--format`: (Optional) Override the output format. Supported values: `webp`, `gif`, `apng`, `png`. If provided, it takes precedence over the output file extension.

**Behavior:**
- If `config.yml` exists, it uses the metadata for frame durations and ordering.
- If `config.yml` is missing, it loads all images in the directory sorted by filename and uses a default duration (100ms).
- Supports flexible extension matching: even if a frame is saved as `.webp` or `.jpg` instead of the original `.png`, it will be matched based on the filename key.
- If the filename contains a six digit sequence number, it will be used as the filename key in `config.yml`. e.g, `000003_RLF_cross.png` -> `000003`

---

## 2. Configuration Format (config.yml)

The configuration file is a YAML file that stores metadata required to reconstruct the animation.

**Schema:**
```yaml
format: string       # Original format (WEBP, GIF, PNG)
loop: integer         # Number of loops (0 = infinite)
background: [r, g, b, a] # Global background color (list of RGBA values)
frames:
  - file: string     # Filename of the frame (e.g., "000000")
    duration: integer # Duration in milliseconds
    disposal: integer # (GIF only) Disposal method (0-3)
    transparency: int # (GIF only) Transparency index
```

---

## 3. Supported Formats

- **WebP**: Supports full alpha channel and variable durations.
- **GIF**: Supports paletted colors (max 256) and transparency index. The tool handles RGBA to Paletted conversion during packing.
- **APNG**: Supports full alpha channel and variable durations. Output as `.png`.

## 4. Limitations

### 16-bit Grayscale (Depth Images)
- **Unpacking**: The tool fully supports unpacking 16-bit grayscale animations (`I;16` mode) into 16-bit PNG frames without losing precision.
- **Packing**: Due to limitations in the current Pillow library, creating multi-frame animations (APNG) from 16-bit grayscale images is not supported (only the first frame is saved). 
- **Format Conversion**: Packing 16-bit images into WebP or GIF will result in downscaling to 8-bit depth.

### GIF Color Quality
- **Re-quantization**: When packing images into a GIF, the tool recalculates an optimized palette (max 256 colors) for each frame. This may result in slight color shifts or "shimmering" artifacts between frames compared to the original GIF.
- **Transparency**: Transparency is handled by mapping a specific palette index (usually 255).

## 5. Design Philosophy

- **Frames as True Color**: Frames are unpacked as RGBA PNGs to allow easy editing in modern image editors.
- **Config as Source of Truth**: Metadata is stored in an external YAML file to ensure settings (like frame durations) are not lost even if the image files are modified or re-exported in different formats.
- **Flexible Matching**: Matching frames by filename (excluding extension) allows users to change image formats during editing without breaking the reconstruction process.

## NOTICE

by gemini-cli
