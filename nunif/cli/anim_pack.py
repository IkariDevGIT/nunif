# see docs/anim_pack_unpack.md
import argparse
import os
import re
import sys

import yaml
from PIL import Image


def pack(
    input_dir,
    output_path,
    config_path=None,
    loop=None,
    fps=None,
    duration=None,
    format_override=None,
):
    # Determine config path
    if config_path is None:
        potential_config = os.path.join(input_dir, "config.yml")
        if os.path.exists(potential_config):
            config_path = potential_config

    config = {}
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

    # Identify available image files in the directory
    valid_extensions = (".png", ".jpg", ".jpeg", ".webp", ".gif")
    available_files = {}
    for f in os.listdir(input_dir):
        if f.lower().endswith(valid_extensions):
            name_without_ext = os.path.splitext(f)[0]
            key = name_without_ext
            m = re.search(r"\d{6}", name_without_ext)
            if m:
                key = m.group()
            available_files[key] = f

    frame_files = []
    frame_durations = []

    # Identify image files to use
    if "frames" in config:
        for f_info in config["frames"]:
            config_filename = f_info["file"]
            name_key = os.path.splitext(config_filename)[0]

            if name_key in available_files:
                frame_files.append(os.path.join(input_dir, available_files[name_key]))
                frame_durations.append(f_info["duration"])
            else:
                print(
                    f"Warning: Frame {name_key} not found in {input_dir}, skipping.",
                    file=sys.stderr,
                )
    else:
        # No config, read all image files in name order
        sorted_keys = sorted(available_files.keys())
        for key in sorted_keys:
            frame_files.append(os.path.join(input_dir, available_files[key]))
            frame_durations.append(100)  # Default duration

    if not frame_files:
        print(f"No images found in {input_dir}", file=sys.stderr)
        return

    # Override settings from CLI
    if fps is not None:
        duration = int(1000 / fps)

    if duration is not None:
        frame_durations = [duration] * len(frame_files)

    final_loop = loop if loop is not None else config.get("loop", 0)
    final_background = config.get("background", (0, 0, 0, 0))

    # Load images
    images = []
    for f in frame_files:
        images.append(Image.open(f))

    # Determine format
    if format_override:
        fmt = format_override.upper()
        if fmt == "PNG":
            fmt = "APNG"
    else:
        # Determine format from output extension if not specified
        ext = os.path.splitext(output_path)[1].lower()
        if ext == ".webp":
            fmt = "WEBP"
        elif ext == ".png":
            fmt = "APNG"
        else:
            fmt = "GIF"

    save_args = {
        "save_all": True,
        "append_images": images[1:],
        "duration": frame_durations,
        "loop": final_loop,
        "background": final_background,
    }

    if fmt == "GIF":
        # Convert images to 'P' mode for GIF if they are RGBA
        gif_images = []
        disposals = []

        for i, img in enumerate(images):
            if img.mode == "RGBA":
                # Create a mask for transparency
                alpha = img.getchannel("A")
                img_rgb = img.convert("RGB")
                # Convert to P mode, leaving one color for transparency
                img_p = img_rgb.convert("P", palette=Image.ADAPTIVE, colors=255)
                # Apply mask: pixels with low alpha become the transparency index (255)
                mask = Image.eval(alpha, lambda a: 255 if a < 128 else 0)
                img_p.paste(255, mask)
                img_p.info["transparency"] = 255
                gif_images.append(img_p)
            elif img.mode != "P":
                gif_images.append(img.convert("P", palette=Image.ADAPTIVE))
            else:
                gif_images.append(img)

            # Get disposal from config if available
            if "frames" in config and i < len(config["frames"]):
                f_info = config["frames"][i]
                disposals.append(f_info.get("disposal", 2))
            else:
                disposals.append(2)

        save_args_gif = {
            "disposal": disposals,
            "transparency": 255,
        }

        # Remove append_images from save_args to avoid duplication
        base_args = {k: v for k, v in save_args.items() if k != "append_images"}

        gif_images[0].save(
            output_path,
            append_images=gif_images[1:],
            format="GIF",
            **base_args,
            **save_args_gif,
        )
    else:
        # WebP or APNG
        # APNG is saved with format="PNG" in Pillow
        save_fmt = "PNG" if fmt == "APNG" else fmt
        images[0].save(output_path, format=save_fmt, **save_args)

    print(f"Packed {len(frame_files)} frames into {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Pack image sequence into animated WebP/GIF")
    parser.add_argument("-i", "--input", required=True, help="Input directory containing image sequence")
    parser.add_argument("-o", "--output", required=True, help="Output animated image file")
    parser.add_argument("-c", "--config", help="Path to config.yml (optional)")
    parser.add_argument("--loop", type=int, help="Override loop count")
    parser.add_argument("--fps", type=float, help="Override FPS")
    parser.add_argument("--duration", type=int, help="Override frame duration in ms")
    parser.add_argument(
        "--format",
        choices=["webp", "gif", "apng", "png"],
        help="Override output format",
    )
    args = parser.parse_args()

    pack(
        args.input,
        args.output,
        args.config,
        args.loop,
        args.fps,
        args.duration,
        args.format,
    )


if __name__ == "__main__":
    main()
