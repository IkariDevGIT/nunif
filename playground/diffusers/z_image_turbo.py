# Z-Image-Turbo T2I/I2I pipeline
#
# With RTX 5070 Ti, it can generate a 720x720 in 7 seconds.
# It requires 6GB VRAM.

import os

# Disable XET transfer because my home router keeps freezing
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_EMERGENCY_RETRY"] = "1"

import argparse
import subprocess
import sys
from datetime import datetime

import torch
import torchvision.transforms.functional as TF
from diffusers import (
    ZImageImg2ImgPipeline,
    ZImagePipeline,
)
from diffusers.utils import load_image
from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory

TORCH_DTYPE = torch.bfloat16


def show_image(file_path: str):
    env = os.environ.copy()
    env.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

    fd = subprocess.DEVNULL
    options = {"stderr": fd, "stdout": fd, "stdin": fd, "env": env}
    if sys.platform == "win32":
        subprocess.Popen(["start", "", file_path], shell=True, **options)
    elif sys.platform == "linux":
        subprocess.Popen(["xdg-open", file_path], start_new_session=True, **options)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", file_path], **options)


def parse_size(size: str):
    if "x" in size:
        w, h = size.split("x")
        width = int(w)
        height = int(h)
    else:
        width = int(size)
        height = width

    return width, height


def preprocess_image(image, base_size=1024):
    width, height = image.size
    if width > height:
        new_width = base_size
        new_height = round((new_width / width) * height)
        new_image = TF.resize(image, size=(new_height, new_width))
        new_height = new_height // 32 * 32
        new_image = TF.center_crop(new_image, (new_height, new_width))
    else:
        new_height = base_size
        new_width = round((new_height / height) * width)
        new_image = TF.resize(image, size=(new_height, new_width))
        new_width = new_width // 32 * 32
        new_image = TF.center_crop(new_image, (new_height, new_width))

    return new_image


def main():
    default_output_dir = os.path.join(os.path.dirname(__file__), "output")

    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=str, default="1024x1024", help="output image resolution")
    parser.add_argument("--output-dir", "-o", type=str, default=default_output_dir, help="output image resolution")
    parser.add_argument("--reference-image", "-i", type=str, default=None, help="Image for i2i")
    parser.add_argument("--strength", type=float, default=0.6, help="denosing strength for reference image")
    args = parser.parse_args()

    width, height = parse_size(args.size)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.reference_image:
        reference_image = preprocess_image(load_image(args.reference_image), base_size=max(width, height))
        pipeline_class = ZImageImg2ImgPipeline
    else:
        reference_image = None
        pipeline_class = ZImagePipeline

    device_gpu = torch.device("cuda")
    pipe = pipeline_class.from_pretrained(
        "unsloth/Z-Image-Turbo-unsloth-bnb-4bit",
        torch_dtype=TORCH_DTYPE,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    pipe.enable_model_cpu_offload(device=device_gpu)
    torch.cuda.empty_cache()

    print("> Type /q to quit")
    memory_history = InMemoryHistory()
    generator = torch.Generator(device=device_gpu)
    while True:
        try:
            prompt_text = prompt("Prompt > ", history=memory_history).strip()
            if prompt_text.startswith("/q") or prompt_text == "quit":
                sys.exit(0)
            if not prompt_text:
                continue

            kwargs = dict(
                prompt=prompt_text,
                width=width,
                height=height,
                num_inference_steps=9,
                guidance_scale=1.0,
                generator=generator.manual_seed(42),
            )
            if reference_image is not None:
                kwargs.update(
                    dict(
                        image=reference_image,
                        width=reference_image.width,
                        height=reference_image.height,
                        strength=args.strength,
                    )
                )
            image = pipe(**kwargs).images[0]
            output_file = os.path.join(args.output_dir, datetime.now().strftime("z-%Y%m%d-%H%M%S.png"))
            image.save(output_file)
            print(f"Save: {output_file}")
            show_image(output_file)

        except (EOFError, KeyboardInterrupt):
            sys.exit(0)


if __name__ == "__main__":
    main()
