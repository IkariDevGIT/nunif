# Qwen-Image-2512 GGUF/BNB T2I Image 4-steps pipeline
#
# With RTX 5070 Ti, it can generate a 512x512 image in 6 seconds and a 1328x1328 image in 13 seconds,
# --model gguf_2: requires 10GB VRAM
# --model gguf_4: requires 16GB VRAM
# --model bnb_4:  requires 13GB VRAM

import os

# Disable XET transfer because my home router keeps freezing
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_EMERGENCY_RETRY"] = "1"

import argparse
import math
import subprocess
import sys
from datetime import datetime

import torch
import torchvision.transforms.functional as TF
from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    GGUFQuantizationConfig,
    QwenImageImg2ImgPipeline,
    QwenImagePipeline,
    QwenImageTransformer2DModel,
)
from diffusers.utils import load_image
from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory

TORCH_DTYPE = torch.bfloat16
QWEN_IMAGE_ID = "unsloth/Qwen-Image-2512-unsloth-bnb-4bit"
MODEL_URL = {
    "gguf_2": "https://huggingface.co/unsloth/Qwen-Image-2512-GGUF/blob/main/qwen-image-2512-Q2_K.gguf",
    "gguf_4": "https://huggingface.co/unsloth/Qwen-Image-2512-GGUF/blob/main/qwen-image-2512-Q4_K_M.gguf",
}


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
    parser.add_argument("--size", type=str, default="1328x1328", help="output image resolution")
    parser.add_argument("--output-dir", "-o", type=str, default=default_output_dir, help="output image resolution")
    parser.add_argument(
        "--model",
        type=str,
        default="gguf_2",
        choices=["gguf_2", "gguf_4", "bnb_4"],
        help="gguf_2: 10GB VRAM, guff_4: 16GB VRAM, bnb_4: 13GB VRAM",
    )
    parser.add_argument("--reference-image", "-i", type=str, default=None, help="Image for i2i")
    parser.add_argument("--strength", type=float, default=0.8, help="denosing strength for reference image")
    args = parser.parse_args()

    width, height = parse_size(args.size)
    os.makedirs(args.output_dir, exist_ok=True)

    if not args.reference_image:
        reference_image = None
        pipeline_class = QwenImagePipeline
        lora_weight_name = "Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors"
        num_inference_steps = 4
    else:
        reference_image = preprocess_image(load_image(args.reference_image), base_size=max(width, height))
        pipeline_class = QwenImageImg2ImgPipeline
        # Use 8 steps because strength reduces the number of steps
        lora_weight_name = "Qwen-Image-2512-Lightning-8steps-V1.0-bf16.safetensors"
        num_inference_steps = 8

    device_gpu = torch.device("cuda")
    if args.model in MODEL_URL:
        transformer = QwenImageTransformer2DModel.from_single_file(
            MODEL_URL[args.model],
            quantization_config=GGUFQuantizationConfig(compute_dtype=TORCH_DTYPE),
            torch_dtype=TORCH_DTYPE,
            config=QWEN_IMAGE_ID,
            subfolder="transformer",
            low_cpu_mem_usage=True,
            device_map="cpu",
        )
        model_kwargs = dict(transformer=transformer)
    else:
        model_kwargs = dict()

    scheduler_config = {
        "base_image_seq_len": 256,
        "base_shift": math.log(3),
        "invert_sigmas": False,
        "max_image_seq_len": 8192,
        "max_shift": math.log(3),
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": None,
        "stochastic_sampling": False,
        "time_shift_type": "exponential",
        "use_beta_sigmas": False,
        "use_dynamic_shifting": True,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
    }
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
    pipeline_kwargs = dict(
        scheduler=scheduler,
        torch_dtype=TORCH_DTYPE,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    pipeline_kwargs.update(model_kwargs)
    pipe = pipeline_class.from_pretrained(QWEN_IMAGE_ID, **pipeline_kwargs)
    pipe.load_lora_weights("lightx2v/Qwen-Image-2512-Lightning", weight_name=lora_weight_name)
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
                prompt=prompt_text + "\n. Ultra HD, 4K.",
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                true_cfg_scale=1.0,
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
            output_file = os.path.join(args.output_dir, datetime.now().strftime("qwen-%Y%m%d-%H%M%S.png"))
            image.save(output_file)
            print(f"Save: {output_file}")
            show_image(output_file)

        except (EOFError, KeyboardInterrupt):
            sys.exit(0)


if __name__ == "__main__":
    main()
