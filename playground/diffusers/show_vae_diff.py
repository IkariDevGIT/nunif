# Show VAE encode -> decode degradation result
#
# pip3 install --upgrade diffusers[torch]
#
# python -m playground.diffusers.show_vae_diff -i path_to_image.png

import argparse

import torch
import torchvision.transforms.functional as TF
from diffusers import AutoencoderDC, AutoencoderKL, AutoencoderKLQwenImage

from nunif.utils import pil_io

VAE_OPTIONS = {
    "sd-mse": {"pretrained_model_name_or_path": "stabilityai/sd-vae-ft-mse"},
    "dcae": {
        "pretrained_model_name_or_path": "mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers",
    },
    "qwen": {
        "pretrained_model_name_or_path": "Qwen/Qwen-Image",
        "subfolder": "vae",
    },
}


def safe_pad(x, mod):
    # make the image size a multiple of 8 to avoid image size changes in encode/decode
    c, h, w = x.shape
    pad_bottom = mod - h % mod if h % mod != 0 else 0
    pad_right = mod - w % mod if w % mod != 0 else 0
    if h != 0 or w != 0:
        x = TF.pad(x, (0, 0, pad_right, pad_bottom), padding_mode="edge")
    return x


def main():
    import time

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True, help="input image file")
    parser.add_argument(
        "--vae", type=str, default="sd-mse", choices=["sd-mse", "dcae", "qwen"], help="vae pretrained model"
    )
    args = parser.parse_args()
    im, _ = pil_io.load_image_simple(args.input, color="rgb")

    if args.vae == "sd-mse":
        vae = AutoencoderKL.from_pretrained(
            torch_dtype=torch.float32, use_auth_token=False, **VAE_OPTIONS[args.vae]
        ).cuda()
        mod = 8
    elif args.vae == "dcae":
        vae = AutoencoderDC.from_pretrained(
            torch_dtype=torch.float32, use_auth_token=False, **VAE_OPTIONS[args.vae]
        ).cuda()
        mod = 32
    elif args.vae == "qwen":
        vae = AutoencoderKLQwenImage.from_pretrained(
            torch_dtype=torch.float32, use_auth_token=False, **VAE_OPTIONS[args.vae]
        ).cuda()
        mod = 32

    vae.eval()
    with torch.no_grad():
        x = pil_io.to_tensor(im)
        x = safe_pad(x, mod)
        pil_io.to_image(x).show()
        time.sleep(1)

        x = (x - 0.5) / 0.5

        if args.vae == "sd-mse":
            mu = vae.encode(x.cuda()).latent_dist.mode()
            y = vae.decode(mu).sample
        elif args.vae == "dcae":
            mu = vae.encode(x.unsqueeze(0).cuda()).latent
            y = vae.decode(mu).sample
        elif args.vae == "qwen":
            # BC1HW
            mu = vae.encode(x.unsqueeze(0).unsqueeze(2).cuda()).latent_dist.mode()
            y = vae.decode(mu).sample
            y = y.squeeze(2)

        y = y.squeeze(0).cpu()
        y = ((y * 0.5) + 0.5).clamp(0, 1)
        x = ((x * 0.5) + 0.5).clamp(0, 1)

        pil_io.to_image(y).show()
        time.sleep(1)

        diff = (x - y).abs()
        min_v = diff.min()
        max_v = diff.max()
        diff = (diff - min_v) / (max_v - min_v)

        pil_io.to_image(diff).show()


if __name__ == "__main__":
    main()
