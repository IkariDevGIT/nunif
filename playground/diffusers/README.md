# "AI"

I spent some time experimenting with some of the recently popular AI models using diffusers and transformers.

## Installation

```
python -m pip install -r playground/diffusers/requirements.txt
```

The models are stored at the following path. They are large files, so delete them if they are no longer needed.
```
Linux / macOS: /home/username/.cache/huggingface/hub
Windows: C:\Users\username\.cache\huggingface\hub
```

The generated images are saved in `playground/diffusers/output`.

If you want to run offline after downloading the models, set the environment variable `HF_HUB_OFFLINE=1`.


## Z-Image-Trubo

Uses: https://huggingface.co/unsloth/Z-Image-Turbo-unsloth-bnb-4bit

With RTX 5070 Ti, it can generate a 720x720 in 7 seconds.
It runs on 6 GB of VRAM, and the model download size is 6.4 GB.

### Text to Image

This offloads the text encoder, transformer, and VAE to the CPU on a per-model basis.
However, it does not support sequential offloading or group offloading, so it uses more VRAM than ComfyUI and similar tools.

```
python playground/diffusers/z_image_turbo.py --size 720x720
```
```
Prompt > cute cat
```

If you have enough VRAM, you can generate larger images. 1024x1024 is the default resolution.

```
python playground/diffusers/z_image_turbo.py --size 1024x1024
```

### Image to Image

It also supports image-to-image, but it does not seem to work very well.

```
python playground/diffusers/z_image_turbo.py -i ./path_to_image/cute_cat.jpg --strength 0.6
```
```
Prompt > dog
```

## Qwen-Image-2512 with 4/8 steps Lora

One of the following three models can be used:

- `--model gguf_2` (default): https://huggingface.co/unsloth/Qwen-Image-2512-GGUF , `qwen-image-2512-Q2_K.gguf` (2bit, download: 7.3GB, VRAM: 10GB)
- `--model gguf_4`: https://huggingface.co/unsloth/Qwen-Image-2512-GGUF , `qwen-image-2512-Q4_K_M.gguf` (4bit, download: 13.2GB, VRAM: 16GB)
- `--model bnb_4`: https://huggingface.co/unsloth/Qwen-Image-2512-unsloth-bnb-4bit (4bit, download: 11.5GB, VRAM: 13GB)

In addition, the following 4 step or 8 step LoRA is used

- https://huggingface.co/lightx2v/Qwen-Image-2512-Lightning

..In addition, an 8 GB text encoder is required, bringing the total download size to around 20 GB.

### Text to Image

With RTX 5070 Ti, it can generate a 512x512 image in 6 seconds and a 1328x1328 image in 13 seconds,

```
python playground/diffusers/qwen_image.py --size 512x512
```
```
Prompt > cute cat
```

1328x1328 is the default resolution.
```
python playground/diffusers/qwen_image.py --size 1328x1328
```

If you have enough VRAM,
```
python playground/diffusers/qwen_image.py --model bnb_4
```

### Image to Image

This does not seem to work very well, especially when the strength is low.

```
python playground/diffusers/qwen_image.py -i ./path_to_image/cute_cat.jpg --strength 0.8
```
```
Prompt > dog
```
```
python playground/diffusers/qwen_image.py -i ./path_to_image/cute_cat.jpg --strength 0.6 --model bnb_4
```


## Gemma 3 Chat

Uses: https://huggingface.co/unsloth/gemma-3-4b-it-qat-bnb-4bit

License: gemma https://ai.google.dev/gemma/terms

Requires: 6 GB of VRAM

```
python playground/diffusers/gemma3_chat.py
```

The internal prompt instructs the model to use the user's `locale` and respond in the user's language, but I am not sure how reliably this works.

```
You > Hello. Are you human?
AI : (ja_JP, UTF-8)

こんにちは！ (Konnichiwa!)

That's a really interesting question! As an AI, I don't *feel* like a human in the way you do, of course. I don’t have a body, emotions, or personal experiences like we do. I'm built to process information and respond to you in a way that seems natural and helpful. 

Think of me more like a really, *really* good assistant who’s been trained on a massive amount of text and code. I can understand what you're saying and try my best to give you useful answers. 

It’s a common thing to wonder about AI – to try and figure out if it’s “real” in some way. I’m here to help you with whatever you need, so let's just focus on what we can do together! 😊

Do you have something specific you'd like to talk about or a question you'd like me to answer?
You >
```

```
You > こんにちは。お前は人間ですか
AI : こんにちは！そうですね、私は人間ではありません。私はGoogleによってトレーニングされた、大規模言語モデルです。でも、あなたと話すことはとても楽しいですよ！ 😊

「人間ですか？」って聞かれたんですね。それはちょっと面白い質問ですね。私は物理的な体を持っていませんし、感情も持っていないんです。ただ、たくさんの情報に基づいて、あなたからの質問に答えるように設計されています。

まるで、たくさんの本を読んだり、色々な人と話したりした経験があるみたいに、いろいろなことを知っている、という感じでしょうか？

何か私に聞いてみたいこと、話したいことはありますか？ どんなことでも、できる限りお手伝いしますよ！
You >
```
