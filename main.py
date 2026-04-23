import torch
from diffusers import AutoPipelineForText2Image

# Change this to switch models.
#MODEL_KEY = "sdxl-turbo"
MODEL_KEY = "sdxl-turbo"

MODELS = {
    # Full SDXL — highest quality, slow on MPS/limited RAM (~6–7 GB fp16).
    "sdxl": {
        "id": "stabilityai/stable-diffusion-xl-base-1.0",
        "steps": 25,
        "guidance": 7.5,
        "width": 864,
        "height": 1152,
    },
    # Distilled SDXL — 1–4 step inference, ~6 GB fp16. Guidance must be 0.
    "sdxl-turbo": {
        "id": "stabilityai/sdxl-turbo",
        "steps": 2,
        "guidance": 0.0,
        "width": 1024,
        "height": 1024,
    },
    # Distilled SD 2.1 — smallest/fastest, ~2 GB fp16, 512-native.
    "sd-turbo": {
        "id": "stabilityai/sd-turbo",
        "steps": 2,
        "guidance": 0.0,
        "width": 512,
        "height": 512,
    },
    # Classic SD 1.5 — ~2 GB fp16, 512-native, full CFG sampling.
    "sd15": {
        "id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "steps": 25,
        "guidance": 7.5,
        "width": 512,
        "height": 512,
    },
}

#PROMPT = "a watercolor painting of a cat astronaut floating above Mars"
#OUTPUT_PREFIX = "hello_world"
PROMPT = "Visualise how AWS IAM Policies work together with SCPs etc."
OUTPUT_PREFIX = "cloud_pc"
NUM_IMAGES = 4
SEEDS = [0, 1, 2, 3]


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    config = MODELS[MODEL_KEY]
    device = pick_device()
    print(f"Using device: {device} | model: {MODEL_KEY} ({config['id']})")

    # fp16 on MPS halves memory and speeds up inference on M-series GPUs.
    dtype = torch.float16 if device in {"mps", "cuda"} else torch.float32

    pipe = AutoPipelineForText2Image.from_pretrained(
        config["id"], torch_dtype=dtype, variant="fp16", use_safetensors=True
    )
    pipe = pipe.to(device)

    # MPS doesn't support torch.Generator — fall back to CPU generator, which
    # still produces deterministic per-seed latents.
    gen_device = "cpu" if device == "mps" else device

    for seed in SEEDS[:NUM_IMAGES]:
        generator = torch.Generator(device=gen_device).manual_seed(seed)
        image = pipe(
            PROMPT,
            num_inference_steps=config["steps"],
            guidance_scale=config["guidance"],
            width=config["width"],
            height=config["height"],
            generator=generator,
        ).images[0]
        output = f"{OUTPUT_PREFIX}_{MODEL_KEY}_seed{seed}.png"
        image.save(output)
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
