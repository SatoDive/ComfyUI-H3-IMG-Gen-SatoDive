# ComfyUI-H3-IMG-Gen-SatoDive

Simple, predictable ComfyUI nodes for **MiniMax H3** image generation.

One main node, one sampling pass, and exact control over the final resolution. What you set is what you get.

## Video Tutorial & Walkthrough

<p align="center">
  <a href="https://www.youtube.com/watch?v=A53fIhsyTm8">
    <img src="https://img.youtube.com/vi/A53fIhsyTm8/maxresdefault.jpg" alt="MiniMax-H3 ComfyUI Tutorial" width="750">
  </a>
</p>

<p align="center">
  ▶️ <i>Click the image above to watch the complete step-by-step walkthrough on YouTube.</i>
</p>

## Features

- **One main node** (*H3 Image (Simple)*): prompt, up to 9 reference images, size, sampling and decode in one place.
- **Exact resolution**: pick an aspect ratio and megapixels, or type any custom width and height. With `exact_size` on, the output is exactly the size you typed.
- **No hidden passes**: no automatic refine pass and no automatic upscale.
- **Multi-reference support**: connect images to `ref_image_1` to `ref_image_9` and use `<Picture N>` in the prompt.
- **Optional detail pass**: an upscale model adds fine detail (faces, textures), then the image is shrunk back to its original size. The resolution does not change, and `detail_strength` sets how much is blended in.
- **Faster iteration**: prompt and reference encoding is reused when only the seed or sampling settings change.
- **Final Size node** (optional): exact resize, scale, crop or pad, with an optional upscale model.

## Requirements

- A ComfyUI build with the native MiniMax H3 nodes (`comfy_extras/nodes_minimax_h3.py`)
- Your H3 model, text encoder and VAE. A turbo LoRA and an upscale model are optional.

## Installation

```
cd ComfyUI/custom_nodes
git clone https://github.com/SatoDive/ComfyUI-H3-IMG-Gen-SatoDive.git
```

Or download the ZIP and extract it into `ComfyUI/custom_nodes/`. Restart ComfyUI afterwards. `__init__.py` must sit directly inside the `ComfyUI-H3-IMG-Gen-SatoDive` folder.

## Nodes

### H3 Image (Simple) - SatoDive

| Input | What it does |
| --- | --- |
| `model`, `clip`, `vae` | Your H3 model, text encoder and VAE |
| `prompt` | Your prompt. Use `<Picture N>` to point at references |
| `size_mode` | `Aspect + megapixels` or `Custom size` |
| `aspect`, `megapixels` | Used in `Aspect + megapixels` mode |
| `width`, `height` | Used in `Custom size` mode |
| `exact_size` | The model works in multiples of 32. When on, the result is resized by the few leftover pixels to match your typed size exactly |
| `seed`, `steps` | Sampling. Match `steps` to your LoRA: a 4-step LoRA needs about 4, other turbo LoRAs about 20, no LoRA about 50 |
| `lora_name`, `lora_strength` | Optional turbo LoRA |
| `sampler_name`, `scheduler` | Sampler settings |
| `detail_model`, `detail_strength` | Optional detail pass. The resolution stays the same |
| `ref_image_1` to `ref_image_9` | Optional reference images |
| `ref_image_size` | `max` = best likeness, slower and heavier on VRAM. `match` = faster, weaker likeness |

Outputs: `image`, `width`, `height` (the real size of the image).

### H3 Final Size - SatoDive

Optional last step:

- **Exact size**: stretch, crop or pad
- **Scale by factor**
- **Target megapixels**: keeps the aspect ratio
- **Keep size (detail only)**: adds detail with the upscale model, then shrinks back to the current resolution

## Workflows free to download :
https://www.patreon.com/SatoDive/posts/minimax-h3-is-171110570

## Tips

- **Faster iteration:** 4 steps with a 4-step LoRA, `ref_image_size` on `match`, no detail model, lower resolution. Switch to the heavier settings once the composition is right.
- **Detail pass:** an upscale model sharpens and adds texture, but it does not redraw faces. At very low `detail_strength` it does almost nothing.
- **VRAM:** a 4x upscale model turns a 3 MP image into a roughly 48 MP intermediate, which is slow and memory-hungry on small GPUs.

##
No other custom node pack is needed. The single-frame still latent and still decode are built in
(adapted from [ComfyUI-Fizgig-H3-Still](https://github.com/shootthesound/ComfyUI-Fizgig-H3-Still), MIT).

## Status

Work in progress. Report problems in the Issues tab.
