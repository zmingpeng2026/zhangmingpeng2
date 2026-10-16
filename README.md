---
license: apache-2.0
base_model: OpenMOSS-Team/MOSS-Image-Base
base_model_relation: finetune
language:
- en
- zh
pipeline_tag: text-to-image
tags:
- moss-image
- diffusers
- text-to-image
---
# MOSS-Image-SFT

<div align="center">
  <p>
    <a href="https://github.com/OpenMOSS/MOSS-Image">
    <img class="dark:hidden mx-auto" src="assets/lockup-light.svg" alt="MOSS-Image" width="520">
    <img class="hidden dark:block mx-auto" src="assets/lockup-dark.svg" alt="MOSS-Image" width="520">
    </a>
  </p>
  <p>
    <a href="https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base"><img alt="Hugging Face · MOSS-Image-Base" src="https://img.shields.io/badge/🤗%20Hugging%20Face-MOSS--Image--Base-ffc107?style=flat-square"></a>
    &nbsp;
    <a href="https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT"><img alt="Hugging Face · MOSS-Image-SFT" src="https://img.shields.io/badge/🤗%20Hugging%20Face-MOSS--Image--SFT-ffc107?style=flat-square"></a>
    &nbsp;
    <a href="https://modelscope.cn/models/openmoss/MOSS-Image-Base"><img alt="ModelScope · MOSS-Image-Base" src="assets/modelscope-base-badge.svg"></a>
    &nbsp;
    <a href="https://modelscope.cn/models/openmoss/MOSS-Image-SFT"><img alt="ModelScope · MOSS-Image-SFT" src="assets/modelscope-sft-badge.svg"></a>
    &nbsp;
    <a href="https://github.com/OpenMOSS/MOSS-Image"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-Code-181717?style=flat-square&amp;logo=github&amp;logoColor=white"></a>
    &nbsp;
    <a href="https://huggingface.co/spaces/OpenMOSS-Team/MOSS-Image"><img alt="Hugging Face Spaces · Demo" src="https://img.shields.io/badge/🤗%20Spaces-Demo-4f46e5?style=flat-square"></a>
    &nbsp;
    <a href="https://openmoss.github.io/MOSS-Image/"><img alt="Homepage" src="https://img.shields.io/badge/🌐%20Project-Homepage-2563eb?style=flat-square"></a>
    &nbsp;
    <a href="https://github.com/OpenMOSS/MOSS-Image/blob/main/paper/MOSS_Image_Technical_Report.pdf"><img alt="Technical Report" src="https://img.shields.io/badge/📄%20Paper-Technical_Report-b31b1b?style=flat-square"></a>
    &nbsp;
    <a href="https://x.com/Open_MOSS"><img alt="Follow OpenMOSS on X" src="https://img.shields.io/badge/Follow-OpenMOSS-000000?style=flat-square&amp;logo=x&amp;logoColor=white"></a>
    &nbsp;
    <a href="https://qm.qq.com/q/hphM7VhaDu"><img alt="MOSS-Image QQ 交流群：755535143" src="https://img.shields.io/badge/QQ-755535143-12B7F5?style=flat-square&amp;logo=qq&amp;logoColor=white"></a>
    &nbsp;
    <a href="https://applink.feishu.cn/client/chat/chatter/add_by_link?link_token=7a3m2007-815f-409b-9c3d-a4129f5c058e"><img alt="MOSS-Builders Feishu Community" src="assets/feishu-builders-badge.svg"></a>
    &nbsp;
    <a href="https://discord.gg/nXFWEyvbb"><img alt="MOSS-Image Discord Community" src="https://img.shields.io/badge/Discord-Community-5865F2?style=flat-square&amp;logo=discord&amp;logoColor=white"></a>
  </p>
  <p>
    <a href="https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE.md">User guide</a> &nbsp; · &nbsp;
    <a href="https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE_ZH.md">中文用户指南</a> &nbsp; · &nbsp;
    <a href="https://github.com/OpenMOSS/MOSS-Image/blob/main/comfyui/README.md">ComfyUI</a>
  </p>
</div>

![MOSS-Image-Base and MOSS-Image-SFT](assets/teaser.webp)

MOSS-Image-SFT is a supervised fine-tuned text-to-image model in the
**MOSS-Image** family from OpenMOSS, derived from
[MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base).
It uses the same architecture and inference pipeline, supporting natural-language
prompts, tag sequences, and JSON layouts for controlling text placement and
object arrangement.

## Base vs. SFT

In this example, SFT produces clearer text grouping and a more organized whiteboard layout.

| MOSS-Image-Base | MOSS-Image-SFT |
| --- | --- |
| ![MOSS-Image-Base](assets/comparison/base.webp) | ![MOSS-Image-SFT](assets/comparison/sft.webp) |

## Quick start

Use Python 3.10 or later and an NVIDIA GPU with CUDA support. Install a PyTorch
2.9 or later build compatible with your CUDA environment, then install:

```bash
pip install "diffusers>=0.38" "transformers>=5.3" "accelerate>=1.10" "safetensors>=0.6" "Pillow>=10"
```

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "OpenMOSS-Team/MOSS-Image-SFT",
    custom_pipeline="pipeline_moss_image",
    trust_remote_code=True,
    torch_dtype={"default": torch.bfloat16, "vae": torch.float32},
)
pipe.scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config)
pipe.to("cuda")

prompt = '''A wide view of a refrigerator beside a kitchen counter with a fruit bowl. A small sticky note on the refrigerator door is handwritten in Chinese:
"记得买牛奶和鸡蛋！
牛奶两盒，鸡蛋一打。
顺便带一袋面包。"'''

image = pipe(
    prompt,
    height=1024,
    width=1024,
    num_inference_steps=50,
    guidance_scale=3.0,
    uncond_skip_layers=[22],
    generator=torch.Generator("cpu").manual_seed(42),
).images[0]
image.save("refrigerator_note.png")
```

The model and custom pipeline code are downloaded and cached automatically. You
can also pass a complete local model directory to `from_pretrained`.
The transformer and text encoder use bfloat16; the VAE and initial noise use
float32. The bundled VAE uses the FLUX.2 small decoder.

For the Gradio interface, CPU offload, and compilation options, follow the
[project quick start](https://github.com/OpenMOSS/MOSS-Image#quick-start).
For ComfyUI, search for **MOSS-Image** in **ComfyUI Manager** to install the
native plugin. Integration into the official ComfyUI repository is in progress.
See the [installation and usage guide](https://github.com/OpenMOSS/MOSS-Image/blob/main/comfyui/README.md)
and [example workflow](https://github.com/OpenMOSS/MOSS-Image/blob/main/comfyui/moss-image/example_workflows/moss-image.json).

## Sampling settings

| Setting | Default |
| --- | --- |
| Sampler | Flow-matching Euler |
| Steps | 50 |
| CFG | 3 |
| Shift | 5 |
| Layer skipping | Negative branch, layer index 22 (zero-based) |
| Negative prompt | Empty |

The default guidance uses two transformer forwards per step. Pass
`uncond_skip_layers=[]` to disable negative-branch skipping, or set
`negative_prompt` to provide a negative prompt. Python API image dimensions must
be positive multiples of 16.

See the [user guide](https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE.md)
for resolution, sampling, prompt, and LoRA training recommendations.

## Prompt enhancer skill

The [MOSS-Image Prompt Enhancer](https://github.com/OpenMOSS/MOSS-Image/blob/main/skills/moss-image-prompt-enhancer/SKILL.md) turns
image ideas into detailed captions or spatial Layout prompts. It is designed for
GPT-6-family agents with local file and Python tools. The bundled script converts
pixel-coordinate drafts into compact JSON with integer centered coordinates;
it does not call an image-generation service.

Install the [skill folder](https://github.com/OpenMOSS/MOSS-Image/tree/main/skills/moss-image-prompt-enhancer) in your agent's skills
directory, then invoke `$moss-image-prompt-enhancer` with your image idea, desired
caption or Layout output, and any canvas dimensions. Paste the compiled result
into MOSS-Image's positive prompt; for Layout, use the same generation dimensions.

## Layout-to-image generation

Pass a JSON layout as the prompt to describe the scene, rendered text, objects,
and their positions. Layouts use `summary.caption` and `items`, with
`bbox_format: "yxyx"` and `bbox_coord_space: "abs_centralized"`.

With the Python API, serialize the layout using
`json.dumps(layout, ensure_ascii=False)` and pass it to `pipe`. In Gradio, paste
the JSON into the Prompt field and set the matching image dimensions.
See the [layout format and example](https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE.md#layout-prompts-experimental).

Layouts can also support controlled editing and user-friendly layout tools.
Edit text, element descriptions, and bounding boxes to adjust image content and
composition in a structured way.

![Layout control](assets/showcase/layout-bboxes.webp)

## Showcase

![Examples overview](assets/showcase/overview.webp)

![Examples overview 2](assets/showcase/overview2.webp)

## Citation

```bibtex
@misc{mossimage2026,
  title = {MOSS-Image},
  author = {{OpenMOSS Team}},
  year = {2026},
  howpublished = {\url{https://github.com/OpenMOSS/MOSS-Image}}
}
```

## License and acknowledgements

The model weights and bundled inference code are released under the
[Apache License, Version 2.0](LICENSE).

We thank the Diffusers, Qwen, and FLUX teams.
