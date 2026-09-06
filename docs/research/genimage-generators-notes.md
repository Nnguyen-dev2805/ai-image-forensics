# GenImage Generator Notes

This note summarizes the GenImage validation generators used by the local
`qwen_vertex_all_val` experiment.

## Sources

- GenImage paper: https://arxiv.org/abs/2306.08571
- GenImage official repository: https://github.com/GenImage-Dataset/GenImage
- OpenAI guided diffusion/ADM repository: https://github.com/openai/guided-diffusion
- OpenAI GLIDE repository: https://github.com/openai/glide-text2im
- Microsoft VQ-Diffusion repository: https://github.com/microsoft/VQ-Diffusion
- Wukong-Huahua repository: https://github.com/JeffDing/WuKong-HuaHua
- Latent Diffusion / Stable Diffusion paper: https://arxiv.org/abs/2112.10752
- Midjourney documentation: https://docs.midjourney.com/

## Dataset Context

GenImage is a large-scale real/fake benchmark for AI-generated image detection.
It uses real ImageNet images as the real side and generates fake images from
ImageNet labels. The paper reports 2,681,167 images in total: 1,331,167 real
and 1,350,000 fake images.

The dataset is intentionally multi-generator. Its benchmark protocols emphasize
cross-generator classification and degraded-image classification, because
detectors often learn generator-specific artifacts and degrade when tested on
unseen or transformed images.

The local `qwen_vertex_all_val` run covered seven GenImage validation subsets:
BigGAN, VQDM, Stable Diffusion v1.5, Wukong, ADM, GLIDE, and Midjourney.

## Generator Profiles

### BigGAN

BigGAN is the only GAN-family generator in this subset. GenImage uses an
ImageNet-pretrained BigGAN model and class labels rather than text prompts. The
paper lists its generated resolution as 128 x 128.

For detection, BigGAN is special because its artifacts may differ from the
diffusion-model artifacts that dominate more recent fake-image datasets. It can
also be less semantically confusing for a forensic CNN, but a general MLLM may
miss it if the image looks like a plausible ImageNet object after resizing.

### VQDM

VQDM is VQ-Diffusion, a latent/discrete diffusion approach with a
mask-and-replace mechanism. GenImage lists its generated resolution as
256 x 256.

For detection, VQDM can carry quantization/discrete-token style artifacts that
are not the same as GAN or latent-diffusion artifacts. A text/image MLLM may not
notice these low-level signals unless the object itself looks strange.

### Stable Diffusion v1.5

Stable Diffusion is based on latent diffusion. GenImage says v1.5 uses the same
settings as v1.4 except for longer fine-tuning, and both generate 512 x 512
images in the dataset.

Stable Diffusion images are often semantically coherent and photorealistic, so
zero-shot MLLMs may over-trust surface realism. Forensic detectors usually need
to exploit subtler low-level or model-family traces.

### Wukong

Wukong is a Chinese text-to-image diffusion model. GenImage translates prompts
into Chinese for Wukong because Chinese sentences improve generation quality.
The paper lists the generated resolution as 512 x 512.

This subset is special because the prompt language and model training corpus are
different from English-centric generators. It may produce different object/style
priors, which can affect both semantic detectors and low-level detectors.

### ADM

ADM refers to the guided diffusion model from "Diffusion Models Beat GANs on
Image Synthesis." GenImage uses an ImageNet-pretrained classifier-guided model.

ADM is label-conditioned rather than normal free-form text-to-image in this
dataset. That makes it closer to ImageNet class generation. Forensics may need
to detect diffusion sampling traces, while an MLLM can easily call many ADM
images real if the object category and composition look natural.

### GLIDE

GLIDE is an OpenAI text-conditional diffusion model. GenImage uses
classifier-free guidance and lists the generated resolution as 256 x 256.

GLIDE is older/lower-resolution compared with SD and Midjourney, so some images
may have more visible synthesis limitations. But after resizing or compression,
MLLM judgments based on semantics can still miss fake labels.

### Midjourney

Midjourney is a commercial text-to-image system. GenImage uses Midjourney V5 and
lists the generated resolution as 1024 x 1024. The paper notes that Midjourney
V5 provides intricate details and images close to real photographs.

This subset is especially hard for visual plausibility checks because images can
look polished and high-detail. It is a good stress test for whether a detector
uses forensic evidence rather than just "does this look like a nice photo?"

## Implication For The Qwen Run

The local Qwen report shows balanced accuracy around 0.55 and very low recall.
That pattern is consistent with an MLLM relying heavily on visual plausibility:
when generated images are semantically coherent, Qwen often predicts `real`.

The most important next comparison is against CLIP and NPR on the same manifest.
If NPR or CLIP does much better, that supports the hypothesis that low-level or
representation-based cues are doing work that Qwen's zero-shot visual reasoning
does not capture.
