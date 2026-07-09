## mikomiko sg project — Multi-Modal GenAI Pipeline: Image, Video, & TTS

This project guide outlines how to build an integrated generative AI pipeline using **Stable Diffusion**, **Hugging Face**, and **Civitai**.

---

## 🗺️ Pipeline Architecture Overview

```
[ Hugging Face / Civitai ] 
       │ (Checkpoints, LoRAs, Audio Models)
       ▼
[ Text Prompt / Input ] ──► [ ComfyUI Node Graph ]
                               │
                               ├──► 1. Graphics (SDXL / Flux)
                               ├──► 2. Video (AnimateDiff / SVD)
                               └──► 3. Audio/TTS (Bark / XTTS)

```

---

## 🛠️ Phase 1: Asset Acquisition

To achieve high-quality results, I sourced the right foundational models and fine-tuned weights.

### 1. Civitai (Graphics & Style Fine-Tuning)

Civitai is the premier hub for community-trained Stable Diffusion models.

* **Checkpoints (`.safetensors`):** The foundational base brains (e.g., *Juggernaut XL*, *CyberRealistic*). Download these and place them in:
`└─ ComfyUI/models/checkpoints/`
* **LoRAs (Low-Rank Adaptation):** Small, specialized modifiers used to inject specific characters, art styles, or clothing. Place them in:
`└─ ComfyUI/models/loras/`

### 2. Hugging Face (Video, TTS, & Base Tensors)

Hugging Face acts as the infrastructure backbone for open-source AI weights.

* **Video Hub:** Download models like **StabilityAI's Stable Video Diffusion (SVD)** or **AnimateDiff** motion modules.
`└─ ComfyUI/models/animatediff_models/` or `/checkpoints/`
* **Audio/TTS Hub:** Sourcing model vocabularies and weights for state-of-the-art TTS frameworks like **Suno Bark** or **Coqui XTTS v2**.

---

## 🎨 Phase 2: Core Graphics & LoRA Integration

The core of image generation relies on balancing the base model's knowledge with the specific stylization of a LoRA.

### The LoRA Mathematical Concept

LoRAs freeze the original base model weights $W_0$ and inject a low-rank decomposition matrix pair ($A$ and $B$) to adapt the model without modifying the original architecture:

$$W = W_0 + \Delta W = W_0 + \alpha (B \times A)$$

Where $\alpha$ represents the **LoRA weight multiplier** (typically scaled between `0.1` and `1.0`).

### Implementation Checklist

1. **Load Checkpoint:** Select your base model (e.g., SDXL).
2. **Load LoRA:** Chain the model and CLIP outputs through a LoRA loader node.
3. **Prompting:** Use triggering keywords specified by the Civitai creator to activate the LoRA asset effectively.

---

## 🎬 Phase 3: Video Generation & Temporal Consistency

Turning static graphics into video requires introducing a temporal dimension so frames flow smoothly into one another.

### Method A: AnimateDiff (Text-to-Video / Image-to-Video)

AnimateDiff inserts a **Motion Module** into the frozen SD architecture, trained on thousands of video clips to understand physics and camera panning.

* **How it works:** It injects temporal attention layers between the spatial layers of Stable Diffusion, enforcing frame-to-frame consistency.

### Method B: Stable Video Diffusion (SVD)

SVD is a native image-to-video model that takes a latent image and passes it through an explicitly trained image-to-video conditioning pipeline.

> 💡 **Tip for Video:** Always use **ControlNet (OpenPose or Lineart)** if you need a specific motion pathway. This prevents the video from turning into an unpredictable, morphing hallucination.

---

## 🔊 Phase 4: Text-to-Speech (TTS) Integration

To complete the multi-modal workflow, generated scripts are converted to high-fidelity audio that can be combined with the video output.

```python
# Conceptual Python snippet for Hugging Face TTS pipeline
from transformers import pipeline
import scipy.io.wavfile as wavf

# Load state-of-the-art multi-lingual TTS model from HF
tts_pipeline = pipeline("text-to-speech", model="suno/bark-small")

text_script = "Welcome to the future of deterministic node-based AI generation."
audio_output = tts_pipeline(text_script)

# Save asset for video multiplexing
wavf.write("output_voice.wav", rate=audio_output["sampling_rate"], data=audio_output["audio"])

```

---

## 🎛️ Phase 5: Deterministic Automation via ComfyUI

While traditional WebUIs (like Automatic1111) rely on linear, rigid interfaces, **ComfyUI** treats generation as an explicit **Directed Acyclic Graph (DAG)**.

### Why ComfyUI is Essential for Production

* **Granular Interception:** You can explicitly route latents. For instance, you can pass an image through a KSampler, crop it, pass it to an upscale node, and throw it into an SVD video loop seamlessly.
* **Total Determinism:** By fixing the random seed node (`Control: fixed`), every single math operation inside the VAE, UNet, and Text Encoders happens identically upon execution.
* **API-First Deployment:** Every ComfyUI workflow can be saved as a `.json` API format. You can run ComfyUI headlessly on a cloud server and trigger complex image-to-video-to-audio pipelines via a standard `POST` request.

### Recommended Custom Nodes Extensions

To run this project efficiently, install the following via the **ComfyUI Manager**:

1. `ComfyUI-AnimateDiff-Evolved` (For fluid animations)
2. `ComfyUI-VideoHelperSuite` (For loading audio, stitching frames, and exporting MP4/WebM)
3. `ComfyUI-XTTS` or `ComfyUI-Bark` (For native node-based audio generation)

---
