# Stable-Diffusion-Image-Generation


This repository contains code and instructions for generating images from text prompts using the Stable Diffusion model from the Hugging Face Diffusers library.

## Code Functionality

The provided Python code demonstrates how to generate images using the Stable Diffusion model. Here is a detailed breakdown of its functionality:

### Installation

The following command installs the required libraries:
- `diffusers`: The core library for loading and running diffusion models like Stable Diffusion.
- `transformers`: Provides pre-trained language models often used in text-to-image tasks.
- `torch`: A popular deep learning framework commonly used with Diffusers.
- `pillow`: An image processing library for loading and displaying images.

```bash
!pip install diffusers transformers torch pillow
```

### Imports

The necessary libraries are imported:
- `StableDiffusionPipeline` from `diffusers` for generating images using the Stable Diffusion model.
- `torch` for deep learning operations.
- `Image` from `PIL` for image handling.
- `matplotlib.pyplot` for plotting and visualization.

### Model Selection

Set the ID of the Stable Diffusion model to be loaded from the Hugging Face Hub.

```python
model_id = "CompVis/stable-diffusion-v1-4"
```

### Device Configuration

Check if a GPU (CUDA) is available and set the device accordingly. This ensures faster processing if a GPU is present.

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
```

### Pipeline Loading

Load the Stable Diffusion pipeline from the specified model ID and move it to the selected device (GPU or CPU).

```python
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe.to(device)
```

### User Input

Prompt the user to enter a text description of the desired image. The quality of the generated image depends on the clarity and creativity of the prompt.

```python
prompt = input("Enter a text prompt to generate an image: ")
```

### Image Generation

Generate an image using the Stable Diffusion pipeline based on the provided text prompt.

```python
with torch.autocast("cuda"):
    image = pipe(prompt).images[0]
```

### Image Display

Display the generated image using Matplotlib.

```python
plt.imshow(image)
plt.axis('off')
plt.show()
```
