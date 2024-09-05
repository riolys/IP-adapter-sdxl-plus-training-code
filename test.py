import os
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
from ip_adapter import IPAdapterPlusXL

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def save_prompt_and_image(prompt, images, folder_name, output_filename):
    os.makedirs(folder_name, exist_ok=True)
    prompt_file = os.path.join(folder_name, "prompt.txt")
    with open(prompt_file, "w") as f:
        f.write(prompt)
    output_path = os.path.join(folder_name, output_filename)
    images.save(output_path, format="webp")
    print(f"Saved to {output_path}")

def load_prompts(file_path):
    with open(file_path, "r") as f:
        prompts = [line.strip() for line in f.readlines()]
    return prompts

def load_image_paths(image_file_path):
    with open(image_file_path, "r") as f:
        image_paths = [line.strip() for line in f.readlines()]
    return image_paths

def main(test_prompt_file, test_image_file, checkpoint_dir):
    base_model_path = "models/stable-diffusion-xl-base-1.0"
    image_encoder_path = "models/IP-Adapter/models/image_encoder"
    device = "cuda"
    num_samples = 2

    # Load Stable Diffusion XL pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        add_watermarker=False,
    )

    # Load all checkpoints from the specified directory
    checkpoints = [
        os.path.join(checkpoint_dir, d, "ip_adapter.bin") 
        for d in os.listdir(checkpoint_dir) 
        if os.path.isdir(os.path.join(checkpoint_dir, d))
    ]

    # Load all prompts and images
    prompts = load_prompts(test_prompt_file)
    image_paths = load_image_paths(test_image_file)

    for checkpoint in checkpoints:
        checkpoint_name = os.path.basename(os.path.dirname(checkpoint))
        
        # Load IP-Adapter for the current checkpoint
        ip_model = IPAdapterPlusXL(pipe, image_encoder_path, checkpoint, device, num_tokens=16)
        
        for image_path in image_paths:
            try:
                image = Image.open(image_path)
                image = image.resize((512, 512))
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                continue

            for prompt in prompts:
                # Generate images with the current prompt
                images = ip_model.generate(
                    pil_image=image, 
                    num_samples=num_samples, 
                    num_inference_steps=30, 
                    seed=42,
                    prompt=prompt, 
                    scale=0.5
                )
                grid = image_grid(images, 1, num_samples)

                # Save result for this checkpoint, prompt, and image
                image_name = os.path.splitext(os.path.basename(image_path))[0]
                folder_name = f"test/results/{checkpoint_name}/{image_name}/{prompt.replace(' ', '_')}"
                save_prompt_and_image(prompt, grid, folder_name, f"{checkpoint_name}.webp")

if __name__ == "__main__":
    # Example of how to pass the test prompt file, test image file, and checkpoint directory
    test_prompt_file = "test/test_prompt.txt"  # Path to the prompt file
    test_image_file = "test/test_image.txt"    # Path to the image file
    checkpoint_dir = "P-Adapter/sd-ip_adapter"  # Path to the checkpoint directory
    main(test_prompt_file, test_image_file, checkpoint_dir)
