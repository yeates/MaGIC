import os
import cv2
import torch
import numpy as np
import gradio as gr
from PIL import Image
from einops import repeat


def make_batch_sd(
        image,
        mask,
        txt,
        device,
        num_samples=1):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    batch = {
        "image": repeat(image.to(device=device), "1 ... -> n ...", n=num_samples),
        "txt": num_samples * [txt],
        "mask": repeat(mask.to(device=device), "1 ... -> n ...", n=num_samples),
        "masked_image": repeat(masked_image.to(device=device), "1 ... -> n ...", n=num_samples),
    }
    return batch


def save(input_image):
    out_dir = 'scripts/pair/'
    os.makedirs(out_dir, exist_ok=True)
    init_image = input_image["image"].convert("RGB")
    init_mask = input_image["mask"].convert("RGB")
    base_count = int(sorted(os.listdir(out_dir))[-1].split('_')[0]) + 1 if len(os.listdir(out_dir)) != 0 else 0
    init_image.save(f'{out_dir}/{base_count}.jpg')
    init_mask.save(f'{out_dir}/{base_count}_mask.jpg')
    
    masked = np.array(init_image) * ((255 - np.array(init_mask)) /255.)
    masked = cv2.cvtColor(masked.astype(np.uint8), cv2.COLOR_RGB2BGR)

    return [Image.fromarray(masked[:, :, ::-1].astype(np.uint8))]


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Draw Mask")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', tool='sketch', type="pil")
            run_button = gr.Button(value="Save")
        with gr.Column():
            gallery = gr.Gallery(label="Masked images", show_label=False).style(
                grid=[1], height="auto")

    run_button.click(fn=save, inputs=[input_image], outputs=[gallery])


block.launch()
