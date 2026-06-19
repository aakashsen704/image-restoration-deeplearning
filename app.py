import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import gradio as gr
import os
from unet import get_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = None

def load_model():
    global MODEL
    MODEL = get_model(DEVICE)
    ckpt = torch.load("checkpoints/best_model.pth", map_location=DEVICE)
    MODEL.load_state_dict(ckpt["model_state_dict"])
    MODEL.eval()
    print(f"✅ Model loaded! PSNR: {ckpt['psnr']:.2f}dB")

def restore_image(input_image):
    if input_image is None or MODEL is None:
        return None
    img = Image.fromarray(input_image.astype(np.uint8)).convert("RGB")
    original_size = img.size
    w, h = original_size
    new_w = max((w // 32) * 32, 32)
    new_h = max((h // 32) * 32, 32)
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)
    tensor = transforms.ToTensor()(img_resized).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = torch.clamp(MODEL(tensor), 0, 1)
    output_np = (output.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
    output_np = np.transpose(output_np, (1, 2, 0))
    result = Image.fromarray(output_np).resize(original_size, Image.LANCZOS)
    return np.array(result)

with gr.Blocks(title="Image Restoration AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🔭 Image Restoration AI
    ### Remove blur and noise from images using Deep Learning
    *Custom U-Net with Channel Attention | Trained on GOPRO Large Dataset | PSNR: 25.29dB*
    """)
    with gr.Row():
        input_img  = gr.Image(label="📸 Input (Blurry/Noisy)")
        output_img = gr.Image(label="🌟 Restored Output")
    btn = gr.Button("✨ Restore Image", variant="primary", size="lg")
    btn.click(fn=restore_image, inputs=input_img, outputs=output_img)
    gr.Markdown("""
    ### 📊 Model Details
    - **Architecture:** Custom U-Net with Channel Attention + Residual Blocks
    - **Parameters:** 69 Million
    - **Dataset:** GOPRO Large (2103 train / 1111 val pairs)
    - **Training:** 100 epochs on Tesla T4 GPU
    - **Best PSNR:** 25.29dB | **Best SSIM:** 0.7860
    """)

if __name__ == "__main__":
    load_model()
    demo.launch()
