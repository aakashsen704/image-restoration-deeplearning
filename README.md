# Image Restoration AI - Deblurring and Denoising

A deep learning project that restores blurry and noisy images using a custom U-Net architecture.

## Results

| Metric  | Value     |
|---------|-----------|
| PSNR    | 25.29 dB  |
| SSIM    | 0.7860    |
| Dataset | GOPRO Large |
| Epochs  | 100       |
| GPU     | Tesla T4  |

## Architecture

- Model: Custom U-Net with Channel Attention and Residual Blocks
- Parameters: 69 Million
- Input: Blurry/Noisy RGB image
- Output: Restored sharp RGB image
- Loss: L1 Loss with Cosine Annealing LR scheduler

## Project Structure

- unet.py - Model architecture
- dataset.py - Data loader with augmentation
- metrics.py - PSNR and SSIM metrics
- train.py - Training script
- app.py - Gradio demo UI
- requirements.txt - Dependencies

## Getting Started

1. Clone the repo
git clone https://github.com/aakashsen704/image-restoration-deeplearning.git

2. Install dependencies
pip install -r requirements.txt

3. Download GOPRO Dataset
wget https://huggingface.co/datasets/snah/GOPRO_Large/resolve/main/GOPRO_Large.zip
unzip GOPRO_Large.zip

4. Train the model
python train.py

5. Run the demo
python app.py

## Training Progress

| Epoch | PSNR     | SSIM   |
|-------|----------|--------|
| 1     | 21.20 dB | 0.6992 |
| 20    | 24.66 dB | 0.7716 |
| 50    | 25.06 dB | 0.7804 |
| 99    | 25.29 dB | 0.7860 |

## Tech Stack

- PyTorch
- OpenCV and PIL
- Gradio
- Google Colab Tesla T4 GPU
- GOPRO Large Dataset

## References

- MPRNet: https://github.com/swz30/MPRNet
- GOPRO Dataset: https://seungjunnah.github.io/Datasets/gopro.html
- U-Net Paper: https://arxiv.org/abs/1505.04597

## Author

Aakash - Deep Learning enthusiast building computer vision projects.
