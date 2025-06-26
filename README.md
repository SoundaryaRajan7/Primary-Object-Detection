# Segment Anything Box Prompt - Object Cropping

This project uses Meta AI's Segment Anything Model (SAM) to segment and crop a primary object using a **box prompt**.

## 🖼️ What it does

- Loads an image
- Lets the user draw a bounding box on it
- SAM predicts a mask for the object in that box
- The cropped object is saved automatically

---

## 📦 Requirements

- Python 3.7+
- OpenCV
- PyTorch
- NumPy
- Matplotlib
- Segment Anything

### Install dependencies:
```bash
pip install opencv-python matplotlib numpy torch torchvision
pip install git+https://github.com/facebookresearch/segment-anything.git

📥 Download SAM Model Weights
Download sam_vit_b_01ec64.pth from:
https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

Place it in the project folder.
