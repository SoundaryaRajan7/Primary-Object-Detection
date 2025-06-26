import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

image_path = "C:/Users/Admin/Documents/FINAL DOCUMENTS/INTERNSHIP/INT task/car.jpg"
image_bgr = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
predictor.set_image(image_rgb)

box_coords = []

def on_select(eclick, erelease):
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)
    box_coords.append([x1, y1, x2, y2])
    plt.close()

fig, ax = plt.subplots()
ax.imshow(image_rgb)
plt.title("Draw a box around the object")
from matplotlib.widgets import RectangleSelector
rect = RectangleSelector(ax, onselect=on_select, useblit=True,
                         button=[1], 
                         minspanx=5, minspany=5, spancoords='pixels',
                         interactive=True)
plt.show()

if not box_coords:
    print(" No box drawn. Exiting.")
    exit()

input_box = np.array(box_coords[0])

masks, scores, _ = predictor.predict(
    box=input_box,
    multimask_output=True
)

best_mask = masks[np.argmax(scores)]
mask_uint8 = (best_mask * 255).astype(np.uint8)

masked_image = cv2.bitwise_and(image_bgr, image_bgr, mask=mask_uint8)
y_indices, x_indices = np.where(mask_uint8 > 0)
y1, y2 = np.min(y_indices), np.max(y_indices)
x1, x2 = np.min(x_indices), np.max(x_indices)
cropped = masked_image[y1:y2, x1:x2]

output_path = "sam_box_segmented_object.jpg"
cv2.imwrite(output_path, cropped)
print(f" Cropped object saved as: {output_path}")
