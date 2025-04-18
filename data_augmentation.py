import os
import cv2
import numpy as np
from imgaug import augmenters as iaa

# Paths
CARD_DIR = "cards"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Augmentation pipeline
augment_pipeline = iaa.Sequential([
    iaa.Affine(
        rotate=(-25, 25),
        scale=(0.8, 1.2),
        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
        shear=(-8, 8)
    ),
    iaa.OneOf([
        iaa.AdditiveGaussianNoise(scale=(0, 10)),
        iaa.Multiply((0.8, 1.2)),  # Brightness
        iaa.GammaContrast((0.8, 1.2)),
        iaa.GaussianBlur(sigma=(0.0, 1.0)),
    ]),
    iaa.PerspectiveTransform(scale=(0.01, 0.08))
])

def augment_card(card_path, output_path, num_augments=5):
    card_img = cv2.imread(card_path)
    card_name = os.path.splitext(os.path.basename(card_path))[0]

    if card_img is None:
        print(f"Failed to read {card_path}")
        return

    for i in range(num_augments):
        augmented = augment_pipeline(image=card_img)
        out_file = os.path.join(output_path, f"{card_name}_aug{i}.jpg")
        cv2.imwrite(out_file, augmented)
        print(f"Saved: {out_file}")

# Run augmentation
for card_file in os.listdir(CARD_DIR):
    if card_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        augment_card(os.path.join(CARD_DIR, card_file), OUTPUT_DIR)
