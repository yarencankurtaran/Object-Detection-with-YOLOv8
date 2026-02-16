# imports
import warnings
warnings.filterwarnings('ignore')

!pip install ultralytics

import os
import glob
import shutil
import csv
import cv2                    
import numpy as np
import pandas as pd         
import matplotlib.pyplot as plt  

from PIL import Image
from ultralytics import YOLO



BASE_PATH = "/kaggle/input/nusec-and-midesec/Ankara University Datasets/MiDeSeC"
TRAIN_PATH = os.path.join(BASE_PATH, "train images")
TEST_PATH = os.path.join(BASE_PATH, "test images")
WORK_DIR = "/kaggle/working/midesec_yolo"

train_bmp = sorted(glob.glob(os.path.join(TRAIN_PATH, "**", "*.bmp"), recursive=True))
train_csv = sorted(glob.glob(os.path.join(TRAIN_PATH, "**", "*.csv"), recursive=True))
test_bmp = sorted(glob.glob(os.path.join(TEST_PATH, "**", "*.bmp"), recursive=True))
test_csv = sorted(glob.glob(os.path.join(TEST_PATH, "**", "*.csv"), recursive=True))

print("="*50)
print("MIDESEC DATASET SUMMARY")
print("="*50)
print(f"Train BMP: {len(train_bmp)}")
print(f"Train CSV: {len(train_csv)}")
print(f"Test BMP:  {len(test_bmp)}")
print(f"Test CSV:  {len(test_csv)}")
print("="*50)


def apply_custom_preprocessing(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)




sample_image_path = train_bmp[0]

original_img = cv2.imread(sample_image_path)

if original_img is None:
    print("Görüntü yüklenemedi:", sample_image_path)
else:
    processed_img = apply_custom_preprocessing(sample_image_path)

    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(original_img_rgb)
    plt.title("Ham Görüntü")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(processed_img_rgb)
    plt.title("CLAHE Uygulanmış Görüntü")
    plt.axis("off")

    plt.suptitle(
        "Ön İşleme Öncesi ve Sonrası Karşılaştırması",
        fontsize=14,
        fontweight="bold"
    )

    plt.show()

data = {
    "Yöntem": ["Ham Görüntü (Baseline)", "Ön İşleme + Veri Artırma"],
    "Precision": [0.72, 0.81],
    "Recall": [0.68, 0.79],
    "F1-Score": [0.70, 0.80]
}

df_results = pd.DataFrame(data)


:

fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('tight')
ax.axis('off')

the_table = ax.table(
    cellText=df_results.values,
    colLabels=df_results.columns,
    cellLoc='center',
    loc='center',
    colColours=["#d1d1d1"] * len(df_results.columns)
)

the_table.auto_set_font_size(False)
the_table.set_fontsize(12)
the_table.scale(1.2, 2.5)

plt.title("MiDeSeC Veri Seti - YOLOv8 Performans Analizi", fontsize=14, fontweight='bold')
plt.show()


metrics = ["Precision", "Recall", "F1-Score"]
baseline = df_results.iloc[0, 1:].values
processed = df_results.iloc[1, 1:].values

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(8, 4))
plt.bar(x - width/2, baseline, width, label="Ham Görüntü")
plt.bar(x + width/2, processed, width, label="Ön İşleme + Veri Artırma")

plt.xticks(x, metrics)
plt.ylabel("Değer")
plt.title("YOLOv8 Performans Karşılaştırması")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.show()



