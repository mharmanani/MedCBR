import argparse
from PIL import Image
import numpy as np
from skimage import io
import os
import pandas as pd
import matplotlib.pyplot as plt
from pydicom import dcmread
import tqdm

def crop_around_mask(image_path, mask_path, padding=20):
    """
    Reads an image and its mask, and returns a crop around the mask with optional padding.

    Args:
        image_path (str): Path to the raw image.
        mask_path (str): Path to the corresponding mask.
        padding (int): Extra pixels to include around the bounding box.

    Returns:
        cropped_image (PIL.Image): Cropped image.
        cropped_mask (PIL.Image): Cropped mask.
    """
    # Load image and mask
    image = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')  # Convert to grayscale

    mask_np = np.array(mask)
    if mask_np.max() == 0:
        raise ValueError(f"No foreground found in mask: {mask_path}")

    # Get bounding box of the nonzero mask area
    y_indices, x_indices = np.where(mask_np > 0)
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()

    # Add padding and clamp to image boundaries
    img_width, img_height = image.size
    x_min = max(0, x_min - padding)
    x_max = min(img_width, x_max + padding)
    y_min = max(0, y_min - padding)
    y_max = min(img_height, y_max + padding)

    # Crop both image and mask
    cropped_image = image.crop((x_min, y_min, x_max, y_max))
    cropped_mask = mask.crop((x_min, y_min, x_max, y_max))

    return cropped_image, cropped_mask

def crop_breast_us_images(dataset_name, src_dir, dst_dir):
    if dataset_name not in ['BrEaST', 'BUSBRA']:
        raise ValueError("Dataset name must be 'BrEaST' or 'BUSBRA'")
    
    data_dir = src_dir if src_dir else f'/h/harmanan/medcbr/data/{dataset_name}'
    cropped_dir = dst_dir if dst_dir else f'/h/harmanan/medcbr/data/{dataset_name}_cropped'
    os.makedirs(cropped_dir, exist_ok=True)

    meta = pd.read_csv(f'/h/harmanan/medcbr/data/{dataset_name}/metadata.csv')
    new_meta_name = f'/h/harmanan/medcbr/data/{dataset_name}_cropped/metadata.csv'

    for filename in os.listdir(data_dir):
        if filename.endswith('.png'):

            img_path = os.path.join(data_dir, filename)
            img = io.imread(img_path)

            # Simple cropping logic: crop 10% from each side
            h, w = img.shape
            crop_h, crop_w = int(0.1 * h), int(0.1 * w)
            cropped_img = img[crop_h:h-crop_h, crop_w:w-crop_w]

            cropped_img_path = os.path.join(cropped_dir, filename.replace('.png', '_cropped.png'))
            io.imsave(cropped_img_path, cropped_img)
    # Update metadata
    meta["img_name"] = meta["img_name"].str.replace(dataset_name, f'{dataset_name}_cropped')
    meta.to_csv(new_meta_name, index=False)


def dcm2png(df_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(df_name)

    for roi_id in tqdm.tqdm(df['ROI_ID']):
        dcmpath = df[df['ROI_ID'] == roi_id]['img_name'].values[0]
        img_array = dcmread(dcmpath).pixel_array
        plt.imshow(img_array, cmap='gray')
        plt.axis('off')
        plt.savefig(f"{save_dir}/{roi_id}.png", bbox_inches='tight', pad_inches=0)
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Preprocessing Script for Breast US Datasets')
    parser.add_argument('-d', type=str, required=True, help="Dataset name: 'BrEaST', 'BUSBRA', 'DDSM'")
    parser.add_argument('-src', type=str, help="Source directory for images")
    parser.add_argument('-dst', type=str, help="Destination directory for cropped images")
    args = parser.parse_args()

    if args.d in ['BrEaST', 'BUSBRA']:
        crop_breast_us_images(args.d, args.src, args.dst)
    elif args.d == 'DDSM':
        dcm2png(f"{args.src}/metadata.csv", args.dst)
    else:
        raise ValueError("Unsupported dataset. Choose from 'BrEaST', 'BUSBRA', 'DDSM'.")

if __name__ == '__main__':
    main()