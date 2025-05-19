import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_stats(img):
    """
    Compute mean and std of a grayscale image.
    Assumes img is in [0, 255] uint8 format.
    """
    img_float = img.astype(np.float32) / 255.0
    mean = np.mean(img_float)
    std = np.std(img_float)
    return mean, std

def normalize_image(img, mean, std):
    """
    Normalize image to [-1, 1]
    """
    img_float = img.astype(np.float32) / 255.0
    img_normalized = (img_float - mean) / (std + 1e-8)
    return img_normalized

def plot_comparison(original, norm_01, norm_fixed, norm_real, title):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(title, fontsize=16)

    # Original image
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')

    # Histograms
    axes[1, 0].hist(original.ravel(), bins=256, range=[0,255], color='black', alpha=0.7)
    axes[1, 0].set_title("Original Histogram")

    # [0,1] normalized
    axes[0, 1].imshow(norm_01, cmap='gray')
    axes[0, 1].set_title("[0,1] Normalized")
    axes[0, 1].axis('off')

    axes[1, 1].hist(norm_01.ravel(), bins=50, color='blue', alpha=0.7)
    axes[1, 1].set_title("[0,1] Histogram")

    # [-1,1] with hardcoded stats
    axes[0, 2].imshow(norm_fixed, cmap='gray')
    axes[0, 2].set_title("Fixed Norm\n(mean=0.5, std=0.5)")
    axes[0, 2].axis('off')

    axes[1, 2].hist(norm_fixed.ravel(), bins=50, color='green', alpha=0.7)
    axes[1, 2].set_title("Fixed Histogram")

    # [-1,1] with real stats
    axes[0, 3].imshow(norm_real, cmap='gray')
    axes[0, 3].set_title("Real Norm\n(mean/std from image)")
    axes[0, 3].axis('off')

    axes[1, 3].hist(norm_real.ravel(), bins=50, color='purple', alpha=0.7)
    axes[1, 3].set_title("Real Histogram")

    plt.tight_layout()
    plt.show()

def main(image_path):
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    print("Image loaded successfully.")

    # [0,1] normalization
    norm_01 = img.astype(np.float32) / 255.0

    # Fixed normalization [-1,1]
    fixed_mean, fixed_std = 0.5, 0.5
    norm_fixed = (norm_01 - fixed_mean) / fixed_std

    # Real stats normalization
    real_mean, real_std = compute_stats(img)
    norm_real = normalize_image(img, real_mean, real_std)

    print(f"Computed Mean: {real_mean:.4f}")
    print(f"Computed Std:  {real_std:.4f}")

    # Plot comparison
    plot_comparison(img, norm_01, norm_fixed, norm_real, image_path)

if __name__ == "__main__":

    image_path = r"C:\Users\kevin\Documents\GitHub\YOLO11-Enfartment-PoopPS\data_getting\screenshots\frame1.png"
    main(image_path)