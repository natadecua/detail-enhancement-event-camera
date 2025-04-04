# DSIGPRO EQ1 GROUP 3 - FINAL PROJECT
# BOQUER, CUA, OCAMPO, SORIANO, TABIOLO
# SUBMITTED TO: EDWIN SYBINGCO

import cv2
import numpy as np
from matplotlib import pyplot as plt
import lpips
import torch
from torchvision import transforms
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error


# Functions created for organization and easy modifying for other methods
# Functions to enhance contrast
def enhance_contrast_hsv(image):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Extract the V (Value) channel for contrast enhancement <--------------------- Based from RRL
    h, s, v = cv2.split(hsv_image)
    
    # Apply Histogram Equalization on the V channel (enhance contrast)
    v_eq = cv2.equalizeHist(v)
    
    # Merge the enhanced V channel back with the original H and S channels
    hsv_enhanced = cv2.merge([h, s, v_eq])
    
    return hsv_enhanced

def enhance_contrast_grayscale(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Histogram Equalization to enhance contrast
    gray2enhance = cv2.equalizeHist(gray)
    
    return gray2enhance



# Function to reduce noise using Bilateral Filtering  <----------------------- Bilateral Filter for smoother noise reduction while preserving edges
def reduce_noise(image):
    denoised_image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    return denoised_image

# Function to apply Gaussian Thresholding
def gaussian_thresholding(image, max_value=255):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Thresholding (adaptive thresholding using Gaussian filter)
    thresholded = cv2.adaptiveThreshold(gray, max_value, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    return thresholded

def calculate_loe(original, processed):
    # Convert images to grayscale if they aren't already
    if len(original.shape) == 3:
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        processed_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    else:
        original_gray = original
        processed_gray = processed

    # Calculate gradients
    orig_gx = cv2.Sobel(original_gray, cv2.CV_64F, 1, 0, ksize=3)
    orig_gy = cv2.Sobel(original_gray, cv2.CV_64F, 0, 1, ksize=3)
    proc_gx = cv2.Sobel(processed_gray, cv2.CV_64F, 1, 0, ksize=3)
    proc_gy = cv2.Sobel(processed_gray, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate gradient magnitudes
    orig_mag = np.sqrt(orig_gx**2 + orig_gy**2)
    proc_mag = np.sqrt(proc_gx**2 + proc_gy**2)

    # Calculate gradient directions
    orig_dir = np.arctan2(orig_gy, orig_gx)
    proc_dir = np.arctan2(proc_gy, proc_gx)

    # Calculate LOE
    direction_diff = np.abs(orig_dir - proc_dir)
    magnitude_diff = np.abs(orig_mag - proc_mag)
    
    # Combine direction and magnitude differences
    loe = np.mean(direction_diff * magnitude_diff)
    
    return loe

def calculate_loe_color(reference, processed):
    # Convert both images to LAB color space
    reference_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)
    processed_lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)

    # Split LAB images into L, A, B channels
    ref_l, ref_a, ref_b = cv2.split(reference_lab)
    proc_l, proc_a, proc_b = cv2.split(processed_lab)

    # Define a helper function to compute LOE for a given channel (L, A, or B)
    def channel_loe(ref_channel, proc_channel):
        # Sobel gradients for each channel
        gx_ref = cv2.Sobel(ref_channel, cv2.CV_64F, 1, 0, ksize=3)
        gy_ref = cv2.Sobel(ref_channel, cv2.CV_64F, 0, 1, ksize=3)
        gx_proc = cv2.Sobel(proc_channel, cv2.CV_64F, 1, 0, ksize=3)
        gy_proc = cv2.Sobel(proc_channel, cv2.CV_64F, 0, 1, ksize=3)

        # Calculate magnitude and direction of gradients
        mag_ref = np.sqrt(gx_ref**2 + gy_ref**2)
        mag_proc = np.sqrt(gx_proc**2 + gy_proc**2)
        dir_ref = np.arctan2(gy_ref, gx_ref)
        dir_proc = np.arctan2(gy_proc, gx_proc)

        # Calculate the magnitude and direction differences
        direction_diff = np.abs(dir_ref - dir_proc)
        magnitude_diff = np.abs(mag_ref - mag_proc)

        # Combine magnitude and direction differences for LOE of this channel
        return np.mean(direction_diff * magnitude_diff)

    # Calculate LOE for each channel (L, A, B)
    loe_l = channel_loe(ref_l, proc_l)
    loe_a = channel_loe(ref_a, proc_a)
    loe_b = channel_loe(ref_b, proc_b)

    # Combine the LOE values from all channels (you can weight them as needed)
    total_loe = (loe_l + loe_a + loe_b) / 3

    return total_loe


def calculate_lpips(original, processed):
    # Load pre-trained LPIPS model
    loss_fn = lpips.LPIPS(net='vgg')  # You can use other models such as alex or squeezenet

    # Convert images to tensors
    transform = transforms.Compose([transforms.ToTensor()])
    original_tensor = transform(original).unsqueeze(0)
    processed_tensor = transform(processed).unsqueeze(0)

    # Compute LPIPS score
    return loss_fn(original_tensor, processed_tensor).item()

def calculate_ssim(original, processed):
    return ssim(original, processed, multichannel=True, win_size=3)

def calculate_mse(original, processed):
    # Normalize images to 0-1 range
    original_norm = original.astype(float) / 255
    processed_norm = processed.astype(float) / 255
    
    # Calculate MSE
    mse = mean_squared_error(original_norm.flatten(), processed_norm.flatten())
    
    # Optional: Convert to PSNR (Peak Signal-to-Noise Ratio)
    psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    
    return mse, psnr

# Main function to process the image
def process_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # For evaluation, use an original high-quality image for comparison
    original_image = cv2.imread('Test.png')  # Replace with the reference image path

    # Step 1: Enhance Contrast in HSV and Grayscale
    hsv_enhanced = enhance_contrast_hsv(image)
    gray_enhanced = enhance_contrast_grayscale(image)
    
    # Step 2: Apply Noise Reduction
    enhanced_image = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
    denoised_image = reduce_noise(enhanced_image)
    
    # Reduce the noise of the grayscale image
    gray2enhance = reduce_noise(gray_enhanced)
    
    # Step 3: Apply Gaussian Thresholding to the denoised image
    thresholded_image = gaussian_thresholding(denoised_image)
    
    # Calculate Evaluation Metrics
    mse_value, psnr = calculate_mse(original_image, denoised_image)
    ssim_value = calculate_ssim(original_image, denoised_image)
    #lpips_value = calculate_lpips(original_image, denoised_image)
    loe_value = calculate_loe(original_image, denoised_image)
    loe_color = calculate_loe_color(original_image, denoised_image)
    
    # Print the results
    #print("Evaluation Metrics:")
    print(f"MSE: {mse_value}")
    print(f"PSNR: {psnr}")
    print(f"SSIM: {ssim_value}")
    #print(f"LPIPS: {lpips_value}")
    print(f"LOE: {loe_value}")
    print(f"LOE COLOR: {loe_color}")


    # Show results
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 4, 1)
    plt.title("Original Image", fontsize = 8)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.title("Gaussian Thresholded Image", fontsize = 8)
    plt.imshow(thresholded_image, cmap='gray')
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.title("Denoised Enhanced Grayscale Image", fontsize = 8)
    plt.imshow(gray2enhance, cmap='gray')
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.title("Denoised Enhanced Color Restored Image", fontsize = 8)
    plt.imshow(cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.show()

    return thresholded_image

# Call the function
image_path = 'Test.png'  # Replace with your image file
processed_image = process_image(image_path)