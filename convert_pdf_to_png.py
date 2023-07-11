import cv2
import numpy as np
import sys
from pdf2image import convert_from_path
from PIL import Image

def process_pdf(pdf_path):
    # Convert PDF to images with higher DPI
    # Increase the DPI if the image quality is too low
    images = convert_from_path(pdf_path, dpi=300)

    for i, image in enumerate(images):
        # Convert PIL Image to OpenCV Image (numpy array)
        image = np.array(image)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Increase brightness and decrease contrast
        # Adjust alpha (1.2) to control contrast (lower value increases contrast)
        # Adjust beta (50) to control brightness (higher value increases brightness)
        img_enhanced = cv2.convertScaleAbs(gray, alpha=1.2, beta=50)

        # Binarization
        # Adjust the second parameter (0) to control the threshold for binarization
        _, binary = cv2.threshold(img_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Noise removal
        # Adjust the kernel size (1, 1) to control the amount of noise removal
        # Larger kernel size removes more noise but can also result in loss of detail
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        img_denoised = cv2.medianBlur(closing, 3)

        # Scaling with a smaller factor
        # Adjust fx and fy (1.1) to control the scaling factor
        # Higher values increase the size of the image
        img_scaled = cv2.resize(img_denoised, None, fx=1.1, fy=1.1, interpolation=cv2.INTER_CUBIC)

        # Convert back to PIL Image for saving
        img_ready = Image.fromarray(img_scaled)

        # Save the image
        base_name = pdf_path.rsplit('.', 1)[0]
        img_ready.save(f"{base_name}_processed_{i}.png")


# In this code:

# - The DPI in `convert_from_path` is set to 600. If the image quality is too low, you can try increasing this value.
# - The `alpha` and `beta` parameters in `cv2.convertScaleAbs` are set to 1.2 and 50, respectively. If the image is too dark or too light, you can adjust `beta`. If the image has too much or too little contrast, you can adjust `alpha`.
# - The threshold in `cv2.threshold` is set to 0, which means that the threshold value is automatically determined using the Otsu's binarization method. If the binarization result is not satisfactory, you can try adjusting this value or use adaptive thresholding with `cv2.adaptiveThreshold`.
# - The kernel size in the noise removal steps is set to (1, 1). If the image has a lot of noise, you can try increasing these values. If the image is losing too much detail, you can try decreasing these values.
# - The scaling factor in `cv2.resize` is set to 1.1. If the text in the image is too small, you can try increasing these values. If the image is becoming too distorted or if the processing time is too long, you can try decreasing these values.

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <pdf_path>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    process_pdf(pdf_path)