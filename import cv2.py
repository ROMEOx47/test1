import cv2
import numpy as np
import pytesseract

# Set the path to the Tesseract OCR executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the image
img = cv2.imread('C:\\Users\\Administrator\\Desktop\\Project CM\\car.jpg', cv2.IMREAD_COLOR)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Edge detection
edges = cv2.Canny(blur, 50, 150)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours by area (Assuming the license plate will be one of the largest contours)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

# Iterate over the contours to find the license plate
for contour in contours:
    # Get the rectangle that contains the contour
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # Crop the image
    crop_img = img[min(box[:, 1]):max(box[:, 1]), min(box[:, 0]):max(box[:, 0])]

    # Convert the cropped image to grayscale
    gray_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    # Use pytesseract to extract text
    text = pytesseract.image_to_string(gray_crop)
    print("Detected license plate Number is:", text)
