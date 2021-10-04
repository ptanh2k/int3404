"""
Name: Pham Tuan Anh
Class: K63K2
MSSV: 18020116

You should understand the code you write.
"""

import numpy as np
import cv2
import argparse
from matplotlib import pyplot as plt


def q_0(input_file, output_file, ):
    img = cv2.imread(input_file, cv2.IMREAD_COLOR)
    cv2.imshow('Test img', img)
    cv2.waitKey(5000)

    cv2.imwrite(output_file, img)


def q_1(input_file, output_file):
    """
    Convert the image to gray channel of the input image.
    """
    img = cv2.imread(input_file, cv2.IMREAD_COLOR)
    cv2.imshow('Color', img)
    
    R, G, B = img[:, :, 2], img[:, :, 1], img[:, :, 0]

    # Convert image to gray channgel
    gray = 0.299 * R + 0.587 * G + 0.114 * B
    img_gray = gray.astype(np.uint8)
    cv2.imwrite(output_file, img_gray)
    cv2.imshow('Gray', img_gray)
    cv2.waitKey(0)
    
# Normalized histogram
def normallizedHistogram(img):
    (height, width) = img.shape[:2]
    # uint64 works while uint8 doesn't???
    # h = np.zeros((256, ), np.uint8)       //Wrong?
    # h= np.zeros((256,), dtype=int)        //Right??
    h = [0] * 256
    for i in range(height):
        for j in range(width):
            h[img[i, j]] += 1
    return np.array(h) / (height * width)

    
# Finds cumulative sum of a numpy array, list        
def cummulativeSum(normalized_hist):
    cummulative_sum = np.zeros_like(normalized_hist, np.float64)
    hist_length = len(normalized_hist)
    for i in range(hist_length):
        cummulative_sum[i] = sum(normalized_hist[:i+1])
    return cummulative_sum

def q_2(input_file, output_file):
    """
    Performs a histogram equalization on the input image.
    """
    img = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
    (height, width) = img.shape[:2]
    
    # Analysing original image and original histogram
    # original_hist = cv2.calcHist([img], [0], None, [256], [0, 256])       # Mask: None, value from 0 - 255
    # plt.figure()
    # plt.axis("off")
    # plt.imshow(img, cmap='gray')
    # plt.figure()
    # plt.title('Histogram')
    # plt.xlabel('Bins')
    # plt.ylabel('Number of pixel')
    # plt.plot(original_hist)
    # plt.xlim([0, 256])
    # plt.show()
    
    # Histogram equalization
    norm_hist = normallizedHistogram(img)
    
    cumulative_sum = cummulativeSum(norm_hist)
    new_hist = np.array(np.rint(255 * cumulative_sum))
    
    # Convert image 
    img_eq = np.zeros_like(img)
    
    for i in range(height):
        for j in range(width):
            img_eq[i, j] = new_hist[img[i, j]]
            
    # Check        
    hist_test = cv2.calcHist([img_eq], [0], None, [256], [0, 256])         # Mask: None, value from 0 - 255
    plt.figure()
    plt.axis("off")
    plt.imshow(img_eq, cmap='gray')
    plt.figure()
    plt.title('Histogram')
    plt.xlabel('Bins')
    plt.ylabel('Number of pixel')
    plt.plot(hist_test)
    plt.xlim([0, 256])
    plt.show()

    cv2.imwrite(output_file, img_eq)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", "-i", type=str, help="Path to input image")
    parser.add_argument("--output_file", "-o", type=str, help="Path to output image")
    parser.add_argument("--question", "-q", type=int, default=0, help="Question number")

    args = parser.parse_args()

    q_number = args.question

    if q_number == 1:
        q_1(input_file=args.input_file, output_file=args.output_file)
    elif q_number == 2:
        q_2(input_file=args.input_file, output_file=args.output_file)
    else:
        q_0(input_file=args.input_file, output_file=args.output_file)
