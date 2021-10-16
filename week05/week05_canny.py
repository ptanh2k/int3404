"""
Name: Pham Tuan Anh
Class: K63K2
MSSV: 18020116

You should understand the code you write.
"""

import cv2
import numpy as np
import argparse

def conv_sum(a, b):
    s = 0
    for i in range(len(a)):
        for j in range(len(a[0])):
            s += a[i][j]*b[i][j]
    return s

def my_convolution(I, g, mode='valid', boundary='zero_padding'):
    h, w = len(g), len(g[0])
    H, W = I.shape[0], I.shape[1]

    if mode == 'valid':
        output_h = H - (h - 1)
        output_w = W - (w - 1)
    else:
        output_h = H
        output_w = W

    output = [[0 for _ in range(output_w)] for _ in range(output_h)]
    for i in range(output_h):
        for j in range(output_w):
            output[i][j] = conv_sum(I[i:i + h, j:j + w], g)

    return output

# Calculate gradient magnitude and gradient direction
def gradient(img, s):
    img = np.array(img)
    
    Mx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    My = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    
    smoothed_img = cv2.GaussianBlur(img, (s, s), 0)
    img_x = my_convolution(smoothed_img, Mx)
    img_y = my_convolution(smoothed_img, My)
    
    gradient_magnitude = np.sqrt(np.square(img_x) + np.square(img_y))
    gradient_direction = np.arctan2(img_y, img_x)
    
    return (gradient_magnitude, gradient_direction)

# Non-maximum suppression
def non_maximum_suppression(gradient_magnitude, gradient_direction):
    (h, w) = gradient_magnitude.shape[:2]
    out_img = np.zeros((h, w), dtype=np.int32)
        
    theta = gradient_direction * 180 / np.pi
    theta[theta < 0] += 180
    pi = 180
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            direction = theta[i, j]
            
            r_pixel = 255
            l_pixel = 255
            
            # angle 0
            if (0 <= direction < pi / 8) or (7 * pi / 8 <= direction <= pi):
                r_pixel = gradient_magnitude[i, j+1]
                l_pixel = gradient_magnitude[i, j-1]
                    
            # angle 45
            elif (pi / 8 <= direction < 3 * pi / 8):
                r_pixel = gradient_magnitude[i+1, j-1]
                l_pixel = gradient_magnitude[i-1, j+1]
                    
            # angle 90
            elif (3 * pi / 8 <= direction < 5 * pi / 8):
                r_pixel = gradient_magnitude[i+1, j]
                l_pixel = gradient_magnitude[i-1, j]
                    
                # angle 135
            elif (5 * pi / 8 <= direction < 7 * pi / 8):
                r_pixel = gradient_magnitude[i-1, j-1]
                l_pixel = gradient_magnitude[i+1, j+1]
                
            if (gradient_magnitude[i, j] >= r_pixel) and (gradient_magnitude[i, j] >= l_pixel):
                out_img[i, j] = gradient_magnitude[i, j]
            else:
                out_img[i, j] = 0
           
    return out_img

# Hysteresis threshold:
def threshold(non_max_img, lowThresholdRatio=0.05, highThresholdRatio=0.1):
    highThreshold = non_max_img.max() * highThresholdRatio
    lowThreshold = lowThresholdRatio * highThreshold
    
    (h, v) = non_max_img.shape[:2]
    out_img = np.zeros((h, v), dtype=np.int32)
    
    weak = np.int32(25)
    strong = np.int(255)
    
    strong_i, strong_j = np.where(non_max_img >= highThreshold)
    weak_i, weak_j = np.where((lowThreshold <= non_max_img) & (non_max_img <= highThreshold))
    irrelevant_i, irrelevant_j = np.where(non_max_img < lowThreshold)
    
    out_img[strong_i, strong_j] = strong
    out_img[weak_i, weak_j] = weak
    
    return (out_img, weak, strong)

# Determine which pixels are part of real edges
def hysteresis(img, weak, strong=255):
    (h, w) = img.shape[:2]
    for i in range(1, h-1):
        for j in range(1, w-1):
            if (img[i, j] == weak):
                if (img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong) or (img[i, j-1] == strong) or (img[i, j+1] == strong) or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong):
                    img[i, j] = strong
                else:
                    img[i, j] = 0
                    
    return img
    
# Canny algorithm implementation
def canny(input_file, output_file, s):
    img = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
    
    # Apply Gaussian blur
    smoothed_img = cv2.GaussianBlur(img, (s, s), 0)
    gradient_magnitude, gradient_direction = gradient(smoothed_img, s)
    output = non_maximum_suppression(gradient_magnitude, gradient_direction)
    cv2.imshow("Output 1", np.array(output).astype(np.uint8))
    output, weak, strong = threshold(output)
    cv2.imshow("Output 2", np.array(output).astype(np.uint8))
    output = hysteresis(output, weak, strong)
    cv2.imwrite(output_file, np.array(output).astype(np.uint8))
    cv2.imshow("Output 3", np.array(output).astype(np.uint8))
    cv2.waitKey(0)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", "-i", type=str, help="Path to input image")
    parser.add_argument("--output_file", "-o", type=str, help="Path to output image")
    parser.add_argument("--size", "-s", type=int, default=3, help="Size of gaussian filter")

    args = parser.parse_args()
    canny(args.input_file, args.output_file, args.size)