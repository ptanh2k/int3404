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

def gradient_sobel(input_file, output_file, s):
    Mx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    My = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    
    img = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
    smoothed_img = cv2.GaussianBlur(img, (s, s), 0)
    img_x = my_convolution(smoothed_img, Mx)
    # cv2.imshow('Horizontal image', np.array(img_x).astype(np.uint8))
    
    img_y = my_convolution(smoothed_img, My)
    # cv2.imshow('Vertical image', np.array(img_y).astype(np.uint8))
    
    gradient = np.sqrt(np.square(img_x) + np.square(img_y))
    cv2.imwrite(output_file, np.array(gradient).astype(np.uint8))
    cv2.imshow('Gradient', np.array(gradient).astype(np.uint8))
    cv2.waitKey(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", "-i", type=str, help="Path to input image")
    parser.add_argument("--output_file", "-o", type=str, help="Path to output image")
    parser.add_argument("--size", "-s", type=int, default=3, help="Size of gaussian filter")

    args = parser.parse_args()
    
    gradient_sobel(args.input_file, args.output_file, args.size)
