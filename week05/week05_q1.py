"""
Name: Pham Tuan Anh
Class: K63K2
MSSV: 18020116

You should understand the code you write.
"""

import cv2
import numpy as np
import argparse
import time

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


def init_kernel(sz=3):
    s = sz*sz
    g = [[1.0/s for i in range(sz)] for i in range(sz)]

    return g

# Mean filter
def mean_filter(input_file, output_file, kernel_size):
    # Read input file with gray value
    img = cv2.imread(input_file, 0)
    # print(np.array(img))
    g = init_kernel(kernel_size)

    # Calculate times needed to complete the process
    start_time = time.time()
    output_img = my_convolution(img, g)
    run_time = time.time() - start_time

    # for input/output
    cv2.imwrite(output_file, np.array(output_img))
    print("Run convolution in: %.2f s" % run_time)
    # print(np.array(output_img))
    cv2.imshow("Output", np.array(output_img))
    cv2.waitKey(0)
    
def cal_median(img):
    flatten_img = img.flatten()
    flatten_img = np.sort(flatten_img)
    l = len(flatten_img)
    if l < 0:
        return None
    elif l % 2 == 0:
        return (flatten_img[(l - 1) / 2] + flatten_img[(l + 1) / 2]) / 2
    else:
        return flatten_img[(l - 1) /2] 
    
# Median filter
def median_filter(input_file, output_file, kernel_size):
    #Read input file with gray value
    img = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
    (h, w) = img.shape[:2]
    print(h, w)
    output_img = np.empty(shape=(h, w))
    output_img.fill(0)
    print(output_img[0])
    window = np.zeros(kernel_size * kernel_size)
    edge = kernel_size // 2
    
    for x in range(edge, w - edge + 1):
        for y in range(edge, h - edge + 1):
            i = 0
            for fx in range(kernel_size):
                for fy in range(kernel_size):
                    window[i] = img[x + fx - edge, y + fy - edge]
                    i = i + 1
            np.sort(window)
            output_img[x, y] = window[kernel_size * kernel_size // 2]
                    
    cv2.imwrite(output_file, np.array(output_img))
    cv2.imshow("Output", np.array(output_img))
    cv2.waitKey(0)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", "-i", type=str, help="Path to input image")
    parser.add_argument("--output_file", "-o", type=str, help="Path to output image")
    parser.add_argument("--filter_type", "-t", type=str, default='mean', help="One of mean/median/sharpness")
    parser.add_argument("--size", "-s", type=int, default=3, help="Size of filter")
    parser.add_argument("--alpha", "-a", type=float, default=0.2, help="Strengh of sharpen operator")

    args = parser.parse_args()
    if args.filter_type == 'mean':
        mean_filter(args.input_file, args.output_file, args.size)
    elif args.filter_type == 'median':
        median_filter(args.input_file, args.output_file, args.size)