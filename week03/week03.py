"""
Name: Pham Tuan Anh
Class: K63K2
MSSV: 18020116

You should understand the code you write.
"""

import numpy as np
import cv2
import matplotlib as plt
import argparse


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

def q_2(input_file, output_file):
    """
    Performs a histogram equalization on the input image.
    """
    img = cv2.imread(input_file, 0)

    # Convert image to gray channgel
    img_eq = None

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
