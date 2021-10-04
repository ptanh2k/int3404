"""
Name: Pham Tuan Anh
Class: K63-K2
MSSV: 18020116

You should understand the code you write.
"""

import numpy as np
import cv2
import sys


def q_0(input_file, output_file, delay=1):
    """

    :param input_file:
    :param output_file:
    :param delay:
    :return:
    """
    img = cv2.imread(input_file, cv2.IMREAD_COLOR)
    
    if img is None:
        sys.exit("Could not read the image")
    
    cv2.imshow('Apple image', img)
    cv2.waitKey(delay)

    cv2.imwrite(output_file, img)
    
"""
c2.waitKey(a) sẽ đợi trong một khoảng thời gian ít nhất a (ms). 
Trong khoảng thời gian đó nếu người dùng nhấn phím bất kỳ, chương trình sẽ dừng; 
nếu không, chương trình sẽ tiếp tục chạy ít nhất cho đến khi hết a (ms). 
Tham khảo: https://web.archive.org/web/20120122022754/http://opencv.willowgarage.com/wiki/documentation/c/highgui/WaitKey
"""

def q_1(input_file):
    """
        imread() -> Order: BRG (Blue, Green, Red)
    """
    img1 = cv2.imread(input_file, cv2.IMREAD_COLOR)
    
    if img1 is None:
        sys.exit("Could not read the image")
        
    (height, width, depth) = img1.shape
    print("height={}, width={}, depth={}".format(height, width, depth))
    
    yCrCbImg = cv2.cvtColor(img1, cv2.COLOR_BGR2YCR_CB)
    
    avgY = np.mean(yCrCbImg[:, : , 0])
    avgCr = np.mean(yCrCbImg[:, :, 1])
    avgCb = np.mean(yCrCbImg[:, : , 2])
    
    print("YCrCb")
    print("Average of Y: %.2f" % avgY)
    print("Average of Cr: %.2f" % avgCr)
    print("Average of Cb: %.2f" % avgCb)
    
    rgbImg = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    
    avgR = np.mean(rgbImg[:, :, 0])
    avgG = np.mean(rgbImg[:, :, 1])
    avgB = np.mean(rgbImg[:, :, 2])

    print("RGB")
    print("Average of R: %.2f" % avgR)
    print("Average of G: %.2f" % avgG)
    print("Average of B: %.2f" % avgB)    
    
def q_2(input_file):
    img2 = cv2.imread(input_file, cv2.IMREAD_COLOR)
    
    if img2 is None:
        sys.exit("Could not read the image")

    clear_apple = img2[297:471, 363:539]
    cv2.imshow("Clear apple", clear_apple)
    cv2.imwrite("./result/clear_apple.png", clear_apple)
    
    blurred_apple = img2[39:127, 90:176]
    cv2.imshow("Blurred apple", blurred_apple)
    cv2.imwrite("./result/blurred_apple.png", blurred_apple)
    cv2.waitKey(0)
    
if __name__ == "__main__":

    q_0('./sample_data/apple.png', './result/test_apple.png', 1000)
    q_1('./sample_data/chromatic_aberration.png')
    q_2("./sample_data/apple.png")


