"""
Name: Pham Tuan Anh
Class: K63-K2
MSSV: 18020116

You should understand the code you write.
"""

import numpy as np
import cv2


def q_0(input_file, output_file, delay=1):
    """

    :param input_file:
    :param output_file:
    :param delay:
    :return:
    """
    img = cv2.imread(input_file, 1)
    cv2.imshow('Test img', img)
    cv2.waitKey(delay)

    cv2.imwrite(output_file, img)
    
"""
c2.waitKey(a) sẽ đợi trong một khoảng thời gian ít nhất a (ms). 
Trong khoảng thời gian đó nếu người dùng nhấn phím bất kỳ, chương trình sẽ dừng; 
nếu không, chương trình sẽ tiếp tục chạy ít nhất cho đến khi hết a (ms). 
Tham khảo: https://web.archive.org/web/20120122022754/http://opencv.willowgarage.com/wiki/documentation/c/highgui/WaitKey
"""

def q_1():
    print("Task 1")


def q_2():
    print("Task 2")


if __name__ == "__main__":

    q_0('./sample_data/apple.png', './result/test_apple.png', 1000)
#     q_1()
#     q_2()


