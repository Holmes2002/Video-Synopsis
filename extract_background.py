# -*- coding: utf-8 -*-
# @Time    : 2020/1/1 18:29
# @Author  : zhanghao
# @FileName: get_background.py
# @Software: PyCharm

# a file used to create background image

import cv2
import numpy as np
from tqdm import tqdm
import argparse
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        type=str,
    )
    parser = parser.parse_args()
    return parser

def method_1(file):
    VIDEO_WIDTH = 1920
    VIDEO_HEIGHT = 1080
    capture = cv2.VideoCapture(file)
    mog = cv2.createBackgroundSubtractorMOG2()
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    while True:
        ret, image = capture.read()
        if ret is True:
            fgmask = mog.apply(image)
            ret, binary = cv2.threshold(fgmask, 220, 255, cv2.THRESH_BINARY)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, se)
            back_image = mog.getBackgroundImage()
            # cv2.imshow("back_image", back_image)
            # Press Q to stop!
            cv2.imwrite('background_MOG.jpg', back_image)
        else:
            break

    cv2.destroyAllWindows()
def method_2(file):
	cap = cv2.VideoCapture(file)
	n_frame = 300
	_,f = cap.read()
	lst_frame = []
	for i in tqdm(range(n_frame)):
	    for ii in range(5):
	        _,f = cap.read()
	    lst_frame.append(f)
	bg = np.median(np.array(lst_frame), axis=0).astype('uint8')
	cv2.imwrite('backgroundjpg', bg)
if __name__ == "__main__":
	args = parse_arguments()
	method_2(args.video_path)
