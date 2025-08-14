import numpy as np
import cv2

def main():
    img_src = cv2.imread("blobs_pattern.png", cv2.IMREAD_GRAYSCALE)
    height, width = img_src.shape
    theta = -np.pi/4
    shiftx = 0
    shifty = height/2
    rotate_mat = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0,0,1]
    ]).reshape(3,3)
    shift_mat = np.array([
        [1, 0, shiftx],
        [0, 1, shifty],
        [0, 0, 1]
    ]).reshape(3,3)
    affine_mat = shift_mat @ rotate_mat
    # アフィン変換
    img_dst = cv2.warpAffine(
        img_src,
        affine_mat[:2, :],
        (width, height),
        flags=cv2.INTER_LINEAR,
    )

    # cv2.getRotationMatrix2D利用
    center = (width/2, height/2)
    angle = theta * 180 / np.pi
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1)
    img_dst = cv2.warpAffine(img_dst, rot_mat, (width, height), flags=cv2.INTER_CUBIC)
    cv2.namedWindow("img_src", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("img_src", img_src)
    cv2.namedWindow("img_dst", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("img_dst", img_dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()