import numpy as np
import cv2

def main():
    folder_path = "rgbd-dataset/apple/apple_1/"
    filename_rgb = "apple_1_1_1_crop.png"
    filename_depth = "apple_1_1_1_depthcrop.png"
    img_src = cv2.imread(folder_path+filename_rgb, cv2.IMREAD_COLOR_BGR)
    img_depth = cv2.imread(folder_path+filename_depth, cv2.IMREAD_GRAYSCALE)
    height, width = img_src.shape[:2]
    print(f"depth max: {np.max(img_depth)}")

    # 深度2以下の画素を抽出
    thresh_depth = 2
    mask = cv2.threshold(img_depth, thresh_depth, 255, cv2.THRESH_BINARY_INV)[1]
    
    cv2.imshow("mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()