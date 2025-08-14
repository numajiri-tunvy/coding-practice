import numpy as np
import cv2

def main():
    img_src = cv2.imread("src.JPG", cv2.IMREAD_COLOR_BGR)
    height, width = img_src.shape[:2]
    # 射影変換行列
    pts1 = np.float32([[0, 0], [0, height], [width, height], [width, 0]])
    proh = round(height*3/4)
    prow = round(width*3/4)
    pts2 = np.float32([
        [round((width-prow)/2), height-proh],
        [0,height], 
        [width, height], 
        [round((width-prow)/2) + prow, height-proh]
        ])
    proj_mat = cv2.getPerspectiveTransform(pts1, pts2)
    img_dst = cv2.warpPerspective(img_src, proj_mat, (width, height), flags=cv2.INTER_CUBIC)
    # 表示
    cv2.namedWindow("img_src", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("img_src", img_src)
    cv2.namedWindow("img_dst", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("img_dst", img_dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()