import cv2
import numpy as np
import math

def main():
    [height, width] = [256, 256]
    img = np.zeros([height, width,3], dtype=np.uint8)
    # 円形パターンの作成
    r = 100 #px
    c = [width/2, height/2] # 中心座標
    N = 1024 #角度分解能
    for i in range(int(N/4)):
        theta = 2 * math.pi * i / N
        x0 = round(c[0] - r * math.cos(theta))
        x1 = round(c[0] + r * math.cos(theta))
        y1 = round(c[1] + r * math.sin(theta))
        y0 = round(c[1] - r * math.sin(theta))
        if x0 < 0 or x0 >= width or y0 < 0 or y0 >= height or x1 < 0 or x1 >= width or y1 < 0 or y1 >= height:
            continue
        img[y0:y1,x0:x1, 0:2] = 255

    cv2.namedWindow("img", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("img_b", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("img_g", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("img_r", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("dst_img", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("img", img)
    img_bgr = cv2.split(img)#各色の成分に分割する
    cv2.imshow("img_b", img_bgr[0])
    cv2.imshow("img_g", img_bgr[1])
    cv2.imshow("img_r", img_bgr[2])
    dst_img = cv2.merge((img_bgr[2],img_bgr[1], img_bgr[0]))
    cv2.imshow("dst_img", dst_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
main()