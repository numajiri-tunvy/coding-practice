import cv2
from cv2.gapi import imgproc
import numpy as np

"""
半径r,中心(cx, cy) pxの円を描画する関数
"""
def draw_circle(img_src, r, cx, cy):
    [height, width] = img_src.shape[:2]
    resolution = r / (256)
    resolution = round(resolution) if resolution > 1 else 1
    N = 1024 * resolution
    for i in range(round(N/4)):
        theta = 2 * np.pi * i / N
        genx= round(r*np.cos(theta))
        geny = round(r*np.sin(theta))
        x0 = np.clip(genx + cx, 0, width-1)
        y0 = np.clip(geny + cy, 0, height-1)
        x1 = np.clip(-genx + cx, 0, width-1)
        y1 = np.clip(-geny + cy, 0, height-1)
        img_src[y1:y0, x1:x0] = 255
    return img_src


def main():
    width = 512
    height = 512
    img_src = np.zeros((width, height), dtype=np.uint8)
    #20個ランダムに円を描画
    """
    for i in range(40):
        r = np.random.randint(5, 50)
        cx = np.random.randint(0, width-1)
        cy = np.random.randint(0, height-1)
        img_src = draw_circle(img_src, r, cx, cy)
    """
    #img_src = draw_circle(img_src, 200, 256, 256)
    img_src = cv2.imread("src.JPG", cv2.IMREAD_COLOR_BGR)[0:512, 0:512]
    print(np.max(img_src))
    cv2.imshow("img_src", img_src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("blobs_pattern2.png", img_src)

main()
    
