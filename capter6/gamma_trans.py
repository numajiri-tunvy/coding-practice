import cv2
import numpy as np

"""
ガンマ変換を行う関数
img: 入力画像
gamma: ガンマ値
"""
def gamma_trans(img, gamma):
    img_gamma = np.clip(np.uint8(255*((img/255)**(1/gamma))), 0, 255)
    return img_gamma


def main():
    src_img = cv2.imread("src.JPG", cv2.IMREAD_COLOR)
    if(src_img is None):
        print("Failed to load image")
        return -1;
    img_gamma = gamma_trans(src_img, 1.5)
    win_name = "gamma trans"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow("src_img", src_img)
    cv2.imshow(win_name, img_gamma)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    ## cv2
    gamma = 1.5
    # ルックアップテーブル作成
    lut = np.array(range(256), dtype=np.uint8)
    lut = np.clip(np.uint8(255*((lut/255)**(1/gamma))), 0, 255)
    img_gamma_cv2 = cv2.LUT(src_img, lut)
    cv2.namedWindow("gamma trans cv2", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("gamma trans cv2", img_gamma_cv2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0

main()