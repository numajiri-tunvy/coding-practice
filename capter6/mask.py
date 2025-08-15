import numpy as np
import cv2

def make_mask(img_src):
    img_input = img_src
    if(len(img_src.shape) >= 3):
        # グレースケールに変換
        img_input = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    # 2値化
    ret, img_bin = cv2.threshold(img_input, 128, 255, cv2.THRESH_BINARY)
    #カラー画像生成
    msk_color = cv2.merge([img_bin, img_bin, img_bin])
    return msk_color

def main():
    img_src1 = cv2.imread("blobs_pattern.png", cv2.IMREAD_COLOR_BGR)
    img_src2 = cv2.imread("blobs_pattern2.png", cv2.IMREAD_COLOR_BGR)
    print(img_src1.shape)
    print(img_src2.shape)
    mask = make_mask(img_src1)
    #切り出し
    img_src_cut = cv2.bitwise_and(img_src1, mask)
    # マスク反転
    mask_inv = cv2.bitwise_not(mask)
    # img_srcにマスク適用
    img_src2_masked = cv2.bitwise_and(img_src2, mask_inv)
    img_dst = cv2.bitwise_or(img_src_cut, img_src2_masked)
    cv2.imshow("img_dst", img_dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
main()