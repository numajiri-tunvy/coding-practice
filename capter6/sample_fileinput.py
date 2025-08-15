"""
opencv画像処理雛形
画像はファイル入力
出力はウィンドウ表示
"""

import cv2
import numpy as np

def main():
    file_src = "src.JPG" #入力画像ファイル
    file_dst = "dst.png" #出力画像ファイル

    #画像の読み込み
    #img_src = cv2.imread(file_src, cv2.IMREAD_COLOR) #カラー画像として読み込み
    img_src = cv2.imread(file_src, cv2.IMREAD_GRAYSCALE) #グレースケール画像として読み込み
    if(img_src is None):
        print("画像の読み込みに失敗しました。")
        return -1
    
    #画像の表示
    # ウィンドウを作成
    win_src = "src"
    cv2.namedWindow(win_src, cv2.WINDOW_AUTOSIZE)
    win_dst = "dst"
    cv2.namedWindow(win_dst, cv2.WINDOW_AUTOSIZE)

    #########################################################
    ############### 画像処理ここから ###############
    #########################################################

    img_dst = cv2.flip(img_src, 2)
    #########################################################
    ############### 画像処理ここまで ###############
    #########################################################

    #画像の表示
    cv2.imshow(win_src, img_src)
    cv2.imshow(win_dst, img_dst)

    cv2.waitKey(0)
    return 0

main()