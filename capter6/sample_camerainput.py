import cv2
import numpy as np
import math
import sys

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("カメラが開けませんでした。")
        sys.exit()
        return -1
    
    # ウィンドウを作成
    win_src = "src"
    cv2.namedWindow(win_src, cv2.WINDOW_AUTOSIZE)

    # 解像度の設定
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    #解像度の確認
    ret, img_src = cap.read()
    print(f"height, width: {img_src.shape[1], img_src.shape[0]}")

    # key入力で'q'が押された時にカメラに写っていた画像をimg_srcに保存する。それまでは、whileループでカメラの画像を表示する。
    while True:
        ret, img_src = cap.read() #キャプチャ
        cv2.imshow(win_src, img_src)
        key = cv2.waitKey(1)
        if( key == ord('q')):
            break
    cv2.imwrite("capture.png", img_src)# 保存処理
    cap.release()
    cv2.destroyAllWindows()
    return 0

main()