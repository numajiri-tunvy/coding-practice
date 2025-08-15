import numpy as np
import cv2
"""
練習問題5.1
白、黄、シアン、緑、マゼンタ、赤、青、黒の8色のカラーバーを表示した画像を生成
width = 800, height = 240px
"""
def problem1(show = True):
    # 画像設定
    height = 240
    width = 800
    img_color_bar = np.zeros((height, width, 3), dtype=np.uint8)
    # 色設定
    colors = [(255,255,255), (255,255,0), (0,255,255), (0,255,0), (255,0,255), (255,0,0), (0,0,255), (0,0,0)]
    for i in range(len(colors)):
        img_color_bar[0:height, i*(width//8):(i+1)*(width//8), :] = colors[i]
    if show:
        win_name = "color bar"
        cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(win_name, img_color_bar)
        cv2.waitKey(0)
        cv2.imwrite("sample1.png", img_color_bar)
        cv2.destroyAllWindows()
    return img_color_bar

"""
画像のR,G,Bの入れ替えを行う
original: (b,g,r)
"""
def problem2(c0, c1, c2, show = True):
    rgb = {'r': 2, 'R': 2, 'g': 1, 'G': 1, 'b': 0, 'B': 0}
    index_list = [rgb.get(c0, None), rgb.get(c1, None), rgb.get(c2, None)]
    if(index_list.count(None) > 0):
        print("Invalid color")
        return
    img_color_bar = problem1(False)
    img_color_bar_shuffle = np.zeros(img_color_bar.shape, dtype=np.uint8)
    for ch in range(img_color_bar.shape[2]):
        print(ch)
        img_color_bar_shuffle[:,:,ch] = img_color_bar[:,:,index_list[ch]]
    if(show):
        win_name = "channel changed color bar"
        cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(win_name, img_color_bar_shuffle)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return img_color_bar_shuffle


"""
練習問題5.3
RGB->YCbCr
"""
def problem3(img = None, show = True):
    if(img is None):
        img = cv2.imread("sample1.png")
        if(img is None):
            print("Failed to load image")
            return -1
    img = np.float32(img)
    b,g,r = cv2.split(img)
    img_ybcr = np.zeros(img.shape, dtype=np.uint8)
    img_ybcr[:,:,0] = np.uint8(16+(235-16)*(0.257*r + 0.504*g + 0.098*b + 16)//253)#16~235に丸める
    img_ybcr[:,:,1] = np.uint8(16+(240-16)*(-0.148*r - 0.291*g + 0.439*b + 128)//240)#16~240に丸める
    img_ybcr[:,:,2] = np.uint8(16+(240-16)*(0.439*r - 0.368*g - 0.071*b + 128)//240)#16~240に丸める
    if(show):
        win_name = "YCbCr"
        cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(win_name, cv2.cvtColor(img_ybcr, cv2.COLOR_YCrCb2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return img_ybcr
    
"""
練習問題5.4
彩度または色相だけの画像を生成
saturationOrHue: 0:彩度, 1:色相 (0以外は色相)
"""
def problem4(saturationOrHue, img = None, show = True):
    if(img is None):
        img = cv2.imread("sample1.png")
        if(img is None):
            print("Failed to load image")
            return -1
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(img_hsv)
    result_img= np.zeros(img.shape[0:2], dtype=np.uint8)
    win_name = "saturation image"
    if(saturationOrHue == 0):
        result_img = s
    else:
        win_name = "hue image"
        result_img = np.uint8(np.clip(h/179*255, 0, 255))
    if(show):
        cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(win_name, result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return result_img

"""
練習問題5.5
入力画像の明度をsする
"""
def problem5(s:int, img = None, show = True):
    if(img is None):
        img = cv2.imread("sample1.png")
        if(img is None):
            print("Failed to load image")
            return -1
    # 明度変化量sを0<=|s|<=255に丸める
    s = np.clip(abs(s), 0, 255)*(1 if s > 0 else -1)
    # 明度を変化させる
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hsv[:,:,2] = np.uint8(np.clip(np.int32(img_hsv[:,:,2] + s), 0, 255))
    img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    if(show):
        win_name = "brightness changed image"
        cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(win_name, img_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return img_bgr


"""
練習問題5.6
入力画像の彩度を+sする
"""
def problem6(s:int, img = None, show = True):
    if(img is None):
        img = cv2.imread("sample1.png")
        if(img is None):
            print("Failed to load image")
            return -1
    plusMinus = 1 if s > 0 else -1
    s = np.clip(abs(s), 0, 255)*plusMinus
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hsv[:,:,0] = np.uint8(np.clip(np.int32(img_hsv[:,:,0]) + s, 0, 179))
    img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    if(show):
        win_name = "saturation changed image"
        cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(win_name, img_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return img_bgr

"""
練習問題5.7
入力画像の色相を変化させる
"""
def problem7(s:int, img = None, show = True):
    if(img is None):
        img = cv2.imread("sample1.png")
        if(img is None):
            print("Failed to load image")
            return -1
    # 色相変化量sを0<=|s|<=179に丸める
    s = np.clip(abs(s), 0, 179)*(1 if s > 0 else -1)
    # 色相を変化させる
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_img,s_img,v_img = cv2.split(img_hsv)
    h_img = np.uint8(np.clip(np.int32(h_img) + s, 0, 179))
    img_hsv = cv2.merge((h_img, s_img, v_img))
    img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
def main():
    problem6(-100,show=True)

main()