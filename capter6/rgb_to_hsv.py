import cv2
import numpy as np
import math

def main():
    # 画像ファイル入力
    src_file = "src.JPG"
    img = cv2.imread(src_file, cv2.IMREAD_COLOR)

    # RGB -> HSV
    img_hsv = np.zeros(img.shape, dtype=np.uint8)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            b,g,r = img[y,x]
            channels:dict[str, float] = {'r':float(r)/255, 'g':float(g)/255, 'b':float(b)/255}
            diff = 0.0
            max_channel = max(channels, key=channels.get)
            max_val = max(channels.values());
            min_val = min(channels.values());
            c = 0
            if(max_channel == 'r'):
                c = 0
                diff = (channels['g']) - (channels['b'])
            elif(max_channel == 'g'):
                c = 120
                diff = (channels['b']) - (channels['r'])
            else:
                c = 240
                diff = (channels['r']) - (channels['g'])
            v = max_val
            h = 0 if max_val == min_val else ((60*(diff/(max_val - min_val)) + c)*10 % 3600) /10
            while(h < 0):
                h += 360.0
            s = (max_val - min_val) / max_val if max_val > 0 else 0
            img_hsv[y,x,0] = np.uint8(np.clip(h*179/360.0, 0, 179))
            img_hsv[y,x,1] = np.uint8(np.clip(s*255, 0, 255))
            img_hsv[y,x,2] = np.uint8(np.clip(v*255, 0, 255))

    win_src = "rgb image"
    win_dst = "hsv image"
    cv2.namedWindow(win_src, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(win_dst, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(win_src, img)
    img_hsv_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow(win_dst, img_hsv_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



    
main()