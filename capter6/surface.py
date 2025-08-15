from concurrent.futures import thread
import numpy as np
import cv2
import math


def count_xy(y, x, index):
    if(index == 0):
        return (y+1, x)
    elif(index == 1):
        return (y+1, x+1)
    elif(index == 2):
        return (y, x+1)
    elif(index == 3):
        return (y-1, x+1)
    elif(index == 4):
        return (y-1, x)
    elif(index == 5):
        return (y-1, x-1)
    elif(index == 6):
        return (y, x-1)
    elif(index == 7):
        return (y+1, x-1)
    else: return (y, x)

def search_perimeter8(img_src):
    [height, width] = img_src.shape[:2]
    #画素探索して初めて255になったらその画素を返す
    start_x = None
    start_y = None
    for i in range(height*width):
        y = i // width
        x = i % width
        if(img_src[y, x] == 255):
            print(f"start_x: {x}, start_y: {y}")
            start_x = x
            start_y = y
            break
    if(start_x is None or start_y is None):
        print("start_x or start_y is None")
        return 0
    # 8近傍の画素を探索
    index = 0
    [y0,x0] = [start_y, start_x]
    perimeter = 0
    while(True):
        found = False
        # 近傍の白画素を探索
        for i in range(8):
            [y1, x1] = count_xy(y0,x0, (i + index) % 8)
            if(y1 < 0 or y1 >= height or x1 < 0 or x1 >= width):
                continue
            if(img_src[y1, x1] == 255):
                # 近傍の白画素を探索したら、その画素を起点に探索を続ける
                index = (i + index - 1) % 8
                found = True
                break
        if(not found):
            print("周囲長が見つかりませんでした")
            break
        distance = np.sqrt((y1 - y0) ** 2 + (x1 - x0) ** 2)
        if(distance <= 0):
            print("distance <= 0")
            break
        elif(distance < np.sqrt(2)):
            # distance == 1
            [y0, x0] = [y1, x1]
            perimeter += 1
        else:
            perimeter += np.sqrt(2)
            [y0, x0] = [y1, x1]
        if(x1 == start_x and y1 == start_y):
            break
    return perimeter

"""
ラベリング処理を行い、最大面積のブロブを返す
img_src: 入力画像
return: 最大面積のブロブs
"""
def getMaxSurfaceBlob(img_src):
    nlabel, img_lab = cv2.connectedComponents(img_src)
    max_s_lab = max(
        range(1, nlabel), 
        key=lambda lab: cv2.contourArea(
            cv2.findContours(
                cv2.compare(img_lab, lab, cv2.CMP_EQ), 
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )[0][0]
        )
    )
    max_s_blob = cv2.compare(img_lab, max_s_lab, cv2.CMP_EQ)
    return max_s_blob

"""
面積フィルター
img_src: 入力画像
thread: 面積閾値
return: 面積フィルター後の画像
"""
def surface_filter(img_src, thread):
    nlabel, img_lab = cv2.connectedComponents(img_src)
    imgs = [
        cv2.compare(img_lab, lab, cv2.CMP_EQ)
        for lab in range(1, nlabel)
    ]
    filtered_img = [
        img for img in imgs
        if(
            cv2.contourArea(
                cv2.findContours(
                    img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )[0][0]
            ) > thread
        )
    ]
    return filtered_img

def main():
    img_src = cv2.imread("blobs_pattern.png", cv2.IMREAD_GRAYSCALE)
    if(img_src is None):
        print("画像を読み込めませんでした")
        return
    _, img_src = cv2.threshold(img_src, 128, 255, cv2.THRESH_BINARY, img_src)
    [height, width] = img_src.shape[:2]
    contours = cv2.findContours(img_src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    # 面積
    s = cv2.contourArea(contours[0])
    print(f"面積: {s}")
    # 周囲長
    l = cv2.arcLength(contours[0], True)
    print(f"周囲長: {l}")
    print(f"周囲長(自作): {search_perimeter8(img_src)}")
    #円形度
    circularity = 4 * np.pi * s / (l ** 2)
    print(f"円形度: {circularity}")
    # 重心
    m = cv2.moments(contours[0])
    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"]
    print(f"重心: ({cx}, {cy})")
    # 主軸角度
    ang = 0.5 * math.atan2(2*m["mu11"], m["mu20"] - m["mu02"])
    print(f"主軸角度: {ang}")

    # ラベリング処理
    nlabel, img_lab = cv2.connectedComponents(img_src)
    img_dst = cv2.compare(img_lab, 4, cv2.CMP_EQ)

    # ラベルの面積を求める
    max_s_blob = getMaxSurfaceBlob(img_src)

    # 面積フィルター
    thread = 5000
    filtered_imgs = surface_filter(img_src, thread)
    for i, img in enumerate(filtered_imgs):
        cv2.namedWindow(f"filtered img {i}", cv2.WINDOW_AUTOSIZE)
        cv2.imshow(f"filtered img {i}", img)

    # 面積最大のブロブの重心
    m = cv2.moments(max_s_blob)
    cx = round(m["m10"] / m["m00"])
    cy = round(m["m01"] / m["m00"])
    ang = 0.5 * math.atan2(2*m["mu11"], m["mu20"] - m["mu02"])
    # 重心を赤色でマーク
    max_s_blob_color = cv2.merge([max_s_blob, max_s_blob, max_s_blob])
    max_s_blob_color[cy-5:cy+5, cx-5:cx+5] = [0, 0, 255]
    # 主軸角度を青色でマーク y = ax + b
    a = math.tan(ang)
    b = cy - a * cx
    for x in range(width):
        y = round(a*x+b) if(a*x+b >= 0 and a*x+b < height) else None
        if(y is not None):
            max_s_blob_color[y,x] = [255, 0, 0]
    cv2.namedWindow("max surface blob", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("max surface blob", max_s_blob_color)

    # ラベル1の画像
    cv2.namedWindow("label 1 image", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("label 1 image", img_dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()