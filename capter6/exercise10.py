import numpy as np
import cv2

"""
問題10.1
画像の再標本化を行わない画像の拡大
"""
def problem1(img_src, sx, sy, aff = True):
    scale_mat = np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1]
    ]).reshape(3,3)
    height, width = img_src.shape[:2]
    if(aff):
        img_dst = cv2.warpAffine(
            img_src, 
            np.float32(scale_mat[:2, :]), 
            (width, height), 
            flags=cv2.INTER_CUBIC
            )
        return img_dst
    else:
        img_dst = np.zeros([height*sy, width*sx, 3], dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                p = np.array([x, y, 1])
                p_dst = scale_mat @ p
                x_dst = round(p_dst[0])
                y_dst = round(p_dst[1])
                img_dst[y_dst, x_dst] = img_src[y, x]
        return img_dst

"""
問題10.2
半時計回りにangle°回転と水平方向にshiftピクセル移動した画像を生成
shift[0]は水平方向の移動量、shift[1]は垂直方向の移動量
cは中心座標
"""
def problem2(img_src, angle, shift, c):
    [height, width] = img_src.shape[:2]
    # c[0],c[1を中心として反時計回りにangle°回転
    theta = angle * np.pi / 180
    rot_mat = np.array([
        [np.cos(theta), np.sin(theta), 0],
        [-np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ]).reshape(3,3)
    # 回転中心をcにするために、並行移動
    # 中心移動量計算
    c_rot =  rot_mat @ np.array([c[0], c[1], 1]).reshape(3,1)
    print(c_rot)
    dcx = c_rot[0][0] - c[0]
    dcy = c_rot[1][0] - c[1]
    print(dcx, dcy)
    shift_mat = np.array([
        [1, 0, shift[0]-dcx],
        [0, 1, shift[1]-dcy],
        [0, 0, 1]
    ]).reshape(3,3)
    affine_mat = shift_mat @ rot_mat
    img_dst = cv2.warpAffine(img_src, affine_mat[:2, :], (width, height), flags=cv2.INTER_CUBIC)
    return img_dst

def problem3(img_src, angle:float, isx: bool):
    theta = angle * np.pi / 180
    [height, width] = img_src.shape[:2]
    if(isx):
        # x軸方向のせん断変形
        shear_mat = np.array([
            [1, np.tan(theta), 0],
            [0, 1, 0],
            [0, 0, 1]
        ]).reshape(3,3)
    else:
        # y軸方向のせん断変形
        shear_mat = np.array([
            [1, 0, 0],
            [np.tan(theta), 1, 0],
        ])
    img_dst = cv2.warpAffine(img_src, shear_mat[:2, :], (width, height), flags=cv2.INTER_CUBIC)
    return img_dst



def main():
    img_src = cv2.imread("src.JPG", cv2.IMREAD_COLOR_BGR)
    [height, width] = img_src.shape[:2]
    #img_dst = problem2(img_src, 45, [0, 1000], [width/2, height/2])
    img_dst = problem3(img_src, 45, True)
    # 表示
    cv2.namedWindow("img_src", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("img_src", img_src)
    cv2.namedWindow("img_dst", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("img_dst", img_dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

main()