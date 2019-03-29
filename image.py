import numpy as np
import cv2
import sys
import contrastchange_not_save as cchange
import os 
import size_reduction

# ------  定数いろいろ   ------

# 閾値調整用 
VALUE_THRESHOLD = 200
SATURATION_THRESHOLD = 20 # 20

# テスト用にいろいろなものを出力するかどうか
TEST = False

# 横長だと漢字を正しく読み取りづらくなるため若干縦長に補正する、その割合
HEIGHT_CORRECTION_RATIO = 1.2

# 実際に処理する際のサイズ（高速化のため縮小を行う）
REDUC_SIZE = 2000
# 【参考】 ... 2000なら約0.5秒（元画像のサイズにもよる）



def main(img_path, cont_up = False):
    """

メインロジック

    """

    raw_image = cv2.imread(img_path)

    # コントラストを向上させる
    if cont_up:
        raw_image = cchange.change(raw_image)  

    image, ratio = size_reduction.reduction(raw_image, REDUC_SIZE)
    restore_ratio = 1 / ratio # 復元用

    height, width = image.shape[:2]
    menseki = height * width

    # hsv変換してから、白色領域を抽出
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    threashhold_min = np.array([0,0,VALUE_THRESHOLD], np.uint8)
    threashhold_max = np.array([255,SATURATION_THRESHOLD,255], np.uint8)
    image_white = cv2.inRange(image_hsv, threashhold_min, threashhold_max)

    if TEST:
        imgshow(image_white, ratio=0.3)
        cv2.imwrite("whitedayo.jpg", image_white)

    # 輪郭取得
    _, contours, hierarchy = cv2.findContours(image_white,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # 輪郭から面積の一番大きいものを抽出
    best_contour = max(contours, key=lambda x:cv2.contourArea(x))

    if TEST:
        img_best_cont = cv2.drawContours(image.copy(), [best_contour], -1, (0, 255, 0), 3)
        imgshow(img_best_cont, ratio=0.3)

    # 輪郭の凸包を取得
    box = cv2.convexHull(best_contour)

    if TEST:
        imbox = cv2.polylines(image.copy(), [box], True, (255, 0, 0))
        imgshow(imbox, ratio=0.3)

    # 輪郭だけを画像に出力
    white = np.zeros((height, width), np.uint8) # np.uint8の指定をしないとエラーになる
    image_cont_only = cv2.drawContours(white, [box], -1, 255, 3)
    if TEST:
        imgshow(image_cont_only, ratio=0.3)

    # ハフ変換によって直線検出
    lines = cv2.HoughLines(image_cont_only,1,np.pi/180,300)

    # θでならびかえ（あまり意味はない、、）
    lines = sorted(lines, key= lambda x: x[0][1])

    arranged_lines = []

    # 近くにある複数の直線をまとめる
    for line in lines:
        for key, a_lines in enumerate(arranged_lines):
            if radians_angle(get_radian_mean(a_lines), line[0][1]) < (np.pi / 4) and abs(get_rho_mean(a_lines) - line[0][0]) < 100:
                arranged_lines[key].append(line)
                break
        else:
            arranged_lines.append([line])

    # より多くの直線を集めたグループから優先させる
    arranged_lines = sorted(arranged_lines, key= lambda x: -len(x))

    # クラスタリングした集団ごとに代表的な直線をつくる
    mean_lines = []
    for lines in arranged_lines:
        rho = get_rho_mean(lines)
        theta = get_radian_mean(lines)
        mean_lines.append([rho, theta])

    # （テスト用）描画
    if TEST:
        lines_img = image.copy()
        for key, lines in enumerate(arranged_lines):
            color = (0, 0, 255)        
            for line in lines:
                rho, theta = line[0]
                start_p, end_p = hough_line_calc(rho, theta)
                cv2.line(lines_img, start_p, end_p, color, 3)

            rho = get_rho_mean(lines)
            theta = get_radian_mean(lines)
            start_p, end_p = hough_line_calc(rho, theta)
            cv2.line(lines_img, start_p, end_p, (255, 0, 0), len(lines)*2)

        imgshow(lines_img, ratio=0.3)
    
    # 辺の選択
    sides = [[] for i in range(4)]
    sides[0] = mean_lines[0] # いちばん多いやつはとりあえず入れておく   
    
    # のこりの直線について、辺となるかチェック
    for line in mean_lines[1:]:
        if not [] in sides:
            break
        rho, theta = line
        angle = radians_angle(sides[0][1], theta) 

        # 最初に選んだ直線とだいたい平行である場合
        if angle < np.pi * 0.1:
            # すでに向かいの辺が決まっているなら追加しない
            if sides[2]:
                break
            # 画像内で交点を持っている場合も追加しない
            x, y = map(int, get_interaction(sides[0], line))
            if not ( 0 < x < width and 0 < y < height ):
                sides[2] = line
                continue

        # 最初に選んだ直線とだいたい垂直である場合
        elif angle > np.pi * 0.4:
            if sides[3]: # 3に値が入っているなら追加しない
                continue
            if not sides[1]: # 1に値が入ってないなら追加しない
                sides[1] = line
                continue
            # 画像内で交点をもつかどうか
            x, y = map(int, get_interaction(sides[1], line))
            if not ( 0 < x < width and 0 < y < height ):
                sides[3] = line
                continue

        # 垂直でも平行でもないなら含めない
        else:
            continue

    if [] in sides:
        return False # 四隅を発見できなかったならば検出失敗、Falseを返しておわり

    # 四辺描画
    if TEST:
        sides_img = image.copy()
        for line in sides:
            rho, theta = line
            start_p, end_p = hough_line_calc(rho, theta)
            cv2.line(sides_img, start_p, end_p,(255, 0, 0),3)
        imgshow(sides_img, ratio=0.3)

    # 四隅の抽出
    apexes = []
    for i in range(-1, 3):
        apexes.append(tuple(map(int, get_interaction(sides[i], sides[i+1]))))

    if TEST:
        print(apexes)

        apex_img = image.copy()
        for apex in apexes:
            apex_img = cv2.circle(apex_img, apex, 30, (0, 0, 255), thickness=-1)

        imgshow(apex_img, ratio=0.3)


    # 向きを確定
    apexes = sorted(apexes, key=lambda x: x[1])

    top_half = apexes[:2]
    bottom_half = apexes[2:]

    # widthを決定
    l_top = np.array(min(top_half, key=lambda x:x[0]))
    r_top = np.array(max(top_half, key=lambda x:x[0]))
    l_btm = np.array(min(bottom_half, key=lambda x:x[0]))
    r_btm = np.array(max(bottom_half, key=lambda x:x[0]))

    # サイズの復元
    l_top = np.array(list(map(lambda x: int(x * restore_ratio), l_top)))
    r_top = np.array(list(map(lambda x: int(x * restore_ratio), r_top)))
    l_btm = np.array(list(map(lambda x: int(x * restore_ratio), l_btm)))
    r_btm = np.array(list(map(lambda x: int(x * restore_ratio), r_btm)))

    # 変換後の長さを確定
    after_width = max(get_length(l_top, r_top), get_length(l_btm, r_btm))
    after_height = max(get_length(l_top, l_btm), get_length(r_top, r_btm))

    after_height = int(after_height * HEIGHT_CORRECTION_RATIO)

    if TEST:
        print(after_width, "x", after_height)

    # (中心にあるなら：)画像を変換
    pts1 = np.float32([l_top, r_top, l_btm, r_btm])
    pts2 = np.float32([[0, 0], [after_width, 0], [0, after_height], [after_width, after_height]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(raw_image, M, (after_width, after_height))

    if TEST:
        imgshow(dst, ratio=0.3)
        cv2.imwrite("output.jpg", dst)

    return dst # 画像を返す



def radians_angle(rad_1, rad_2):
    """
ラジアン間の角度を取得する関数
    """
    num_angle = abs(rad_1 - rad_2) # 数字上の角度
    angle = min(num_angle, np.pi - num_angle) # 実際のアングル
    return angle

def get_length(point_1, point_2):
    """
二点間の距離を取得する関数
    """

    u = point_1 - point_2
    return int(np.linalg.norm(u))

def get_radian_mean(lines):
    """
複数の直線の角度の平均を求める関数
    """

    rad_sum = 0
    for line in lines:
        rad_sum += line[0][1]

    rad_mean = rad_sum / len(lines)
    return rad_mean

def get_rho_mean(lines):
    """
複数の直線のρの平均を求める関数
    """
    rho_sum = 0
    for line in lines:
        rho_sum += line[0][0]

    rho_mean = rho_sum / len(lines)
    return rho_mean

def get_interaction(line1, line2):
    """
交点を取得する関数
ρ, θからなるlineをふたつ入力すると、(x, y)でかえってくる
    """
    rho1, theta1 = line1
    rho2, theta2 = line2

    # 逆行列が存在しない場合には計算せずに返す
    if theta1 == theta2:
        return -100, -100

    mat = np.array([[np.cos(theta1), np.sin(theta1)],
                    [np.cos(theta2), np.sin(theta2)]])

    mat_1 = np.linalg.inv(mat)

    rho_vec = np.array([[rho1],
                        [rho2]])

    x_y = np.dot(mat_1, rho_vec)

    return x_y

def hough_line_calc(rho, theta, length=10000):
    """
ρ, θ から 二次元座標の始点・終点を求める関数
    """

    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + length*(-b))
    y1 = int(y0 + length*(a))
    x2 = int(x0 - length*(-b))
    y2 = int(y0 - length*(a))

    start = (x1, y1)
    end = (x2, y2)

    return start, end

def imgshow(image, title="image", save=False, ratio=1, max_size=False):
    """
画像を表示する関数
比率、最大サイズ等によって表示の大きさを返ることができる（ratio, max_size）
    """
    if max_size:
        hikaku = max(image.shape[:2])
        ratio = max_size / hikaku
        if ratio > 1:
            ratio = 1

    if ratio != 1:
        image = cv2.resize(image, None, fx = ratio, fy = ratio)

    cv2.imshow(title,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    
    if len(sys.argv) < 2:
        print("引数をください")
        sys.exit()

    if len(sys.argv) > 2:
        if sys.argv[2] == 'True':
            TEST = True

    img_path = sys.argv[1]
    cropped = main(img_path)

    if type(cropped) == bool:
        print("レシートを検出できなかった")
    else:
        imgshow(cropped, max_size=500)
