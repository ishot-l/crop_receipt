import cv2
import sys

def reduction(image, upper_limit = 2000):
    """
入力された画像を、縦横比はそのままに辺の長さがupper_limit以下になるよう縮小

入力：
    image ... cv2.imreadで読み込んだndarrayの画像
    upper_limit ... 上限値（入れなくてもいい）

出力：
    red_image ... 縮小されたndarrayの画像
    ratio ... 縮小率(float)
    """
    
    size = image.shape[:2]

    # 縮小比率を求める（長辺がupper_limitになるようにする）
    ratio = upper_limit  / max(size)

    # 縮小しなくていい場合
    #if ratio > 1:
    #    return image, 1

    # 求めた比率で画像を変換
    resize = tuple(map(lambda x: int(x * ratio), size[::-1]))
    red_image = cv2.resize(image, resize)

    return red_image, ratio

if __name__ == '__main__':
    """

直接呼び出すなら

python size_reducton.py [ファイル名] ([上限値])
（openCVが必要）

    """
    image = cv2.imread(sys.argv[1])

    if len(sys.argv) > 2:
        upper_limit = int(sys.argv[2])
    else:
        upper_limit = 2000

    red_image, restore_ratio = reduction(image, upper_limit)
    cv2.imwrite(sys.argv[1], red_image)
