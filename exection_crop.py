# コマンドで呼び出された場合ファイルを書き換えるやつ
import image
import sys
import cv2

if __name__ == '__main__':

    img_path = sys.argv[1]
    sys.stderr.write(img_path + "\n")
    cropped = image.main(img_path)

    if type(cropped) == bool:
        sys.stderr.write("Cannot crop")
    else:
        cv2.imwrite(img_path, cropped)
        sys.stderr.write("Succeeded")
