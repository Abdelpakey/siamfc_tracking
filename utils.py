import numpy as np
import os
import cv2
import re

from draw_box import get_bbox_from_image

def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def natural_sort(str_list):
    return sorted(str_list, key=natural_key)

def convert_video_to_img_folder(video_file, img_folder, dims=(None,None), ext=".jpg"):

    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        print("Failed to open video file %s"%(video_file))
        return

    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    frame_cnt = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret is False:
            break

        frame_cnt += 1

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_name = "%s%s"%(os.path.join(img_folder, str(frame_cnt)), ext)
        cv2.imwrite(img_name, frame)
        print("Saved to %s"%(img_name))

        # cv2.imshow('frame',gray)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    print("Finished saving to %s. Total frames %d"%(img_folder, frame_cnt))
    cap.release()
    cv2.destroyAllWindows()

def display_img_folder(image_folder, ext=".jpg"):
    frame_name_list = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(ext)]
    frame_name_list = natural_sort(frame_name_list)
    for f in frame_name_list:
        print(f)
        cv2.imshow("asd", cv2.imread(f))
        cv2.waitKey(20)


if __name__ == '__main__':
    img_folder = "./data/interoll2"
    convert_video_to_img_folder("./data/interoll2.avi", img_folder)
    # display_img_folder(img_folder)