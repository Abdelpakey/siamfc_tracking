from __future__ import division
import sys
import os
import numpy as np
from collections import namedtuple
import json
from PIL import Image
import cv2
import tensorflow as tf

# from src.tracker import tracker
# from src.parse_arguments import parse_arguments
from src.region_to_bbox import region_to_bbox
# from draw_box import get_bbox_from_image
from utils import get_bbox_from_image, natural_sort
from src.siamese_network import SiameseNetwork


def main(config_path, image_folder, start_idx=0, skip_frame=0):

    # load nn config
    with open(config_path) as json_file:
        config = json.load(json_file, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))         

    net_path = config.net_path

    '''Get images'''
    frame_name_list = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(".jpg")]
    frame_name_list = natural_sort(frame_name_list)[start_idx:]
    num_frames = np.size(frame_name_list)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        '''Load network'''
        siam_net = SiameseNetwork(sess, config, trainable=False)
        siam_net.load_mat_model(net_path)

        sess.run(tf.global_variables_initializer())

        # manually label first frame
        first_frame = cv2.imread(frame_name_list[start_idx])
        first_frame_bbox = get_bbox_from_image(first_frame)
        pos_x, pos_y, target_w, target_h = region_to_bbox(first_frame_bbox)

        pos_x = [pos_x]
        pos_y = [pos_y]
        target_w = [target_w]
        target_h = [target_h]

        # init with exemplar 'z' image
        siam_net.forward_z(first_frame, pos_x, pos_y, target_w, target_h)

        rnd_colors_bgr = np.array([(np.random.rand(3)*255).astype(np.int) for p in pos_x])

        for i in xrange(start_idx + 1, num_frames):
            if i % (skip_frame+1) != 0:
                continue
            frame = cv2.imread(frame_name_list[i])
            img_copy = frame
            bboxes = siam_net.forward(frame)  # get tracked bounding boxes from image

            for ix,bbox in enumerate(bboxes):
                color_bgr = tuple(rnd_colors_bgr[ix])
                # color_rgb_f = rnd_colors_rgb_float[ix]
                pt1 = bbox[:2].astype(np.int)
                pt2 = pt1 + bbox[2:4].astype(np.int)
                mid_pt = np.mean([pt1,pt2],axis=0).astype(np.int)
                cv2.putText(frame, "%d"%(ix), tuple(mid_pt), cv2.FONT_HERSHEY_SIMPLEX, 1, color_bgr, 2)
                img_copy = cv2.rectangle(img_copy, tuple(pt1), tuple(pt2), color_bgr)
                cv2.imshow("tracker (skip frame: %d)"%(skip_frame), img_copy)
            cv2.waitKey(5)


if __name__ == '__main__':
    #  python run_tracker_demo.py -cfg ./parameters/design.json -f ./data/interoll2/
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--config", type=str, help="config path", required=True)
    parser.add_argument("-f", "--folder", type=str, help="image folder", required=True)
    parser.add_argument("--start_idx", type=int, help="Start frame index (zero-indexed). Default 0", default=0)
    parser.add_argument("--skip", type=int, help="Number of frames to skip between inference. Default 0", default=0)
    args = parser.parse_args()

    image_folder = args.folder
    start_idx = args.start_idx
    cfg = args.config
    skip_frame = args.skip
    main(cfg, image_folder, start_idx, skip_frame)
