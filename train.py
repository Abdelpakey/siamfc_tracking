import numpy as np
np.core.arrayprint._line_width = 160

import tensorflow as tf
import os
import os.path as osp
import time
import cv2
from PIL import Image
import functools
from scipy.misc import imresize
from src.region_to_bbox import region_to_bbox
# from src.crops import py_extract_crops_x
from src.siamese import _import_from_matconvnet, _find_params, _create_siamese# , _get_templates_score_map


def pad_frame(im, frame_sz, pos_x, pos_y, patch_sz, avg_chan):
    c = patch_sz / 2

    # calculate padding dimensions
    xleft_pad = tf.maximum(0, -tf.cast(tf.round(pos_x - c), tf.int32))
    ytop_pad = tf.maximum(0, -tf.cast(tf.round(pos_y - c), tf.int32))
    xright_pad = tf.maximum(0, tf.cast(tf.round(pos_x + c), tf.int32) - frame_sz[1])
    ybottom_pad = tf.maximum(0, tf.cast(tf.round(pos_y + c), tf.int32) - frame_sz[0])

    # get the maximum padded size
    npad = tf.reduce_max([xleft_pad, ytop_pad, xright_pad, ybottom_pad])  

    paddings = [[npad, npad], [npad, npad], [0, 0]]  
    im_padded = im

    # pad the image, where the new padded values take the average pixel value in the image
    if avg_chan is not None:
        im_padded = im_padded - avg_chan
    im_padded = tf.pad(im_padded, paddings, mode='CONSTANT')  # pads with zeros
    if avg_chan is not None:
        im_padded = im_padded + avg_chan
    return im_padded, npad

def tf_crop_and_resize(im, crop_info, sz_dst):
    # crop_info must follow: tl_y,tl_x,height,width,offset
    ci = crop_info
    offset = ci[4]
    tl_y = tf.cast(ci[0] + offset, tf.int32)
    tl_x = tf.cast(ci[1] + offset, tf.int32)
    height = tf.cast(ci[2] - offset*2, tf.int32)
    width = tf.cast(ci[3] - offset*2, tf.int32)
    crop = tf.image.crop_to_bounding_box(im, tl_y, tl_x, height, width)
    crop = tf.image.resize_images(crop, [sz_dst, sz_dst], method=tf.image.ResizeMethod.BILINEAR)
    # crops = tf.expand_dims(crop, axis=0)
    return crop


def extract_crops_z(im, npad, pos_x, pos_y, sz_src, sz_dst):
    c = sz_src / 2

    crop_info = get_crop_info(c, npad, pos_x, pos_y)
    crop_info_padded = tf.pad(crop_info, [[0,0],[0,1]]) # pad with 0 to represent offset value
    crops = tf.map_fn(lambda x: tf_crop_and_resize(im, x, sz_dst), crop_info_padded)
    return crops


config = {
    "exemplar_sz": 127,  # z
    "search_sz": 255,  # x
    "score_sz": 17,
    "tot_stride": 8,
    "context": 0.5,
    "pad_with_image_mean": True
}
hyperparameters = {
    "response_up": 16,  # upsample scale factor (during the upsample score map step)
    "window_influence": 0.25,
    "z_lr": 0.01,
    "scale_num": 1,  
    "scale_step": 1.04,
    "scale_penalty": 0.97,
    "scale_lr": 0.59,
    "scale_min": 0.2,
    "scale_max": 5
}
hp = hyperparameters
SCALE_NUM = 1 # hp["scale_num"]

def get_crop_info(c, npad, pos_x, pos_y):
    npad = tf.cast(npad, tf.float32)
    # get top-right corner of bbox and consider padding
    tl_x = npad + tf.round(pos_x - c)
    tl_y = npad + tf.round(pos_y - c)

    # Compute size from rounded co-ords to ensure rectangle lies inside padding.
    width = tf.round(pos_x + c) - tf.round(pos_x - c)
    height = tf.round(pos_y + c) - tf.round(pos_y - c)

    crop_info = tf.stack([tl_y,tl_x,height,width], axis=1)  # all floats
    return crop_info

def tf_extract_crops_x(im, npad, pos_x, pos_y, sz_src_list, sz_dst):
    batch_sz = tf.shape(pos_x)[0]
    im_h = tf.cast(tf.shape(im)[0], tf.float32)
    im_w = tf.cast(tf.shape(im)[1], tf.float32)
    im_cn = tf.shape(im)[2]

    max_sz = tf.reduce_max(sz_src_list, axis=1)
    c = max_sz / 2
    offset = tf.cast((tf.expand_dims(max_sz, axis=1) - sz_src_list) / 2, tf.float32)
    crop_info = get_crop_info(c, npad, pos_x, pos_y)  # returns y,x,h,w

    # pad crop info with offset value 
    crop_info_padded = tf.map_fn(lambda x: tf.concat( [tf.stack([x[0]]*SCALE_NUM), tf.expand_dims(tf.transpose(x[1]),axis=1)], axis=1), (crop_info, offset), dtype=tf.float32)
    crop_info_padded = tf.reshape(crop_info_padded, (batch_sz*SCALE_NUM, 5))  # B * SCALE_NUM, [y,x,h,w,sz_dst]

    crop_offset = crop_info_padded[:,-1]
    crop_inf_y1 = crop_info_padded[:,0] + crop_offset
    crop_inf_x1 = crop_info_padded[:,1] + crop_offset
    crop_inf_y2 = (crop_info_padded[:,2] + crop_inf_y1 - 2*crop_offset)
    crop_inf_x2 = (crop_info_padded[:,3] + crop_inf_x1 - 2*crop_offset)
    bboxes = tf.stack([crop_inf_y1 / im_h, crop_inf_x1 / im_w, crop_inf_y2 / im_h, crop_inf_x2 / im_w], axis=1)

    # box_ind = tf.range(0,batch_sz*SCALE_NUM)
    box_ind = tf.zeros([batch_sz*SCALE_NUM], tf.int32) # zero-indexed: all refer to the same image, since there is only one image i.e. batch size 1

    crops = tf.image.crop_and_resize(tf.expand_dims(im, axis=0), bboxes, box_ind, [sz_dst,sz_dst])  # 
    crops = tf.reshape(crops, (batch_sz, SCALE_NUM, sz_dst, sz_dst, im_cn))

    return crops


def _get_templates_score_map_old(net_z, net_x): #, params_names_list, params_values_list):
    # finalize network
    # z, x are [B, H, W, C] to [H, W, B, C]
    Bz = net_z.shape.as_list()[0]
    Bx = net_x.shape.as_list()[0]
    assert Bz==Bx, ('Z and X should have same Batch size (num_scales)')
    net_z = tf.transpose(net_z, perm=[1,2,0,3])  # (3, 17, 17, 32) to (17, 17, 3, 32)
    net_x = tf.transpose(net_x, perm=[1,2,0,3])  # (3, 49, 49, 32) to (49, 49, 3, 32)

    # Hz, Wz, B, C = tf.unstack(tf.shape(net_z))
    # Hx, Wx, Bx, Cx = tf.unstack(tf.shape(net_x))
    Hz, Wz, B, C = net_z.shape.as_list()
    Hx, Wx, Bx, Cx = net_x.shape.as_list()
    # assert B==Bx, ('Z and X should have same Batch size')
    # assert C==Cx, ('Z and X should have same Channels number')
    net_z = tf.reshape(net_z, (Hz, Wz, B*C, 1))  # (17, 17, 3, 32) to (17, 17, 96, 1)
    net_x = tf.reshape(net_x, (1, Hx, Wx, B*C))  # (49, 49, 3, 32) to (1, 49, 49, 96)
    # final is [1, Hf, Wf, BC]
    net_final = tf.nn.depthwise_conv2d(net_x, net_z, strides=[1,1,1,1], padding='VALID')  # (1, 33, 33, 96)
    # final is [B, Hf, Wf, C]
    net_final = tf.concat(tf.split(net_final, Bz, axis=3), axis=0)  # (3, 33, 33, 32)
    # final is [B, Hf, Wf, 1]  after reduce sum
    net_final = tf.expand_dims(tf.reduce_sum(net_final, axis=3), axis=3)  # # (3, 33, 33,1)
    # net_final = tf.reduce_sum(net_final, axis=3)  # # (3, 33, 33)

    return net_final

def sigmoid(x): 
    return 1. / (1+np.exp(-x))

def _get_templates_score_map(net_z, net_x): #, params_names_list, params_values_list):
    B, N, Hz, Wz, C =  net_z.shape.as_list()
    B, N, Hx, Wx, C =  net_x.shape.as_list()
    net_z_shape = tf.shape(net_z)
    net_x_shape = tf.shape(net_x)

    net_z = tf.transpose(net_z, perm=[2,3,1,4,0]) # (W,W,N,C,B)
    net_x = tf.transpose(net_x, perm=[2,3,1,4,0])

    net_z = tf.reshape(net_z, (Hz, Wz, N * C * net_z_shape[0], 1))  
    net_x = tf.reshape(net_x, (1, Hx, Wx, N * C * net_x_shape[0])) 
    net_final = tf.nn.depthwise_conv2d(net_x, net_z, strides=[1,1,1,1], padding='VALID') 
    net_final_width = net_final.shape.as_list()[1]

    nf = tf.reshape(net_final, (1, net_final_width, net_final_width, N*C, net_z_shape[0]))  # (1,W,W,N*C,B)
    nf = tf.transpose(nf, perm=[4,0,1,2,3])   # (B,1,W,W,N*C)

    nf = tf.concat(tf.split(nf, N, axis=4), axis=1)  # (B,N,W,W,C)

    nf = tf.reduce_sum(nf, axis=4)

    return nf

def bn_adjust_score_map(score_map, params_names_list, params_values_list):
    final_bn_names = _find_params('adjust', params_names_list)
    bn_beta = 0
    bn_gamma = 1
    bn_moving_mean = 0
    bn_moving_variance = 1

    final_bn_names_beta = [p for p in final_bn_names if p[-1]=='b']
    if len(final_bn_names_beta) > 0:
        bn_beta = params_values_list[params_names_list.index(final_bn_names_beta[0])]
    final_bn_names_gamma = [p for p in final_bn_names if p[-1]=='f' or p[-1]=='m']
    if len(final_bn_names_gamma) > 0:
        bn_gamma = params_values_list[params_names_list.index(final_bn_names_gamma[0])]
    final_bn_names_moments = [p for p in final_bn_names if p[-1]=='x']
    if len(final_bn_names_moments) > 0:
        bn_moments = params_values_list[params_names_list.index(final_bn_names_moments[0])]
        bn_moving_mean = bn_moments[:,0]
        bn_moving_variance = bn_moments[:,1]**2
    score_map_bn = tf.layers.batch_normalization(score_map, beta_initializer=tf.constant_initializer(bn_beta),
                                            gamma_initializer=tf.constant_initializer(bn_gamma),
                                            moving_mean_initializer=tf.constant_initializer(bn_moving_mean),
                                            moving_variance_initializer=tf.constant_initializer(bn_moving_variance),
                                            training=False, trainable=False)

    return score_map_bn

# placeholders
pos_x_ph = tf.placeholder(tf.float32, shape=[None])
pos_y_ph = tf.placeholder(tf.float32, shape=[None])
z_sz_ph = tf.placeholder(tf.float32, shape=[None])
x_sz_list_ph = tf.placeholder(tf.float32, shape=[None,None])


tf_filename = tf.placeholder(tf.string, [], name='filename')
image_file = tf.read_file(tf_filename)
# Decode the image as a JPEG file, this will turn it into a Tensor
image = tf.image.decode_jpeg(image_file)
image = 255.0 * tf.image.convert_image_dtype(image, tf.float32)
frame_sz = tf.shape(image)


# get the average pixel values, used to pad the crops
if config["pad_with_image_mean"]:
    avg_chan = tf.reduce_mean(image, axis=(0,1), name='avg_chan')
else:
    avg_chan = None

'''our z (exemplar) image'''
# 1) padding, based on context size
frame_padded_z, npad_z = pad_frame(image, frame_sz, pos_x_ph, pos_y_ph, z_sz_ph, avg_chan)
frame_padded_z = tf.cast(frame_padded_z, tf.float32)
# 2) crop to context window
z_crops = extract_crops_z(frame_padded_z, npad_z, pos_x_ph, pos_y_ph, z_sz_ph, config["exemplar_sz"])

'''our x (search) image'''
# 1) padding
frame_padded_x, npad_x = pad_frame(image, frame_sz, pos_x_ph, pos_y_ph, tf.reduce_max(x_sz_list_ph, axis=1), avg_chan)
frame_padded_x = tf.cast(frame_padded_x, tf.float32)
# 2) crop
x_crops = tf_extract_crops_x(frame_padded_x, npad_x, pos_x_ph, pos_y_ph, x_sz_list_ph, config["search_sz"])  # B,N,W,W,C
# x_crops_new = tf_extract_crops_x(frame_padded_x, npad_x, pos_x_ph, pos_y_ph, x_sz_list_ph, config["search_sz"])  # B,N,W,W,C
# x_crops_t = tf.transpose(x_crops, perm=[2,3,4,1,0])  # W,W,C,B,N
x_crops_reshaped = tf.reshape(x_crops, (tf.shape(x_crops)[0], config["search_sz"], config["search_sz"], 3)) # W,W,C,N*B
# x_crops_reshaped_t = tf.transpose(x_crops_reshaped, perm=[3,0,1,2])  # N*B,W,W,C

net_path = "./pretrained/2016-08-17.net.mat"
template_z, template_x, p_names_list, p_val_list = _create_siamese(net_path, x_crops_reshaped, z_crops) # N*B,Wo,Wo,Co
_,tx_width,tx_width,tx_cn = template_x.shape.as_list()
templates_x = tf.reshape(template_x, (tf.shape(x_crops_reshaped)[0], 1, tx_width, tx_width, tx_cn))

template_z = tf.expand_dims(template_z, axis=1)
templates_z = tf.tile(template_z, [1,1,1,1,1]) # one for each x_crop scale (3)

score_map = _get_templates_score_map(templates_z, templates_x)
score_map = bn_adjust_score_map(score_map, p_names_list, p_val_list)

def create_logisticloss_label(label_size, radius_thresh):
    h, w = label_size
    logloss_label = np.zeros(label_size, np.float32)
    label_center = [h/2,w/2]

    euclidean_dist = lambda a,b: np.linalg.norm(np.array(a)-np.array(b))

    for r in xrange(h):
        for c in xrange(w):
            dist_ = euclidean_dist(label_center, [r,c])  # dist from center
            logloss_label[r,c] = 1 if dist_ <= radius_thresh else -1
    return logloss_label

def create_binary_labels(label_size, radius_thresh):
    assert(len(label_size) == 2)
    h,w = label_size
    ll_label = create_logisticloss_label(label_size, radius_thresh)
    # 'balanced' case only for instance weights
    instance_weight = np.zeros(label_size, np.float32)
    sumP = len(ll_label[ll_label==1])
    sumN = h*w - sumP # len(ll_label[ll_label==0])
    weightP = 0.5 / sumP
    weightN = 0.5 / sumN
    for r in xrange(h):
        for c in xrange(w):
            instance_weight[r,c] = weightP if ll_label[r,c] == 1 else weightN
    return ll_label, instance_weight

if __name__ == '__main__':
    from utils import get_bbox_from_image, natural_sort
    from src.region_to_bbox import region_to_bbox

    net_path = "./pretrained/2016-08-17.net.mat"
    config_path = './parameters/design_ori.json'

    root_dir = '/home/vincent/hd/datasets/ILSVRC2015_crop'
    root_train_dir = osp.join(root_dir, 'Data/VID/train')
    sample_dir = osp.join(root_train_dir, 'ILSVRC2015_VID_train_0000')
    vid_list = [osp.join(sample_dir, f) for f in natural_sort(os.listdir(sample_dir))]
    batch_sz = 8
    vid_batch_list = vid_list[0:batch_sz]
    TRACK_ID = 0
    PICK_Z_IDX = 0
    PICK_X_IDX = PICK_Z_IDX + 10

    crop_z_batch = np.zeros((batch_sz, 127, 127, 3), dtype=np.float32)
    crop_x_batch = np.zeros((batch_sz, 255, 255, 3), dtype=np.float32)

    for b,v in enumerate(vid_batch_list):
        v_files = natural_sort(os.listdir(v))
        crop_x_files = [f for f in v_files if f.endswith('x.jpg')]
        crop_z_files = [f for f in v_files if f.endswith('z.jpg')]
        tracked_crop_z_files = [f for f in crop_z_files if int(f.split(".")[1])==TRACK_ID]
        tracked_crop_x_files = [f for f in crop_x_files if int(f.split(".")[1])==TRACK_ID]
        cz_file = osp.join(v, tracked_crop_z_files[PICK_Z_IDX])
        cx_file = osp.join(v, tracked_crop_x_files[min(PICK_X_IDX, len(tracked_crop_x_files)-1)])

        crop_z_img = cv2.imread(cz_file)
        crop_x_img = cv2.imread(cx_file)

        if crop_z_img is None:
            print("%s is empty!"%(cz_file))
        if crop_x_img is None:
            print("%s is empty!"%(cx_file))
        crop_z_batch[b] = crop_z_img
        crop_x_batch[b] = crop_x_img

    crop_x_batch = np.expand_dims(crop_x_batch, axis=1)
    # crop_z_batch = np.expand_dims(crop_z_batch, axis=1)


    crop_x_sample = crop_x_batch[1:2]
    crop_x_sample = np.tile(crop_x_sample, (1,3,1,1,1))
    crop_z_sample = crop_z_batch[1:2]
    # crop_z_sample = np.tile(crop_z_sample, (3,1,1,1))

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())


    crop_x_feat, crop_z_feat = sess.run([templates_x, templates_z], feed_dict={x_crops: crop_x_batch[:1], z_crops: crop_z_batch[:1]})
    sc_map = sess.run(score_map, feed_dict={x_crops: crop_x_sample, z_crops: crop_z_sample})
    # eltwise_labels = 
    # instanceWeight = 
    # instanceWeight * 

    # from src.siamese_network import SiameseNetwork
    # siam_net = SiameseNetwork(sess)
    
    eltwise_labels, instance_weight = create_binary_labels((17,17), 16/8)
    # log_loss = (sigmoid(sc_map)>0.5) != eltwise_labels
    # loss_weighted = log_loss * instance_weight
    # np.sum(loss_weighted)
    log_loss = np.sum( np.log( 1+np.exp(-eltwise_labels*sc_map)) * instance_weight ) * 0.5

    