import numpy as np
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

def pad_frame_x(im, frame_sz, pos_x, pos_y, patch_sz, avg_chan):
    c = patch_sz / 2
    xleft_pad = tf.maximum(0, -tf.cast(tf.round(pos_x - c), tf.int32))
    ytop_pad = tf.maximum(0, -tf.cast(tf.round(pos_y - c), tf.int32))
    xright_pad = tf.maximum(0, tf.cast(tf.round(pos_x + c), tf.int32) - frame_sz[1])
    ybottom_pad = tf.maximum(0, tf.cast(tf.round(pos_y + c), tf.int32) - frame_sz[0])

    # pad_data = tf.stack([xleft_pad,ytop_pad,xright_pad,ybottom_pad],axis=1)
    npad = tf.reduce_max([xleft_pad, ytop_pad, xright_pad, ybottom_pad], axis=0)  
    max_pad = tf.reduce_max(npad)

    paddings = [[max_pad, max_pad], [max_pad, max_pad], [0, 0]]  
    im_padded = im

    # pad the image, where the new padded values take the average pixel value in the image
    if avg_chan is not None:
        im_padded = im_padded - avg_chan
    im_padded = tf.pad(im_padded, paddings, mode='CONSTANT')  # pads with zeros
    if avg_chan is not None:
        im_padded = im_padded + avg_chan
    return im_padded, npad


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
    "scale_num": 3,  # TRY 5 SCALES 
    "scale_step": 1.04,
    "scale_penalty": 0.97,
    "scale_lr": 0.59,
    "scale_min": 0.2,
    "scale_max": 5
}
hp = hyperparameters
SCALE_NUM = hp["scale_num"]

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

def extract_crops_x(im, npad, pos_x, pos_y, sz_src_list, sz_dst):
    batch_sz = tf.shape(pos_x)[0]
    max_sz = tf.reduce_max(sz_src_list, axis=1)
    c = max_sz / 2
    offset = tf.cast((tf.expand_dims(max_sz, axis=1) - sz_src_list) / 2, tf.float32)
    crop_info = get_crop_info(c, npad, pos_x, pos_y)  # returns y,x,h,w

    # pad crop info with offset value 
    crop_info_padded = tf.map_fn(lambda x: tf.concat( [tf.stack([x[0]]*SCALE_NUM), tf.expand_dims(tf.transpose(x[1]),axis=1)], axis=1), (crop_info, offset), dtype=tf.float32)
    crop_info_padded = tf.reshape(crop_info_padded, (batch_sz*SCALE_NUM, 5))  # B * SCALE_NUM, [y,x,h,w,sz_dst]

    crops = tf.map_fn(lambda crp_i: tf_crop_and_resize(im, crp_i, sz_dst), crop_info_padded)
    crops = tf.reshape(crops, (batch_sz, SCALE_NUM, sz_dst, sz_dst, tf.shape(im)[-1]))

    return crops

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
frame_padded_z, npad_z = pad_frame_x(image, frame_sz, pos_x_ph, pos_y_ph, z_sz_ph, avg_chan)
frame_padded_z = tf.cast(frame_padded_z, tf.float32)
# 2) crop to context window
z_crops = extract_crops_z(frame_padded_z, npad_z, pos_x_ph, pos_y_ph, z_sz_ph, config["exemplar_sz"])

'''our x (search) image'''
# 1) padding
frame_padded_x, npad_x = pad_frame_x(image, frame_sz, pos_x_ph, pos_y_ph, tf.reduce_max(x_sz_list_ph, axis=1), avg_chan)
frame_padded_x = tf.cast(frame_padded_x, tf.float32)
# 2) crop
x_crops = tf_extract_crops_x(frame_padded_x, npad_x, pos_x_ph, pos_y_ph, x_sz_list_ph, config["search_sz"])  # B,N,W,W,C
# x_crops_new = tf_extract_crops_x(frame_padded_x, npad_x, pos_x_ph, pos_y_ph, x_sz_list_ph, config["search_sz"])  # B,N,W,W,C
# x_crops_t = tf.transpose(x_crops, perm=[2,3,4,1,0])  # W,W,C,B,N
x_crops_reshaped = tf.reshape(x_crops, (tf.shape(x_crops)[0]*SCALE_NUM, config["search_sz"], config["search_sz"], frame_sz[2])) # W,W,C,N*B
# x_crops_reshaped_t = tf.transpose(x_crops_reshaped, perm=[3,0,1,2])  # N*B,W,W,C

net_path = "./pretrained/2016-08-17.net.mat"
template_z, template_x, p_names_list, p_val_list = _create_siamese(net_path, x_crops_reshaped, z_crops) # N*B,Wo,Wo,Co
_,tx_width,tx_width,tx_cn = template_x.shape.as_list()
templates_x = tf.reshape(template_x, (tf.shape(x_crops_reshaped)[0] / SCALE_NUM, SCALE_NUM, tx_width, tx_width, tx_cn))

template_z = tf.expand_dims(template_z, axis=1)
templates_z = tf.tile(template_z, [1,SCALE_NUM,1,1,1]) # one for each x_crop scale (3)

score_map = _get_templates_score_map(templates_z, templates_x)
score_map = bn_adjust_score_map(score_map, p_names_list, p_val_list)

# _, _, sm_width, sm_height = score_map.shape.as_list()  # B, C, 33, 33
# final_score_map_width = hp["response_up"] * (sm_width - 1) + 1
# # score_map = tf.transpose(score_map, perm=[0,2,3,1])  # move from B,C,W,W to B,H,W,C for tf resize image
# # scores_up = tf.image.resize_images(score_map, [final_score_map_width, final_score_map_width],
# #     method=tf.image.ResizeMethod.BICUBIC, align_corners=True)

def score_map_test(sess, frame, pos_x, pos_y, scaled_search_area, z_sz):
    x_feat, z_feat = sess.run([templates_x, templates_z], feed_dict={image:frame, pos_x_ph: pos_x, pos_y_ph: pos_y, 
            x_sz_list_ph: scaled_search_area, z_sz_ph: z_sz})

    sc_map = sess.run(score_map, feed_dict={net_z_ph: np.array([z_feat,z_feat]), net_x_ph: np.array([x_feat,x_feat])})
    sc_map_old = sess.run(score_map_old, feed_dict={templates_z: z_feat, templates_x: x_feat})

    sc_map_old = np.squeeze(sc_map_old)
    gt = np.amax(sc_map_old, axis=(1,2))
    output = np.amax(sc_map[0], axis=(1,2))

    print('gt', gt, 'output', output)

def viz_crops(crp):
    crp = crp.astype(np.uint8)
    N, crp_sz, H, W, cn = crp.shape
    canvas = np.zeros((N*H, crp_sz * W, cn), dtype=np.uint8)
    for n in xrange(N):
        r_start = n * H
        r_end = (n+1) * H 
        for c in xrange(crp_sz):
            c_start = c * W
            c_end = (c+1) * W
            canvas[r_start:r_end,c_start:c_end] = crp[n][c]

    cv2.imshow("crops", canvas)
    cv2.waitKey(0)
    return canvas

def get_context_size(config, target_w, target_h):
    context = config["context"]*(target_w+target_h)
    z_sz = np.sqrt((target_w+context)*(target_h+context))
    x_sz = float(config["search_sz"]) / config["exemplar_sz"] * z_sz  # scale the search image according to its final exemplar context size

    return x_sz, z_sz

if __name__ == '__main__':
    from utils import get_bbox_from_image, natural_sort
    from src.region_to_bbox import region_to_bbox

    net_path = "./pretrained/2016-08-17.net.mat"
    config_path = './parameters/design_ori.json'
    image_folder = "./data/interoll2"
    start_idx = 0

    '''Get images'''
    frame_name_list = [os.path.join(image_folder, f) for f in natural_sort(os.listdir(image_folder)) if f.endswith(".jpg")]
    frame_name_list = frame_name_list[start_idx:]
    num_frames = np.size(frame_name_list)

    # manually label first frame
    # first_frame_bbox = get_bbox_from_image(cv2.imread(frame_name_list[0]))
    frame_bbox1 = np.array([123, 285, 86, 69])
    frame_bbox2 = np.array([75, 109, 53, 33])
    frame_bbox3 = np.array([295, 90, 77, 59])
    frame_bbox4 = np.array([263, 76, 48, 32])
    frame_bbox5 = np.array([573, 307, 60, 84])
    frame_bbox6 = np.array([526, 311, 45, 53])

    pos_x, pos_y, target_w, target_h = region_to_bbox(frame_bbox1)
    pos_x2, pos_y2, target_w2, target_h2 = region_to_bbox(frame_bbox2)
    pos_x3, pos_y3, target_w3, target_h3 = region_to_bbox(frame_bbox3)
    pos_x4, pos_y4, target_w4, target_h4 = region_to_bbox(frame_bbox4)
    pos_x5, pos_y5, target_w5, target_h5 = region_to_bbox(frame_bbox5)
    pos_x6, pos_y6, target_w6, target_h6 = region_to_bbox(frame_bbox6)

    x_sz, z_sz = get_context_size(config, target_w, target_h)
    x_sz2, z_sz2 = get_context_size(config, target_w2, target_h2)
    x_sz3, z_sz3 = get_context_size(config, target_w3, target_h3)
    x_sz4, z_sz4 = get_context_size(config, target_w4, target_h4)
    x_sz5, z_sz5 = get_context_size(config, target_w5, target_h5)
    x_sz6, z_sz6 = get_context_size(config, target_w6, target_h6)

    scale_factors = hp["scale_step"]**np.linspace(-np.ceil(SCALE_NUM/2), np.ceil(SCALE_NUM/2), SCALE_NUM)

#     # sess
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())


    frame = cv2.imread(frame_name_list[start_idx])

    # score_map_test(sess, frame, pos_x, pos_y, scaled_search_area, z_sz)

    start_t = time.time()
    zzc = sess.run(z_crops, feed_dict={image:frame, pos_x_ph: [pos_x,pos_x2], pos_y_ph: [pos_y,pos_y2], z_sz_ph: [z_sz,z_sz2]})
    print("Z crop time: %.3f seconds"%(time.time() - start_t))  # 0.005s

    # pos_x_data = [pos_x]#,pos_x2]
    # pos_y_data = [pos_y]#,pos_y2]
    # x_sz_data = [scaled_search_area]#,scaled_search_area2]
    # z_sz_data = [z_sz]#,z_sz2]
    pos_x_data = [pos_x,pos_x2,pos_x3,pos_x4,pos_x5,pos_x6]
    pos_y_data = [pos_y,pos_y2,pos_y3,pos_y4,pos_y5,pos_y6]
    z_sz_data = [z_sz,z_sz2,z_sz3,z_sz4,z_sz5,z_sz6]
    x_sz_data = [x_sz,x_sz2,x_sz3,x_sz4,x_sz5,x_sz6]
    scaled_search_areas = np.array([x*scale_factors for x in x_sz_data])

    z_crp = sess.run(z_crops, feed_dict={image:frame, pos_x_ph: pos_x_data, pos_y_ph: pos_y_data, z_sz_ph: z_sz_data})
    z_feat = sess.run(templates_z, feed_dict={image:frame, pos_x_ph: pos_x_data, pos_y_ph: pos_y_data, z_sz_ph: z_sz_data})

    gt_x_crp = np.array([ 254.,  255.,  254.]) # np.amax(x_crp[0], axis=(1,2,3))
    x_crp = sess.run(x_crops, feed_dict={image:frame, pos_x_ph: pos_x_data, pos_y_ph: pos_y_data, x_sz_list_ph: scaled_search_areas})
    x_feat = sess.run(templates_x, feed_dict={x_crops: x_crp, image:frame})
    np.amax(x_crp[0], axis=(1,2,3))
    
    # np.amax(x_feat[0], axis=(1,2,3))
    #gt_x_feat = np.array([ 3.07099438,  3.12097216,  3.11816764], dtype=float32)

    sc_map = sess.run(score_map, feed_dict={templates_x: x_feat, templates_z: z_feat})
    # np.amax(sc_map[0], axis=(1,2))

    # x_crp_new = sess.run(x_crops_new, feed_dict={image:frame, pos_x_ph: pos_x_data, pos_y_ph: pos_y_data, x_sz_list_ph: scaled_search_areas})
    # viz_crops(x_crp_new)


    # n_runs = 5
    # start_t = time.time()
    # for i in xrange(n_runs):  # 1,2,3 runs: 0.03s, 0.051-0.055s, 0.076-0.08s
    #     start_t_inner = time.time()
    #     x_crp = sess.run(x_crops, feed_dict={image:frame, pos_x_ph: pos_x_data, pos_y_ph: pos_y_data, 
    #                 x_sz_list_ph: scaled_search_areas})
    #     print("Old Crop time: %.3f seconds"%(time.time() - start_t_inner))  # 0.016s 
    #     start_t_inner = time.time()
    #     x_crp = sess.run(x_crops_new, feed_dict={image:frame, pos_x_ph: pos_x_data, pos_y_ph: pos_y_data, 
    #                 x_sz_list_ph: scaled_search_areas})
    #     print("New Crop time: %.3f seconds"%(time.time() - start_t_inner))  # 0.016s 
    #     # start_t_inner = time.time()
    #     # x_feat = sess.run(templates_x, feed_dict={x_crops: x_crp}) 
    #     # print("X feat time: %.3f seconds"%(time.time() - start_t_inner))  # 0.012s
    # # print("Total time: %.3f seconds"%(time.time() - start_t))
    
