from __future__ import division
import numpy as np
import tensorflow as tf
from PIL import Image
import functools
from scipy.misc import imresize


def resize_images(images, size, resample):
    '''Alternative to tf.image.resize_images that uses PIL.'''
    fn = functools.partial(_resize_images, size=size, resample=resample)
    return tf.py_func(fn, [images], images.dtype)


def _resize_images(x, size, resample):
    # TODO: Use tf.map_fn?
    if len(x.shape) == 3:
        return _resize_image(x, size, resample)
    assert len(x.shape) == 4
    y = []
    for i in range(x.shape[0]):
        y.append(_resize_image(x[i]))
    y = np.stack(y, axis=0)
    return y


def _resize_image(x, size, resample):
    assert len(x.shape) == 3
    y = []
    for j in range(x.shape[2]):
        f = x[:, :, j]
        f = Image.fromarray(f)
        f = f.resize((size[1], size[0]), resample=resample)
        f = np.array(f)
        y.append(f)
    return np.stack(y, axis=2)


def pad_frame(im, frame_sz, pos_x, pos_y, patch_sz, avg_chan):
    c = patch_sz / 2
    xleft_pad = tf.maximum(0, -tf.cast(tf.round(pos_x - c), tf.int32))
    ytop_pad = tf.maximum(0, -tf.cast(tf.round(pos_y - c), tf.int32))
    xright_pad = tf.maximum(0, tf.cast(tf.round(pos_x + c), tf.int32) - frame_sz[1])
    ybottom_pad = tf.maximum(0, tf.cast(tf.round(pos_y + c), tf.int32) - frame_sz[0])
    npad = tf.reduce_max([xleft_pad, ytop_pad, xright_pad, ybottom_pad])
    paddings = [[npad, npad], [npad, npad], [0, 0]]
    im_padded = im
    if avg_chan is not None:
        im_padded = im_padded - avg_chan
    im_padded = tf.pad(im_padded, paddings, mode='CONSTANT')
    if avg_chan is not None:
        im_padded = im_padded + avg_chan
    return im_padded, npad

# Can't manage to use tf.crop_and_resize, which would be ideal!
# im:  A 4-D tensor of shape [batch, image_height, image_width, depth]
# boxes: the i-th row of the tensor specifies the coordinates of a box in the box_ind[i] image and is
# specified in normalized coordinates [y1, x1, y2, x2]
# box_ind: specify image to which each box refers to
# crop = tf.image.crop_and_resize(im, boxes, box_ind, sz_dst)

def py_extract_crops_x(im, npad, pos_x, pos_y, sz_src_list, sz_dst):
    max_sz = np.amax(sz_src_list)
    c = max_sz / 2
    npad = int(npad)
    tl_x = np.amin(npad + np.round(pos_x - c).astype(np.int32))
    tl_y = np.amin(npad + np.round(pos_y - c).astype(np.int32))

    width = np.amax(np.round(pos_x + c) - np.round(pos_x - c))
    height = np.amax(np.round(pos_y + c) - np.round(pos_y - c))

    search_area = im[tl_y:tl_y + height, tl_x:tl_x + width]

    # H,W,C = im.shape
    im_channels = im.shape[-1]
    num_scales = sz_src_list.shape[1]
    crops_mat_sz = [len(pos_x), num_scales, sz_dst, sz_dst, im_channels]
    crops = np.zeros(crops_mat_sz, dtype=np.float32)

    for i,px in enumerate(pos_x):
        for j,sz in enumerate(sz_src_list[i]):
            offset = int(max_sz - sz) / 2
            sz_rounded = np.round(sz).astype(np.int32)
            crop_search_a = search_area[offset:offset+sz_rounded, offset:offset+sz_rounded]
            crops[i][j] = imresize(crop_search_a, (sz_dst, sz_dst), interp='bilinear')

    # crops[-1] = imresize(search_area, (sz_dst, sz_dst), interp='bilinear')

    return crops


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

# def extract_crops_x(im, npad, pos_x, pos_y, sz_src_list, sz_dst, num_scales):
#     batch_sz = tf.shape(pos_x)[0]
#     max_sz = tf.reduce_max(sz_src_list, axis=1)
#     c = max_sz / 2
#     offset = tf.cast((tf.expand_dims(max_sz, axis=1) - sz_src_list) / 2, tf.float32)
#     crop_info = get_crop_info(c, npad, pos_x, pos_y)  # returns y,x,h,w

#     # pad crop info with offset value 
#     crop_info_padded = tf.map_fn(lambda x: tf.concat( [tf.stack([x[0]]*num_scales), tf.expand_dims(tf.transpose(x[1]),axis=1)], axis=1), (crop_info, offset), dtype=tf.float32)
#     crop_info_padded = tf.reshape(crop_info_padded, (batch_sz*num_scales, 5))  # B * num_scales, [y,x,h,w,sz_dst]

#     crops = tf.map_fn(lambda crp_i: tf_crop_and_resize(im, crp_i, sz_dst), crop_info_padded)
#     crops = tf.reshape(crops, (batch_sz, num_scales, sz_dst, sz_dst, tf.shape(im)[-1]))

#     return crops

def extract_crops_x(im, npad, pos_x, pos_y, sz_src_list, sz_dst, num_scales):
    batch_sz = tf.shape(pos_x)[0]
    im_h = tf.cast(tf.shape(im)[0], tf.float32)
    im_w = tf.cast(tf.shape(im)[1], tf.float32)
    im_cn = tf.shape(im)[2]

    max_sz = tf.reduce_max(sz_src_list, axis=1)
    c = max_sz / 2
    offset = tf.cast((tf.expand_dims(max_sz, axis=1) - sz_src_list) / 2, tf.float32)
    crop_info = get_crop_info(c, npad, pos_x, pos_y)  # returns y,x,h,w

    # pad crop info with offset value 
    crop_info_padded = tf.map_fn(lambda x: tf.concat( [tf.stack([x[0]]*num_scales), tf.expand_dims(tf.transpose(x[1]),axis=1)], axis=1), (crop_info, offset), dtype=tf.float32)
    crop_info_padded = tf.reshape(crop_info_padded, (batch_sz*num_scales, 5))  # B * num_scales, [y,x,h,w,sz_dst]

    crop_offset = crop_info_padded[:,-1]
    crop_inf_y1 = crop_info_padded[:,0] + crop_offset
    crop_inf_x1 = crop_info_padded[:,1] + crop_offset
    crop_inf_y2 = (crop_info_padded[:,2] + crop_inf_y1 - 2*crop_offset)
    crop_inf_x2 = (crop_info_padded[:,3] + crop_inf_x1 - 2*crop_offset)
    bboxes = tf.stack([crop_inf_y1 / im_h, crop_inf_x1 / im_w, crop_inf_y2 / im_h, crop_inf_x2 / im_w], axis=1)

    # box_ind = tf.range(0,batch_sz*num_scales)
    box_ind = tf.zeros([batch_sz*num_scales], tf.int32) # zero-indexed: all refer to the same image, since there is only one image i.e. batch size 1

    crops = tf.image.crop_and_resize(tf.expand_dims(im, axis=0), bboxes, box_ind, [sz_dst,sz_dst])  # 
    crops = tf.reshape(crops, (batch_sz, num_scales, sz_dst, sz_dst, im_cn))

    return crops