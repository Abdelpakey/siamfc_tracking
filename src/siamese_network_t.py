import tensorflow as tf
import numpy as np
np.core.arrayprint._line_width = 120
import os
import os.path as osp
import cv2

from utils import natural_sort

def weight_variable(shape, name='w'):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape, name='b'):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def max_pool(x, sz=2, stride=2):
    # stride: batch, height, width, channel
    return tf.nn.max_pool(x, ksize=[1, sz, sz, 1],
                        strides=[1, stride, stride, 1], padding='VALID')

def bn(x, phase_train=False):
    bn_x = tf.contrib.layers.batch_norm(x, 
                                      center=True, scale=True, 
                                      is_training=phase_train,
                                      scope='bn')
    return bn_x

def _get_templates_score_map(net_z, net_x, trainable=True): #, params_names_list, params_values_list):
    B, N, Hz, Wz, C =  net_z.shape.as_list()
    B, N, Hx, Wx, C =  net_x.shape.as_list()
    net_z_shape = tf.shape(net_z)
    net_x_shape = tf.shape(net_x)

    net_z = tf.transpose(net_z, perm=[2,3,1,4,0])
    net_x = tf.transpose(net_x, perm=[2,3,1,4,0])

    net_z = tf.reshape(net_z, (Hz, Wz, N * C * net_z_shape[0], 1))  # (17, 17, 3, 32) to (17, 17, 96, 1)
    net_x = tf.reshape(net_x, (1, Hx, Wx, N * C * net_x_shape[0]))  # (49, 49, 3, 32) to (1, 49, 49, 96)
    net_final = tf.nn.depthwise_conv2d(net_x, net_z, strides=[1,1,1,1], padding='VALID')  # (B, 33, 33, N*C)
    net_final_width = net_final.shape.as_list()[1]

    nf = tf.reshape(net_final, (1, net_final_width, net_final_width, N*C, net_z_shape[0]))
    nf = tf.transpose(nf, perm=[4,0,1,2,3])

    nf = tf.concat(tf.split(nf, N, axis=4), axis=1)

    nf = tf.reduce_sum(nf, axis=4)

    nf = bn(nf, trainable)

    return tf.squeeze(nf, axis=1)

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

def create_log_labels(label_size, radius_thresh):
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


class Network(object):

    def __init__(self):
        pass

    def build(self, input_x, reuse=False, trainable=True):

        with tf.variable_scope('conv1', reuse=reuse):
            
            kernel_sz = 11
            n_filters = 96
            stride = 2

            W = weight_variable([kernel_sz, kernel_sz, 3, n_filters], name="w")
            b = bias_variable([n_filters], name="b")

            x = tf.nn.conv2d(input_x, W, strides=[1, stride, stride, 1], padding='VALID') + b

            x = bn(x,trainable)
            x = tf.nn.relu(x)
            x = max_pool(x, sz=3, stride=2)

        with tf.variable_scope('conv2', reuse=reuse):

            kernel_sz = 5
            n_filters2 = 256
            stride = 1

            W = weight_variable([kernel_sz, kernel_sz, n_filters / 2, n_filters2], name="w")
            b = bias_variable([n_filters2], name="b")

            X0, X1 = tf.split(x, 2, 3)
            W0, W1 = tf.split(W, 2, 3)
            h0 = tf.nn.conv2d(X0, W0, strides=[1, stride, stride, 1], padding='VALID')
            h1 = tf.nn.conv2d(X1, W1, strides=[1, stride, stride, 1], padding='VALID')
            x = tf.concat([h0, h1], 3) + b

            x = bn(x, trainable)
            x = tf.nn.relu(x)

            x = max_pool(x, sz=3, stride=2)

        with tf.variable_scope('conv3', reuse=reuse):
            
            kernel_sz = 3
            n_filters3 = 384
            stride = 1

            W = weight_variable([kernel_sz, kernel_sz, n_filters2, n_filters3], name="w")
            b = bias_variable([n_filters3], name="b")

            x = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID') + b

            x = bn(x, trainable)

            x = tf.nn.relu(x)

        with tf.variable_scope('conv4', reuse=reuse):

            kernel_sz = 3
            n_filters4 = 384
            stride = 1

            W = weight_variable([kernel_sz, kernel_sz, n_filters3 / 2, n_filters4], name="w")
            b = bias_variable([n_filters4], name="b")

            X0, X1 = tf.split(x, 2, 3)
            W0, W1 = tf.split(W, 2, 3)
            h0 = tf.nn.conv2d(X0, W0, strides=[1, stride, stride, 1], padding='VALID')
            h1 = tf.nn.conv2d(X1, W1, strides=[1, stride, stride, 1], padding='VALID')
            x = tf.concat([h0, h1], 3) + b

            x = bn(x, trainable)

            x = tf.nn.relu(x)

        with tf.variable_scope('conv5', reuse=reuse):

            kernel_sz = 3
            n_filters5 = 256
            stride = 1

            W = weight_variable([kernel_sz, kernel_sz, n_filters4 / 2, n_filters5], name="w")
            b = bias_variable([n_filters5], name="b")

            X0, X1 = tf.split(x, 2, 3)
            W0, W1 = tf.split(W, 2, 3)
            h0 = tf.nn.conv2d(X0, W0, strides=[1, stride, stride, 1], padding='VALID')
            h1 = tf.nn.conv2d(X1, W1, strides=[1, stride, stride, 1], padding='VALID')
            x = tf.concat([h0, h1], 3) + b

            # x = bn(x,
            # x = tf.nn.relu(x)

        return x

def get_sample_data(root_dir):
    sample_dir = osp.join(root_dir, 'samples')
    # sample_dir = osp.join(root_train_dir, 'ILSVRC2015_VID_train_0000')
    vid_list = [osp.join(sample_dir, f) for f in natural_sort(os.listdir(sample_dir))]
    batch_sz = 2
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

    # crop_x_batch = np.expand_dims(crop_x_batch, axis=1)
    # crop_z_batch = np.expand_dims(crop_z_batch, axis=1)


    # crop_x_sample = crop_x_batch[1:2]
    # crop_z_sample = crop_z_batch[1:2]

    crop_x_batch /= 255
    crop_z_batch /= 255
    return crop_x_batch, crop_z_batch


def sigmoid(x): 
    return 1. / (1+np.exp(-x))

if __name__ == '__main__':
    input_x = tf.placeholder(tf.float32, shape=[None,255,255,3])
    input_z = tf.placeholder(tf.float32, shape=[None,127,127,3])
    x = Network()
    net_x = x.build(input_x, reuse=False)
    net_z = x.build(input_z, reuse=True)
    net_x = tf.expand_dims(net_x, axis=1)
    net_z = tf.expand_dims(net_z, axis=1)
    # net_z = tf.tile(net_z, [1,1,1,1,1]) # one for each x_crop scale (3)

    sc_map = _get_templates_score_map(net_z, net_x)
    _,h,w = sc_map.shape.as_list()
    rPos = 16
    tot_stride = 8
    eltwise_labels, instance_weight = create_log_labels((h,w), rPos/tot_stride)
    log_loss =  tf.reduce_sum( tf.log( 1+tf.exp(-eltwise_labels*sc_map)) * instance_weight, axis=[1,2])
    loss = tf.reduce_mean(log_loss)

    # learning params
    learning_rate = 1e-3

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)  # adam optimizer

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())

    # coord = tf.train.Coordinator()  
    # threads = tf.train.start_queue_runners(coord=coord)

    cx, cz = get_sample_data('./ilsvrc15_curation')

    epochs = 10
    for epoch in range(epochs):
        _, out_loss = sess.run([train_op, loss], feed_dict={input_x: cx, input_z: cz})
        print("Epoch %d) Loss %.3f"%(epoch, out_loss))

