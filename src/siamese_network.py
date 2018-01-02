import tensorflow as tf
import numpy as np
import scipy.io
import sys
import os
import os.path as osp
sys.path.append('../')

import time

from src.crops import pad_frame, get_crop_info, tf_crop_and_resize, extract_crops_z, extract_crops_x

class SiameseNetwork(object):
    def __init__(self, sess, cfg, trainable=True):
        # assert(cfg.scale_num == 3, "Scale num must be 3, other scale nums not supported")

        self.trainable = trainable
        self.cfg = cfg
        self.is_built = False
        self.sess = sess

        # assert(self.cfg.)

        self._pos_x_ph = tf.placeholder(tf.float32)
        self._pos_y_ph = tf.placeholder(tf.float32)
        self._z_sz_ph = tf.placeholder(tf.float32)
        # self._x_sz0_ph = tf.placeholder(tf.float32)
        # self._x_sz1_ph = tf.placeholder(tf.float32)
        # self._x_sz2_ph = tf.placeholder(tf.float32)
        self._x_sz_list_ph = tf.placeholder(tf.float32, shape=[None,None])

        self._filename = tf.placeholder(tf.string, [], name='filename')
        image_file = tf.read_file(self._filename)
        # Decode the image as a JPEG file, this will turn it into a Tensor
        image = tf.image.decode_jpeg(image_file)
        self._image = 255.0 * tf.image.convert_image_dtype(image, tf.float32)

        # generated
        self.score_map = None
        self.score_map_up = None


        self.x_crops, self.z_crops = self.setup()
        self.net_x = None 
        self.net_z = None

        self.scale_factors = self.cfg.scale_step**np.linspace(-np.ceil(self.cfg.scale_num/2), np.ceil(self.cfg.scale_num/2), self.cfg.scale_num)
        
        # cosine window to penalize large displacements    
        self.final_score_sz = self.cfg.response_up * (self.cfg.score_sz - 1) + 1
        hann_1d = np.expand_dims(np.hanning(self.final_score_sz), axis=0)
        penalty = np.transpose(hann_1d) * hann_1d
        self.penalty = penalty / np.sum(penalty)

        # output variables
        self.pos_x = None
        self.pos_y = None
        self.target_w = None
        self.target_h = None
        self.z_sz = None
        self.x_sz = None
        self.output_z = None
        self.output_x = None

        self.history = {"score": [], "target_w": [], "target_h": []}

    def setup(self):

        image = self._image
        frame_sz = tf.shape(image)
        # used to pad the crops
        if self.cfg.pad_with_image_mean:
            avg_chan = tf.reduce_mean(image, axis=(0,1), name='avg_chan')
        else:
            avg_chan = None
        # pad with if necessary
        frame_padded_z, npad_z = pad_frame(image, frame_sz, self._pos_x_ph, self._pos_y_ph, self._z_sz_ph, avg_chan)
        frame_padded_z = tf.cast(frame_padded_z, tf.float32)
        # extract tensor of z_crops
        z_crops = extract_crops_z(frame_padded_z, npad_z, self._pos_x_ph, self._pos_y_ph, self._z_sz_ph, self.cfg.exemplar_sz)
        # frame_padded_x, npad_x = pad_frame(image, frame_sz, self._pos_x_ph, self._pos_y_ph, self._x_sz2_ph, avg_chan)
        frame_padded_x, npad_x = pad_frame(image, frame_sz, self._pos_x_ph, self._pos_y_ph, tf.reduce_max(self._x_sz_list_ph), avg_chan)
        frame_padded_x = tf.cast(frame_padded_x, tf.float32)
        # extract tensor of x_crops (num_scales)
        # x_crops = tf_extract_crops_x(frame_padded_x, npad_x, self._pos_x_ph, self._pos_y_ph, self._x_sz_list_ph, self.cfg.search_sz)
        x_crops = extract_crops_x(frame_padded_x, npad_x, self._pos_x_ph, self._pos_y_ph, self._x_sz_list_ph, self.cfg.search_sz, self.cfg.scale_num)

        # self.x_crops = x_crops
        # self.z_crops = z_crops
        return x_crops, z_crops

    def reset(self):
        self.pos_x = None
        self.pos_y = None
        self.target_w = None
        self.target_h = None
        self.z_sz = None
        self.x_sz = None
        self.output_z = None
        self.output_x = None

    def build(self, params_names_list, params_values_list, input_x, input_z):
        from src.convolutional import set_convolutional

        _conv_stride = self.cfg.net.conv_stride
        _filtergroup_yn = self.cfg.net.filtergroup_yn
        _bnorm_yn = self.cfg.net.bnorm_yn
        _relu_yn = self.cfg.net.relu_yn
        _pool_stride = self.cfg.net.pool_stride
        _pool_sz = self.cfg.net.pool_sz

        assert len(_conv_stride) == len(_filtergroup_yn) == len(_bnorm_yn) == len(_relu_yn) == len(_pool_stride), ('These arrays of flags should have same length')
        assert all(_conv_stride) >= True, ('The number of conv layers is assumed to define the depth of the network')

        num_layers = len(_conv_stride)

        batch_sz = tf.shape(input_x)[0]

        self.net_x = input_x
        self.net_z = input_z

        self.net_x = tf.reshape(self.net_x, (batch_sz*self.cfg.scale_num, self.cfg.search_sz, self.cfg.search_sz, tf.shape(self._image)[-1]))

        for i in xrange(num_layers):
            print '> Layer '+str(i+1)
            # conv
            conv_W_name = _find_params('conv'+str(i+1)+'f', params_names_list)[0]
            conv_b_name = _find_params('conv'+str(i+1)+'b', params_names_list)[0]
            print('\t\tCONV: setting '+conv_W_name+' '+conv_b_name)
            print('\t\tCONV: stride '+str(_conv_stride[i])+', filter-group '+str(_filtergroup_yn[i]))
            conv_W = params_values_list[params_names_list.index(conv_W_name)]
            conv_b = params_values_list[params_names_list.index(conv_b_name)]
            # batchnorm
            if _bnorm_yn[i]:
                bn_beta_name = _find_params('bn'+str(i+1)+'b', params_names_list)[0]
                bn_gamma_name = _find_params('bn'+str(i+1)+'m', params_names_list)[0]
                bn_moments_name = _find_params('bn'+str(i+1)+'x', params_names_list)[0]
                print '\t\tBNORM: setting '+bn_beta_name+' '+bn_gamma_name+' '+bn_moments_name
                bn_beta = params_values_list[params_names_list.index(bn_beta_name)]
                bn_gamma = params_values_list[params_names_list.index(bn_gamma_name)]
                bn_moments = params_values_list[params_names_list.index(bn_moments_name)]
                bn_moving_mean = bn_moments[:,0]
                bn_moving_variance = bn_moments[:,1]**2 # saved as std in matconvnet
            else:
                bn_beta = bn_gamma = bn_moving_mean = bn_moving_variance = []
            
            # set up conv "block" with bnorm and activation 
            print(self.net_x.shape)
            print(self.net_z.shape)
            self.net_x = set_convolutional(self.net_x, conv_W, np.swapaxes(conv_b,0,1), _conv_stride[i], \
                                bn_beta, bn_gamma, bn_moving_mean, bn_moving_variance, \
                                filtergroup=_filtergroup_yn[i], batchnorm=_bnorm_yn[i], activation=_relu_yn[i], \
                                scope='conv'+str(i+1), reuse=False, trainable=self.trainable)
            
            # notice reuse=True for Siamese parameters sharing
            self.net_z = set_convolutional(self.net_z, conv_W, np.swapaxes(conv_b,0,1), _conv_stride[i], \
                                bn_beta, bn_gamma, bn_moving_mean, bn_moving_variance, \
                                filtergroup=_filtergroup_yn[i], batchnorm=_bnorm_yn[i], activation=_relu_yn[i], \
                                scope='conv'+str(i+1), reuse=True, trainable=self.trainable)    

            # add max pool if required
            if _pool_stride[i]>0:
                print '\t\tMAX-POOL: size '+str(_pool_sz)+ ' and stride '+str(_pool_stride[i])
                self.net_x = tf.nn.max_pool(self.net_x, [1,_pool_sz,_pool_sz,1], strides=[1,_pool_stride[i],_pool_stride[i],1], padding='VALID', name='pool'+str(i+1))
                self.net_z = tf.nn.max_pool(self.net_z, [1,_pool_sz,_pool_sz,1], strides=[1,_pool_stride[i],_pool_stride[i],1], padding='VALID', name='pool'+str(i+1))

        # use crops as input of (MatConvnet imported) pre-trained fully-convolutional Siamese net
        _,nx_width,nx_width,tx_cn = self.net_x.shape.as_list()
        self.net_x = tf.reshape(self.net_x, (batch_sz, self.cfg.scale_num, nx_width, nx_width, tx_cn))

        self.net_z = tf.expand_dims(self.net_z, axis=1)
        self.net_z = tf.tile(self.net_z, [1,self.cfg.scale_num,1,1,1]) # one for each x_crop scale (3)

        # compare templates via cross-correlation
        self.score_map = _get_templates_score_map(self.net_z, self.net_x)

        if self.cfg.net.bnorm_adjust:
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
            self.score_map = tf.layers.batch_normalization(self.score_map, beta_initializer=tf.constant_initializer(bn_beta),
                                                    gamma_initializer=tf.constant_initializer(bn_gamma),
                                                    moving_mean_initializer=tf.constant_initializer(bn_moving_mean),
                                                    moving_variance_initializer=tf.constant_initializer(bn_moving_variance),
                                                    training=self.trainable, trainable=self.trainable)

        self.score_map = tf.transpose(self.score_map, perm=[0,2,3,1])  # reshape from B,C,W,W to B,W,W,C for tf resize image
        _, sm_width, sm_width, _ = self.score_map.shape.as_list()  # B,W,W,C
        score_map_up_width = self.cfg.response_up * (sm_width - 1) + 1  # width of upsampled score map 

        self.score_map_up = tf.image.resize_images(self.score_map, [score_map_up_width, score_map_up_width],
                method=tf.image.ResizeMethod.BICUBIC, align_corners=True)
        # self.score_map_up = tf.transpose(self.score_map_up, perm=[0,3,1,2])  # reshape back to B,C,W,W

        # tf.global_variables_initializer().run()

        self.is_built = True


    def load_mat_model(self, net_path):
        pnl, pvl = _import_from_matconvnet(net_path)
        self.build(pnl, pvl, self.x_crops, self.z_crops)

    def forward_z(self, img, pos_x, pos_y, target_w, target_h):
        assert(self.is_built, "Network has not been built!")

        pos_x = np.array(pos_x, dtype=np.float32)
        pos_y = np.array(pos_y, dtype=np.float32)
        target_w = np.array(target_w, dtype=np.float32)
        target_h = np.array(target_h, dtype=np.float32)

        context = self.cfg.context*(target_w+target_h)
        z_sz = np.sqrt((target_w+context)*(target_h+context))
        x_sz = float(self.cfg.search_sz) / self.cfg.exemplar_sz * z_sz

        feed_data = {
            self._pos_x_ph: pos_x,
            self._pos_y_ph: pos_y,
            self._z_sz_ph: z_sz,
        }
        if type(img) == str:
            feed_data[self._filename] = img
        else:
            feed_data[self._image] = img

        image_, self.output_z = self.sess.run([self._image, self.net_z], feed_dict = feed_data)

        # store outputs for next forward pass
        self.target_w = target_w
        self.target_h = target_h
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.z_sz = z_sz
        self.x_sz = x_sz

        return image_

    def forward(self, img):
        assert(self.is_built, "Network has not been built!")

        hp = self.cfg

        bboxes = []

        self.output_x = []

        start_t = time.time()
        start_t_all = time.time()

        feed_data = {
            self._pos_x_ph: self.pos_x,
            self._pos_y_ph: self.pos_y,
            self._x_sz_list_ph: np.array([x*self.scale_factors for x in self.x_sz]),
            self._image: img   
            # self.net_z: np.squeeze(self.output_z[i]),
        }
        x_feat = self.sess.run(self.net_x, feed_dict=feed_data)
        self.output_x = x_feat

        print("Crop time: %.3f seconds"%(time.time() - start_t))

        start_t = time.time()

        all_scores_ = self.sess.run(self.score_map_up, feed_dict={self.net_z: self.output_z, self.net_x: self.output_x})
        # print("all_scores_", all_scores_.shape)

        print("Inference (score map) time: %.3f seconds"%(time.time() - start_t))

        start_t = time.time()

        new_z_sz = self.z_sz
        final_scores = []
        for i, scores_ in enumerate(all_scores_):
            # print("scores_", scores_.shape)

            target_w = self.target_w[i]
            target_h = self.target_h[i]

            scaled_exemplar = self.z_sz[i] * self.scale_factors
            scaled_search_area = self.x_sz[i] * self.scale_factors
            scaled_target_w = target_w * self.scale_factors
            scaled_target_h = target_h * self.scale_factors

            # scores_ = np.squeeze(scores_)

            # penalize change of scale i.e. any scale that is not 1
            for sc_num in xrange(self.cfg.scale_num):
                if scaled_search_area[sc_num] != 1:
                    scores_[sc_num,:,:] *= hp.scale_penalty

            # find scale with highest peak (after penalty)
            max_score = np.amax(scores_, axis=(0,1))
            new_scale_id = np.argmax(max_score)
            # update scaled sizes
            new_scale = scaled_search_area[new_scale_id]
            if new_scale != 1:
                self.x_sz[i] = (1-hp.scale_lr)*self.x_sz[i] + hp.scale_lr*new_scale

                self.target_w[i] = (1-hp.scale_lr)*target_w + hp.scale_lr*scaled_target_w[new_scale_id]
                self.target_h[i] = (1-hp.scale_lr)*target_h + hp.scale_lr*scaled_target_h[new_scale_id]
                # print(self.x_sz[i],self.z_sz[i],self.target_w[i],self.target_h[i])
                
            # select response with new_scale_id

            score_ = scores_[:,:,new_scale_id]
            # normalize scorereshape
            score_ = score_ - np.min(score_)
            score_ = score_/np.sum(score_)
            # apply displacement penalty
            score_ = (1-hp.window_influence)*score_ + hp.window_influence*self.penalty
            self.pos_x[i], self.pos_y[i] = self._update_target_position(self.pos_x[i], self.pos_y[i], score_, self.final_score_sz, hp.tot_stride, hp.search_sz, hp.response_up, self.x_sz[i])
    
            # update new z             
            new_z_sz[i] = (1-hp.scale_lr)*self.z_sz[i] + hp.scale_lr*scaled_exemplar[new_scale_id]

            # final bbox
            bbox = np.array([self.pos_x[i]-self.target_w[i]/2, self.pos_y[i]-self.target_h[i]/2, self.target_w[i], self.target_h[i]])

            bboxes.append(bbox)

            # min_score = np.amin(scores_, axis=(0,1))
            # normalized_score = max_score - min_score
            final_scores.append(np.mean(max_score))

        self.history["score"].append(final_scores)
        self.history["target_w"].append(self.target_w)
        self.history["target_h"].append(self.target_h)

        # update the target representation with a rolling average
        start_t_inner = time.time()
        self._update_z_feat(img)
        print("New output process time: %.4f seconds"%(time.time() - start_t_inner))
            
        # update template patch size
        self.z_sz = new_z_sz

        print("Score process time: %.3f seconds"%(time.time() - start_t))
        print("TOTAL forward time: %.3f seconds\n"%(time.time() - start_t_all))

        return bboxes

    def _update_z_feat(self, img):
        z_lr = self.cfg.z_lr
        if z_lr>0:
            new_output_z = self.sess.run(self.net_z, feed_dict={
                                                            self._pos_x_ph: self.pos_x,
                                                            self._pos_y_ph: self.pos_y,
                                                            self._z_sz_ph: self.z_sz,
                                                            self._image: img
                                                            })

            self.output_z = (1-z_lr)*self.output_z + z_lr*new_output_z

    def _update_target_position(self, pos_x, pos_y, score, final_score_sz, tot_stride, search_sz, response_up, x_sz):
        # find location of score maximizer
        p = np.asarray(np.unravel_index(np.argmax(score), np.shape(score)))
        # displacement from the center in search area final representation ...
        center = float(final_score_sz - 1) / 2
        disp_in_area = p - center
        # displacement from the center in instance crop
        disp_in_xcrop = disp_in_area * float(tot_stride) / response_up
        # displacement from the center in instance crop (in frame coordinates)
        disp_in_frame = disp_in_xcrop *  x_sz / search_sz
        # *position* within frame in frame coordinates
        pos_y, pos_x = pos_y + disp_in_frame[0], pos_x + disp_in_frame[1]
        return pos_x, pos_y


def _import_from_matconvnet(net_path):
    mat = scipy.io.loadmat(net_path)
    net_dot_mat = mat.get('net')
    # organize parameters to import
    params = net_dot_mat['params']
    params = params[0][0]
    params_names = params['name'][0]
    params_names_list = [params_names[p][0] for p in xrange(params_names.size)]
    params_values = params['value'][0]
    params_values_list = [params_values[p] for p in xrange(params_values.size)]
    return params_names_list, params_values_list

# find all parameters matching the codename (there should be only one)
def _find_params(x, params):
    matching = [s for s in params if x in s]
    # assert len(matching)==1, ('Ambiguous param name found')    
    return matching

def _get_templates_score_map(net_z, net_x): #, params_names_list, params_values_list):
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

    return nf



if __name__ == '__main__':
    import json
    from collections import namedtuple
    from utils import get_bbox_from_image, natural_sort
    from src.region_to_bbox import region_to_bbox
    import cv2


    # net_path = "./pretrained/baseline-conv5_e55.mat"
    # config_path = 'config/design.json'
    net_path = "./pretrained/2016-08-17.net.mat"
    config_path = './config/design_ori.json'
    # net_path = "./pretrained/net-epoch-50.mat"
    # config_path = './config/design_custom.json'
    image_folder = "./data/interoll2"
    start_idx = 0

    # load nn config
    with open(config_path) as json_file:
        design = json.load(json_file, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))         

    '''Get images'''
    frame_name_list = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(".jpg")]
    frame_name_list = natural_sort(frame_name_list)[start_idx:]
    num_frames = np.size(frame_name_list)

    # manually label first frame
    # first_frame_bbox = get_bbox_from_image(cv2.imread(frame_name_list[0]))
    frame_bboxes = np.array([[123, 285, 86, 69],[75, 109, 53, 33],[295, 90, 77, 59],[263, 76, 48, 32],[573, 307, 60, 84],[526, 311, 45, 53]])

    bbox_data = np.array([region_to_bbox(bbox) for bbox in frame_bboxes])

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    
    # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

#     '''Load network'''
    siam_net = SiameseNetwork(sess, design, trainable=False)
    siam_net.load_mat_model(net_path)

    sess.run(tf.global_variables_initializer())

    # # init with exemplar 'z' image
    first_frame = cv2.imread(frame_name_list[start_idx])
    first_frame = first_frame.astype(np.float32)

    pos_x = bbox_data[:,0]
    pos_y = bbox_data[:,1]
    target_w = bbox_data[:,2]
    target_h = bbox_data[:,3]
    siam_net.forward_z(first_frame, pos_x, pos_y, target_w, target_h)
    
    rnd_colors_bgr = np.array([(np.random.rand(3)*255).astype(np.int) for p in pos_x])
    rnd_colors_rgb_float = rnd_colors_bgr[:,::-1].astype(np.float) / 255
    labels = [str(i) for i in xrange(len(pos_x))]

    # PLOTTER
    import matplotlib.pyplot as plt
    
    fig = plt.figure()

    # fig2 = plt.figure()

    ax = fig.add_subplot(111)
    # ax.set_title("Score map")
    plt.ion()
    fig.show()
    fig.canvas.set_window_title('Score map')
    fig.canvas.draw()
    # fig2.show()
    # fig2.canvas.draw()
    
    # 
    # fig.gca().set_color_cycle([tuple(c[::-1]) for c in rnd_colors_bgr_float])
    skip_frame = 1
    for i in xrange(start_idx + 1, num_frames):
        if i % (skip_frame+1) != 0:
            continue
        frame = cv2.imread(frame_name_list[i])
        bboxes = siam_net.forward(frame.astype(np.float32))
        img_copy = frame # .astype(np.uint8)
        # img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
        fig.gca().clear()
        # fig2.gca().clear()

        # stats
        history_score = np.array(siam_net.history["score"])
        history_tw = np.array(siam_net.history["target_w"])
        history_th = np.array(siam_net.history["target_h"])

        for ix,bbox in enumerate(bboxes):
            color_bgr = tuple(rnd_colors_bgr[ix])
            color_rgb_f = rnd_colors_rgb_float[ix]
            # rnd_color = (np.random.rand(3)*255).astype(np.int)
            pt1 = bbox[:2].astype(np.int)
            pt2 = pt1 + bbox[2:4].astype(np.int)
            mid_pt = np.mean([pt1,pt2],axis=0).astype(np.int)
            cv2.putText(frame, "%d"%(ix), tuple(mid_pt), cv2.FONT_HERSHEY_SIMPLEX, 1, color_bgr, 2)
            img_copy = cv2.rectangle(img_copy, tuple(pt1), tuple(pt2), color_bgr)
            cv2.imshow("tracker", img_copy)

            _sc_ = history_score[:,ix]

            fig.gca().plot(_sc_, label=labels[ix], color=color_rgb_f)
            # fig2.gca().plot(_t_area, label=labels[ix], color=color_rgb_f)

        cv2.waitKey(5)

        # plot

        fig.gca().legend(labels)   
        # fig2.gca().legend(labels) 
        fig.canvas.draw()
        # fig2.canvas.draw()
