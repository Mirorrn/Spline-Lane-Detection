import tensorflow as tf
import numpy as np
import config as cfg

from keras import backend as K

def huber(true, pred, delta):
    loss = tf.where(tf.abs(true-pred) < delta , 0.5*((true-pred)**2), delta*tf.abs(true - pred) - 0.5*(delta**2))
 #   loss = tf.Print(loss, [loss], message="This is loss: ", summarize=1000)
    return loss

class loss:

    def __init__(self, config):
        self.config = config
        self.norm  = config.img_h // 2.
        self.alpha = config.alpha
        self.focal_loss = True
      #  self.mloss_conf = tf.Variable(0., )

      #  self.mloss_loc = tf.Variable(0., )
    def loss_test(self, y_true, y_pred):
        batch_size = tf.shape(y_true)[0]
        if self.config.staged:
            y_pred = tf.reshape(y_pred, [batch_size, self.config.grid_size, self.config.grid_size, self.config.num_prediction_cells, 4])
            y_true = tf.reshape(y_true, [batch_size, self.config.grid_size, self.config.grid_size,
                                         self.config.num_prediction_cells, (self.config.grid_cel_size + 1)*4  + 5 + 1])
       # alpha = self.config.alpha
     #   y_true = tf.Print(y_true, [tf.shape(y_true)], message="This is y_true: ", summarize=1000)
        conf_true = y_true[:, :, :, :, -1]
        conf_true = conf_true[..., np.newaxis]

        # care rotated coeff normalized
        #y_pred = tf.reshape(y_pred, [])
        a_pred = y_pred[:, :, :, :, 0] * self.config.a_range #+ self.config.a_shift
        a_pred = a_pred[..., np.newaxis]

        b_pred = y_pred[:, :, :, :, 1] * self.config.b_range #+ self.config.b_shift
        b_pred = b_pred[..., np.newaxis]

        c_pred = y_pred[:, :, :, :, 2] * self.config.c_range #+ self.config.c_shift
        c_pred = c_pred[..., np.newaxis]

      #  a_true = y_true[:, :, :, :, -10] * self.config.a_range + self.config.a_shift
      #  a_true = a_true[..., np.newaxis]

     #   b_true = y_true[:, :, :, :, -9] * self.config.b_range + self.config.b_shift
     #   b_true = b_true[..., np.newaxis]

     #   c_true = y_true[:, :, :, :, -8] * self.config.c_range + self.config.c_shift
     #   c_true = c_true[..., np.newaxis]

        conf_pred = y_pred[:, :, :, :, 3]
        conf_pred = conf_pred[..., np.newaxis]

        x_tr = y_true[:, :, :, :, 0:self.config.grid_cel_size + 1]
        ################################################# matching part for splines
      #  dif1_target = y_true[:, :, :, :, -23:-6]
    #    counted_anchors = tf.to_float(tf.count_nonzero(matching_anchor_point1[:, :, :, :,1],  axis=[1,2,3])) # care first cell in left corner assumes to be empty
      #  matching_anchor_point2 = y_true[:, :, :, :, -5:-1]

      #  matching_anchor_coef1 = tf.gather_nd(y_pred, tf.cast(matching_anchor_point1, tf.int32))
      #  matching_anchor_coef2 = tf.gather_nd(y_pred, tf.cast(matching_anchor_point2, tf.int32))

      #  a_pred_for_matching1 = matching_anchor_coef1[:, :, :, :, 0] * self.config.a_range + self.config.a_shift
      #  a_pred_for_matching1 = a_pred_for_matching1[..., np.newaxis]

      #  b_pred_for_matching1 = matching_anchor_coef1[:, :, :, :, 1] * self.config.b_range + self.config.b_shift
      #  b_pred_for_matching1 = b_pred_for_matching1[..., np.newaxis]
#
      #  c_pred_for_matching1 = matching_anchor_coef1[:, :, :, :, 2] * self.config.c_range + self.config.c_shift
    #    c_pred_for_matching1 = c_pred_for_matching1[..., np.newaxis]

     #   a_pred_for_matching2 = matching_anchor_coef2[:, :, :, :, 0] * self.config.a_range + self.config.a_shift
     #   a_pred_for_matching2 = a_pred_for_matching2[..., np.newaxis]

     #   b_pred_for_matching2 = matching_anchor_coef2[:, :, :, :, 1] * self.config.b_range + self.config.b_shift
      #  b_pred_for_matching2 = b_pred_for_matching2[..., np.newaxis]

      #  c_pred_for_matching2 = matching_anchor_coef2[:, :, :, :, 2] * self.config.c_range + self.config.c_shift
     #   c_pred_for_matching2 = c_pred_for_matching2[..., np.newaxis]

        #x_anchor_point_1 = y_true[:, :, :, :, self.config.grid_cel_size]
       # x_anchor_point_1 = x_anchor_point_1[..., np.newaxis]

      #  x_anchor_point_2 = y_true[:, :, :, :, 0]
      #  x_anchor_point_2 = x_anchor_point_2[..., np.newaxis]

     #   y_pre1 = 2 *a_pred * x_tr + b_pred #+ c_pred_for_matching1
       # y_pre1 = tf.expand_dims(matching_anchor_coef1[:, :, :, :,2*( self.config.grid_cel_size+1) -1], -1)
       # y_pre2 = 2 *a_pred_for_matching2 * x_anchor_point_2 + b_pred_for_matching2 #+ c_pred_for_matching2
     #   y_pre1 = tf.Print(y_pre1, [y_pre1], message="This is y_pre1: ", summarize=1000)
       # y_pre2 = tf.expand_dims(matching_anchor_coef2[:, :, :, :, self.config.grid_cel_size+1], -1)
       # dif_cell =  ( y_true[:, :, :, :, -3] - y_true[:, :, :, :, -7]) * self.config.grid_cel_size
        #dif_cell = tf.Print(dif_cell, [dif_cell], message="This is dif_cell: ", summarize=1000)
        #dif_cell = dif_cell[..., np.newaxis]

     #   y_pre2  = dif1_target #+ dif_cell
    #    y_pre2 = tf.Print(y_pre2, [y_pre2], message="This is y_pre2: ", summarize=1000)
       # y_pre2 = y_pre2[..., np.newaxis]

    #    y_pre3 = 2 * a_pred_for_matching1 * x_anchor_point_1 + b_pred_for_matching1# * x_anchor_point_1 #+ c_pred_for_matching1
    #    y_pre4 = 2 * a_pred_for_matching2 * x_anchor_point_2 + b_pred_for_matching2# * x_anchor_point_2 #+ c_pred_for_matching2

      #  dif_0 = tf.expand_dims(huber(y_pre1, y_pre2, 0.5), -1)

      #  dif_1 = huber(y_pre3, y_pre4, 0.5)

       # dif_test = tf.Print(dif_test, [dif_test], message="This is dif_test: ", summarize=1000)
       # sum_dif_0 = tf.reduce_sum(tf.multiply(conf_true, tf.reduce_mean(dif_0, axis=-1)), axis=[1, 2, 3, 4])
       # sum_dif_0 = tf.Print(sum_dif_0, [sum_dif_0], message="This is sum_dif_0: ", summarize=1000)
       # dif_0
      #  sum_dif_1 = tf.reduce_sum(tf.multiply(conf_true, dif_1), axis=[1, 2, 3, 4])

     #   sum_dif_0 = tf.Print(sum_dif_0, [sum_dif_0], message="This is sum_dif_0: ", summarize=1000)
        #counted_anchors = tf.Print(counted_anchors, [counted_anchors], message="This is counted_anchors: ", summarize=1000)
      #  loss_anchor = tf.divide( test_sum, counted_anchors)
      #  loss_anchor = tf.Print(loss_anchor, [loss_anchor], message="This is counted_anchors: ",
      #                             summarize=1000)
        #self.loss_anchor = 0.1 * tf.reduce_mean(test_sum)  # 0.5 factor?
       # self.loss_anchor = tf.Print(self.loss_anchor, [self.loss_anchor], message="This is self.loss_anchor: ", summarize=1000)
        ################################################ end of matching part for splines!

        y_tr = y_true[:, :, :, :, self.config.grid_cel_size + 1:2*(self.config.grid_cel_size + 1)]
        #y_tr = tf.Print(y_tr, [y_tr], message="This is y_tr: ", summarize=1000)
        #non_nans_idc = tf.where((y_tr != 1))
        #non_nans_idc = non_nans_idc[..., np.newaxis]
        counted_non_nan = tf.to_float(tf.count_nonzero(tf.to_float(tf.logical_not(tf.is_nan(y_tr))), axis=-1))
        y_tr = tf.where(tf.is_nan(y_tr), tf.zeros_like(y_tr), y_tr)

        #y_tr = tf.Print(y_tr, [tf.shape(y_tr)], message="This is y_tr: ", summarize=1000)
        y_pre = a_pred * x_tr ** 2 + b_pred * x_tr + c_pred

        weights = y_true[:, :, :, :, 2*(self.config.grid_cel_size + 1): 3*(self.config.grid_cel_size + 1)]
        # rotate prediction x = -y and y = x
        #conf_true = tf.Print(conf_true, [conf_true], message="This is conf: ", summarize=1000)
        #tf.boolean_mask(x, tf.logical_not(tf.is_inf(x))))
       # loss_loc =   tf.multiply(conf_true, tf.expand_dims(tf.divide( tf.reduce_sum(huber(y_tr,y_pre, .5)* weights, axis=-1), counted_non_nan), -1))
        loss_loc = tf.multiply(conf_true, tf.expand_dims(tf.reduce_mean(huber(y_tr, y_pre, 0.5)* weights, axis=-1), -1))
        #loss_loc = tf.Print(loss_loc, [loss_loc], message="This is loss_loc: ", summarize=1000)
        #loss_loc = tf.Print(loss_loc, [loss_loc], message="This is loss: ", summarize=1000)

        numb_of_trues = tf.count_nonzero(conf_true, axis=[1,2,3,4])
        numb_of_trues = tf.where(numb_of_trues == 0, tf.ones_like(numb_of_trues), numb_of_trues)
        numb_of_trues = tf.to_float(numb_of_trues)

        sum_loss_loc = tf.reduce_sum(loss_loc, axis=[1,2,3,4] )
        #numb_of_trues = tf.Print(numb_of_trues, [numb_of_trues], message="This is numb_of_trues: ", summarize=1000)
        #loss_loc = tf.div_no_nan(sum_loss_loc, numb_of_trues) # wrong, but gives a clue of weighting loc and conf

        self.mloss_loc = (1. - self.config.alpha)*(tf.reduce_mean(sum_loss_loc)) # 0.5 factor?
       # self.mloss_loc = tf.reduce_mean(sum_loss_loc)  # 0.5 factor?
       # loss_loc = tf.Print(loss_loc, [loss_loc], message="This is loss: ", summarize=1000)

        # CONF LOSS
       # conf_true_reshaped = tf.reshape(conf_true, [batch_size * (self.config.grid_size ** 2) * self.config.num_prediction_cells, 1])
       # conf_pred = tf.reshape(conf_pred, [batch_size * (self.config.grid_size ** 2) * self.config.num_prediction_cells, 1])
      #  conf_pred = tf.Print(conf_pred, [conf_pred], message="conf_pred: ", summarize=1000)
       # loss_conf = tf.nn.sigmoid_cross_entropy_with_logits(labels=conf_true_reshaped, logits=conf_pred)#,# weights=conf_true_reshaped * 1.5,
       #                                            # reduction=tf.losses.Reduction.NONE, label_smoothing=0.01)

        if self.focal_loss:
            print('Using Focal loss!')
            conf_pred = K.clip(tf.sigmoid(conf_pred), K.epsilon(), 1 - K.epsilon())
            pt_1 = tf.where(tf.equal(conf_true, 1), conf_pred, tf.ones_like(conf_pred))
            pt_0 = tf.where(tf.equal(conf_true, 0), conf_pred, tf.zeros_like(conf_pred))

            alpha= .25
            gamma = 0.01
            sum_loss_conf = -tf.reduce_sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1),axis=[1,2,3,4]) - tf.reduce_sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0),axis=[1,2,3,4])

            #sum_loss_conf = tf.Print(sum_loss_conf, [sum_loss_conf], message="sum_loss_conf: ", summarize=1000)
            #loss_conf = tf.div_no_nan(sum_loss_conf, numb_of_trues)
            self.mloss_conf = self.config.alpha * tf.reduce_mean(sum_loss_conf)  # 0.5 factor?
        else:
            print('Using cross entropie!')
            #conf_pred = K.clip(tf.sigmoid(conf_pred), K.epsilon(), 1 - K.epsilon())
           # pt_1 = tf.where(tf.equal(conf_true, 1), conf_pred, tf.ones_like(conf_pred))
           # pt_0 = tf.where(tf.equal(conf_true, 0), conf_pred, tf.zeros_like(conf_pred))
           # pt_1 = tf.Print(pt_1, [pt_1], message="This is pt_1: ", summarize=1000)
            #sum_loss_conf = -tf.reduce_sum(K.log(pt_1), axis=[1, 2, 3, 4]) - tf.reduce_sum(K.log(1. - pt_0), axis=[1, 2, 3, 4])

            conf_true_reshaped = tf.reshape(conf_true, [batch_size * (self.config.grid_size ** 2) * self.config.num_prediction_cells, 1])
            conf_pred = tf.reshape(conf_pred,[batch_size * (self.config.grid_size ** 2) * self.config.num_prediction_cells, 1])

            loss_conf = tf.losses.sigmoid_cross_entropy(multi_class_labels=conf_true_reshaped, logits=conf_pred,# weights=conf_true_reshaped * 1.5,
                                                    reduction=tf.losses.Reduction.NONE,)# label_smoothing=0)

            loss_conf = tf.reshape(loss_conf, [batch_size, self.config.grid_size, self.config.grid_size,
                                               self.config.num_prediction_cells, 1])
            sum_loss_conf = tf.reduce_sum(loss_conf, axis=[1, 2, 3, 4])
           # loss_conf = tf.div_no_nan(sum_loss_conf, numb_of_trues)
            self.mloss_conf = self.config.alpha * tf.reduce_mean(sum_loss_conf)  # 0.5 factor?


            # conf_pred = tf.reshape(conf_pred, [batch_size * (self.config.grid_size ** 2) * self.config.num_prediction_cells, 1])
      #  loss_conf = (1. + tf.abs(conf_true_reshaped-tf.sigmoid(conf_pred))**self.config.cls_reg) * loss_conf # https://arxiv.org/pdf/1708.02002.pdf
       # loss_conf = tf.Print(loss_conf, [loss_conf], message="loss_conf2: ", summarize=1000)
        #loss_conf = tf.expand_dims(tf.square(conf_true_reshaped -conf_pred ), -1 )
     #   loss_conf = tf.reshape(loss_conf, [batch_size, self.config.grid_size, self.config.grid_size, self.config.num_prediction_cells, 1])


        self.loss_sum = self.mloss_conf + self.mloss_loc
        return self.loss_sum  # + self.loss_anchor
        #return tf.maximum(self.mloss_conf,self.mloss_loc) #+ self.loss_anchor



    def loss(self, y_true, y_pred):
        batch_size = tf.shape(y_true)[0]
        if self.config.staged:
            y_pred = tf.reshape(y_pred, [batch_size, self.config.grid_size, self.config.grid_size,
                                         self.config.num_prediction_cells, 4])
            y_true = tf.reshape(y_true, [batch_size, self.config.grid_size, self.config.grid_size,
                                         self.config.num_prediction_cells, (self.config.grid_cel_size + 1) * 4 + 5 + 1])

        conf_true = y_true[:, :, :, :, -1]
        conf_true = conf_true[..., np.newaxis]

        # care rotated coeff normalized
        a_pred = y_pred[:, :, :, :, 0] * self.config.a_range
        a_pred = a_pred[..., np.newaxis]

        b_pred = y_pred[:, :, :, :, 1] * self.config.b_range
        b_pred = b_pred[..., np.newaxis]

        c_pred = y_pred[:, :, :, :, 2] * self.config.c_range
        c_pred = c_pred[..., np.newaxis]

        conf_pred = y_pred[:, :, :, :, 3]
        conf_pred = conf_pred[..., np.newaxis]

      #  numb_of_trues = tf.count_nonzero(conf_true, axis=[1, 2, 3, 4])
      #  numb_of_trues = tf.where(numb_of_trues == 0, tf.ones_like(numb_of_trues), numb_of_trues)
      #  numb_of_trues = tf.to_float(numb_of_trues)

        x_tr = y_true[:, :, :, :, 0:self.config.grid_cel_size + 1]

        y_tr = y_true[:, :, :, :, self.config.grid_cel_size + 1:2 * (self.config.grid_cel_size + 1)]

        #y_tr = tf.where(tf.is_nan(y_tr), tf.zeros_like(y_tr), y_tr)

        y_pre = a_pred * x_tr ** 2 + b_pred * x_tr + c_pred

        weights = y_true[:, :, :, :, 2 * (self.config.grid_cel_size + 1): 3 * (self.config.grid_cel_size + 1)]
        loss_loc = tf.multiply(conf_true, tf.expand_dims(tf.reduce_mean(huber(y_tr, y_pre, 0.5) * weights, axis=-1), -1))


        sum_loss_loc = tf.reduce_sum(loss_loc, axis=[1, 2, 3, 4])
      #  sum_loss_loc = tf.div_no_nan(sum_loss_loc, numb_of_trues)
        self.mloss_loc = (1. - self.config.alpha) * (tf.reduce_mean(sum_loss_loc))  # 0.5 factor?

        if self.focal_loss:
            print('Using Focal loss!')
            conf_pred = K.clip(tf.sigmoid(conf_pred), K.epsilon(), 1 - K.epsilon())
            pt_1 = tf.where(tf.equal(conf_true, 1), conf_pred, tf.ones_like(conf_pred))
            pt_0 = tf.where(tf.equal(conf_true, 0), conf_pred, tf.zeros_like(conf_pred))

           # sum_loss_conf = -tf.reduce_sum(self.config.alpha_focal * K.pow(1. - pt_1, self.config.gamma) * K.log(pt_1),
           #                                axis=[1, 2, 3, 4]) - tf.reduce_sum(
           #     (1 - self.config.alpha_focal) * K.pow(pt_0, self.config.gamma) * K.log(1. - pt_0), axis=[1, 2, 3, 4])

            self.mloss_conf_TRUE =  tf.reduce_mean(-tf.reduce_sum(self.config.alpha_focal * K.pow(1. - pt_1, self.config.gamma) * K.log(pt_1),
                                           axis=[1, 2, 3, 4]))
            self.mloss_conf_FALSE = tf.reduce_mean( - tf.reduce_sum(
                (1 - self.config.alpha_focal) * K.pow(pt_0, self.config.gamma) * K.log(1. - pt_0), axis=[1, 2, 3, 4]))
            sum_loss_conf = self.mloss_conf_TRUE + self.mloss_conf_FALSE
            # sum_loss_conf = tf.Print(sum_loss_conf, [sum_loss_conf], message="sum_loss_conf: ", summarize=1000)
            # loss_conf = tf.div_no_nan(sum_loss_conf, numb_of_trues)
            self.mloss_conf = self.config.focal_loss_param * self.config.alpha * sum_loss_conf
        #    self.mloss_conf =self.config.focal_loss_param  * self.config.alpha * tf.reduce_mean(sum_loss_conf)  # 0.5 factor?
        else:
            print('Using cross entropie!')
           # conf_true_reshaped = tf.reshape(conf_true, [batch_size * (self.config.grid_size ** 2) * self.config.num_prediction_cells, 1])
           # conf_pred = tf.reshape(conf_pred,[batch_size * (self.config.grid_size ** 2) * self.config.num_prediction_cells, 1])

           # loss_conf = tf.expand_dims(tf.nn.sigmoid_cross_entropy_with_logits(labels=conf_true_reshaped, logits=conf_pred), -1)

           # loss_conf = tf.reshape(loss_conf, [batch_size, self.config.grid_size, self.config.grid_size,
           #                                    self.config.num_prediction_cells, 1])
           # sum_loss_conf = tf.reduce_sum(loss_conf, axis=[1, 2, 3, 4])
            conf_pred = K.clip(tf.sigmoid(conf_pred), K.epsilon(), 1 - K.epsilon())
            pt_1 = tf.where(tf.equal(conf_true, 1), conf_pred, tf.ones_like(conf_pred))
            pt_0 = tf.where(tf.equal(conf_true, 0), conf_pred, tf.zeros_like(conf_pred))

            sum_loss_conf = -tf.reduce_sum(K.log(pt_1),
                                           axis=[1, 2, 3, 4]) - tf.reduce_sum(K.log(1. - pt_0), axis=[1, 2, 3, 4])
           # sum_loss_conf = tf.div_no_nan(sum_loss_conf, numb_of_trues)
            self.mloss_conf = self.config.alpha * tf.reduce_mean(sum_loss_conf)  # 0.5 factor?

        self.loss_sum = self.mloss_conf + self.mloss_loc
        return self.loss_sum  # + self.loss_anchor
        # return tf.maximum(self.mloss_conf,self.mloss_loc) #+ self.loss_anchor


    def loss_KONF(self, y_true, y_pred):
        batch_size = tf.shape(y_true)[0]
        if self.config.staged:
            y_pred = tf.reshape(y_pred, [batch_size, self.config.grid_size, self.config.grid_size,
                                         self.config.num_prediction_cells, 1])
            y_true = tf.reshape(y_true, [batch_size, self.config.grid_size, self.config.grid_size,
                                         self.config.num_prediction_cells, (self.config.grid_cel_size + 1) * 4 + 5 + 1])

        conf_pred = y_pred[:, :, :, :, 0]
        conf_pred = conf_pred[..., np.newaxis]

        conf_true = y_true[:, :, :, :, -1]
        conf_true = conf_true[..., np.newaxis]

        if self.focal_loss:
            print('Using Focal loss!')
            conf_pred = K.clip(tf.sigmoid(conf_pred), K.epsilon(), 1 - K.epsilon())
            pt_1 = tf.where(tf.equal(conf_true, 1), conf_pred, tf.ones_like(conf_pred))
            pt_0 = tf.where(tf.equal(conf_true, 0), conf_pred, tf.zeros_like(conf_pred))

           # sum_loss_conf = -tf.reduce_sum(self.config.alpha_focal * K.pow(1. - pt_1, self.config.gamma) * K.log(pt_1),
           #                                axis=[1, 2, 3, 4]) - tf.reduce_sum(
           #     (1 - self.config.alpha_focal) * K.pow(pt_0, self.config.gamma) * K.log(1. - pt_0), axis=[1, 2, 3, 4])

            self.mloss_conf_TRUE = tf.reduce_mean(-tf.reduce_sum(self.config.alpha_focal * K.pow(1. - pt_1, self.config.gamma) * K.log(pt_1),
                                           axis=[1, 2, 3, 4]))
            self.mloss_conf_FALSE =   tf.reduce_mean( - tf.reduce_sum(
                (1 - self.config.alpha_focal) * K.pow(pt_0, self.config.gamma) * K.log(1. - pt_0), axis=[1, 2, 3, 4]))
            sum_loss_conf = self.mloss_conf_TRUE + self.mloss_conf_FALSE
            # sum_loss_conf = tf.Print(sum_loss_conf, [sum_loss_conf], message="sum_loss_conf: ", summarize=1000)
            # loss_conf = tf.div_no_nan(sum_loss_conf, numb_of_trues)
            self.mloss_conf = self.config.focal_loss_param * self.config.alpha * sum_loss_conf
        #    self.mloss_conf =self.config.focal_loss_param  * self.config.alpha * tf.reduce_mean(sum_loss_conf)  # 0.5 factor?
        else:
            print('Using cross entropie!')
           # conf_true_reshaped = tf.reshape(conf_true, [batch_size * (self.config.grid_size ** 2) * self.config.num_prediction_cells, 1])
           # conf_pred = tf.reshape(conf_pred,[batch_size * (self.config.grid_size ** 2) * self.config.num_prediction_cells, 1])

           # loss_conf = tf.expand_dims(tf.nn.sigmoid_cross_entropy_with_logits(labels=conf_true_reshaped, logits=conf_pred), -1)

           # loss_conf = tf.reshape(loss_conf, [batch_size, self.config.grid_size, self.config.grid_size,
           #                                    self.config.num_prediction_cells, 1])
           # sum_loss_conf = tf.reduce_sum(loss_conf, axis=[1, 2, 3, 4])
            conf_pred = K.clip(tf.sigmoid(conf_pred), K.epsilon(), 1 - K.epsilon())
            pt_1 = tf.where(tf.equal(conf_true, 1), conf_pred, tf.ones_like(conf_pred))
            pt_0 = tf.where(tf.equal(conf_true, 0), conf_pred, tf.zeros_like(conf_pred))

            sum_loss_conf = -tf.reduce_sum(K.log(pt_1),
                                           axis=[1, 2, 3, 4]) - tf.reduce_sum(K.log(1. - pt_0), axis=[1, 2, 3, 4])
           # sum_loss_conf = tf.div_no_nan(sum_loss_conf, numb_of_trues)
            self.mloss_conf = self.config.alpha * tf.reduce_mean(sum_loss_conf)  # 0.5 factor?
        return self.mloss_conf

    def loss_LOK(self, y_true, y_pred):
        batch_size = tf.shape(y_true)[0]
        if self.config.staged:
            y_pred = tf.reshape(y_pred, [batch_size, self.config.grid_size, self.config.grid_size,
                                         self.config.num_prediction_cells, 3])
            y_true = tf.reshape(y_true, [batch_size, self.config.grid_size, self.config.grid_size,
                                         self.config.num_prediction_cells, (self.config.grid_cel_size + 1) * 4 + 5 + 1])

        conf_true = y_true[:, :, :, :, -1]
        conf_true = conf_true[..., np.newaxis]

        # care rotated coeff normalized
        a_pred = y_pred[:, :, :, :, 0] * self.config.a_range
        a_pred = a_pred[..., np.newaxis]

        b_pred = y_pred[:, :, :, :, 1] * self.config.b_range
        b_pred = b_pred[..., np.newaxis]

        c_pred = y_pred[:, :, :, :, 2] * self.config.c_range
        c_pred = c_pred[..., np.newaxis]


        x_tr = y_true[:, :, :, :, 0:self.config.grid_cel_size + 1]

        y_tr = y_true[:, :, :, :, self.config.grid_cel_size + 1:2 * (self.config.grid_cel_size + 1)]

        # y_tr = tf.where(tf.is_nan(y_tr), tf.zeros_like(y_tr), y_tr)

        y_pre = a_pred * x_tr ** 2 + b_pred * x_tr + c_pred

        weights = y_true[:, :, :, :, 2 * (self.config.grid_cel_size + 1): 3 * (self.config.grid_cel_size + 1)]
        loss_loc = tf.multiply(conf_true,
                               tf.expand_dims(tf.reduce_mean(huber(y_tr, y_pre, 0.5) * weights, axis=-1), -1))

        sum_loss_loc = tf.reduce_sum(loss_loc, axis=[1, 2, 3, 4])
        #  sum_loss_loc = tf.div_no_nan(sum_loss_loc, numb_of_trues)
        self.mloss_loc = (1. - self.config.alpha) * (tf.reduce_mean(sum_loss_loc))  # 0.5 factor?
        return self.mloss_loc

    def loss_lane(self, y_true, y_pred):


        conf_true = y_true[:, :, -1]
        conf_true = conf_true[..., np.newaxis]

        # care rotated coeff normalized
        #y_pred = tf.reshape(y_pred, [])
        a_pred = y_pred[:, :, 0] * self.config.a_range + self.config.a_shift
        a_pred = a_pred[..., np.newaxis]

        b_pred = y_pred[:, :, 1] * self.config.b_range + self.config.b_shift
        b_pred = b_pred[..., np.newaxis]

        c_pred = y_pred[:, :, 2] * self.config.c_range + self.config.c_shift
        c_pred = c_pred[..., np.newaxis]

        # ytrue y points
        y_points = y_true[:, :, 3:6]
        #y_points = y_points[..., np.newaxis]

        # ytrue x points
        x_points = y_true[:, :, 0:3]
        #x_points = x_points[..., np.newaxis]

        conf_pred = y_pred[:, :, -1]
        conf_pred = conf_pred[..., np.newaxis]


        y_pre = (a_pred * x_points ** 2 + b_pred * x_points + c_pred)
        #y_pre = tf.Print(y_pre, [y_pre], message="This is conf: ", summarize=1000)
        loss_loc = tf.multiply(conf_true, tf.expand_dims(tf.reduce_sum(huber(y_points,y_pre, 0.5), axis=-1), -1))
        sum_loss_loc = tf.reduce_sum(loss_loc, axis=[1, 2] )
        self.mloss_loc = tf.reduce_mean(sum_loss_loc) # 0.5 factor?

       # x_min = y_true[:, :, 6]
       # x_min = x_min[..., np.newaxis]
       # x_max = y_true[:, :, 11]
       # x_max = x_max[..., np.newaxis]
       # x_minmax = tf.concat([x_min,x_max] , axis=-1)
        x_minmax = x_points
        #x_minmax = tf.Print(x_minmax, [x_minmax], message="This is x_minmax: ", summarize=1000)
        loss_coord = tf.multiply(conf_true, tf.expand_dims(tf.reduce_sum(huber(x_minmax, y_pred[:, :, 3:6], 0.5), axis=-1), -1))
       # loss_coord = tf.Print(loss_coord, [loss_coord], message="This is conf: ", summarize=1000)
        loss_coord = tf.reduce_sum(loss_coord, axis=[1, 2])
     #   sum_loss_coord = tf.Print(sum_loss_coord, [sum_loss_coord], message="This is sum_loss_coord: ", summarize=1000)
        self.mloss_coord = tf.reduce_mean(loss_coord)  # 0.5 factor?

        # CONF LOSS
        batch_size = tf.shape(y_true)[0]
        conf_true_reshaped = tf.reshape(conf_true, [batch_size * self.config.num_prediction_cells, 1])
        conf_pred_reshaped = tf.reshape(conf_pred, [batch_size * self.config.num_prediction_cells, 1])
        loss_conf = tf.expand_dims(tf.losses.sigmoid_cross_entropy(multi_class_labels=conf_true_reshaped, logits=conf_pred_reshaped,# weights=conf_true_reshaped * 1.5,
                                                    reduction=tf.losses.Reduction.NONE, label_smoothing=0), -1)

        loss_conf = tf.reshape(loss_conf, [batch_size, self.config.num_prediction_cells, 1])
       # loss_conf = tf.Print(loss_conf, [loss_conf], message="This is loss_conf: ", summarize=1000)
        sum_loss_conf = tf.reduce_sum(loss_conf, axis=[1, 2])

        self.mloss_conf = tf.reduce_mean(sum_loss_conf)  # 0.5 factor?

        return self.mloss_conf + self.mloss_coord + self.mloss_loc

    def loss_nb(self, y_true, y_pred):
        batch_size = tf.shape(y_true)[0]
        #################################################################################################
        c_tr = y_true

        c_pre = y_pred
    #    c_pre = tf.Print(c_pre, [tf.shape(c_pre)], message="This is c_pre: ", summarize=1000)
      #  error = tf.reduce_sum(tf.square(y_true-y_pred), axis=-1)
     #   c_pre = tf.Print(c_pre, [c_pre], message="This is c_pre: ", summarize=1000)
        c_tr_flatten = tf.reshape(c_tr, [batch_size * (self.config.grid_size ** 2), 1])
        c_pre_flatten = tf.reshape(c_pre, [batch_size * (self.config.grid_size ** 2), 1])
       # c_pre_flatten = tf.Print(c_pre_flatten, [c_pre_flatten], message="This is c_pre_flatten: ", summarize=1000)
        loss_c = tf.expand_dims(
            tf.losses.sigmoid_cross_entropy(multi_class_labels=c_tr_flatten, logits=c_pre_flatten,
                                            reduction=tf.losses.Reduction.NONE, label_smoothing=0), -1)
        loss_c = tf.reshape(loss_c, [batch_size, self.config.grid_size, self.config.grid_size, 1])
       # loss_c = tf.Print(loss_c, [loss_c], message="This is loss_c: ", summarize=1000)
        sum_loss_c = tf.reduce_sum(loss_c, axis=[1, 2])

       # sum_loss_c = tf.Print(sum_loss_c, [sum_loss_c], message="This is sum_loss_c: ", summarize=1000)
        self.mloss_segmentation = tf.reduce_mean(sum_loss_c)
        return self.mloss_segmentation

    def loss_lane_points(self, y_true, y_pred):

        #alpha = self.config.alpha

        conf_true = y_true[:, :, -1]
        conf_true = conf_true[..., np.newaxis]

        # ytrue y points
        y_points = y_true[:, :, 3:6]
        #y_points = y_points[..., np.newaxis]

        # ytrue x points
        x_points = y_true[:, :, 0:6]
        #x_points = x_points[..., np.newaxis]

        conf_pred = y_pred[:, :, -1]
        conf_pred = conf_pred[..., np.newaxis]


        #y_pre = (a_pred * x_points ** 2 + b_pred * x_points + c_pred)
        #y_pre = tf.Print(y_pre, [y_pre], message="This is conf: ", summarize=1000)
        #loss_loc = tf.multiply(conf_true, tf.expand_dims(tf.reduce_sum(huber(y_points,y_pre, 0.5), axis=-1), -1))
        #sum_loss_loc = tf.reduce_sum(loss_loc, axis=[1, 2] )
        #self.mloss_loc = tf.reduce_mean(sum_loss_loc) # 0.5 factor?

       # x_min = y_true[:, :, 6]
       # x_min = x_min[..., np.newaxis]
       # x_max = y_true[:, :, 11]
       # x_max = x_max[..., np.newaxis]
       # x_minmax = tf.concat([x_min,x_max] , axis=-1)
        x_minmax = x_points
        #x_minmax = tf.Print(x_minmax, [x_minmax], message="This is x_minmax: ", summarize=1000)
        loss_coord = tf.multiply(conf_true, tf.expand_dims(tf.reduce_sum(huber(x_minmax, y_pred[:, :, 0:6], 0.5), axis=-1), -1))
       # loss_coord = tf.Print(loss_coord, [loss_coord], message="This is conf: ", summarize=1000)
        loss_coord = tf.reduce_sum(loss_coord, axis=[1, 2])
     #   sum_loss_coord = tf.Print(sum_loss_coord, [sum_loss_coord], message="This is sum_loss_coord: ", summarize=1000)
        self.mloss_coord = tf.reduce_mean(loss_coord)  # 0.5 factor?

        # CONF LOSS
        batch_size = tf.shape(y_true)[0]
        conf_true_reshaped = tf.reshape(conf_true, [batch_size * self.config.num_prediction_cells, 1])
        conf_pred_reshaped = tf.reshape(conf_pred, [batch_size * self.config.num_prediction_cells, 1])
        loss_conf = tf.expand_dims(tf.losses.sigmoid_cross_entropy(multi_class_labels=conf_true_reshaped, logits=conf_pred_reshaped,# weights=conf_true_reshaped * 1.5,
                                                    reduction=tf.losses.Reduction.NONE, label_smoothing=0), -1)

        loss_conf = tf.reshape(loss_conf, [batch_size, self.config.num_prediction_cells, 1])
       # loss_conf = tf.Print(loss_conf, [loss_conf], message="This is loss_conf: ", summarize=1000)
        sum_loss_conf = tf.reduce_sum(loss_conf, axis=[1, 2])

        self.mloss_conf = tf.reduce_mean(sum_loss_conf)  # 0.5 factor?

        return self.mloss_conf + self.mloss_coord #+ self.mloss_loc

    def loss_sum(self,y_true, y_pred):
        return self.loss_sum

    def lossTRUE(self,y_true, y_pred):
        return self.mloss_conf_TRUE

    def lossFALSE(self,y_true, y_pred):
        return self.mloss_conf_FALSE

    def confidence_loss(self,y_true, y_pred):
        return self.mloss_conf

    def loc_loss(self,y_true, y_pred):
        return self.mloss_loc

    def loss_coord(self,y_true, y_pred):
        return self.mloss_coord

    def loss_segmentation(self,y_true, y_pred):
        return self.mloss_segmentation

#    sum_loss_loc = tf.reduce_sum(loss_loc, axis=[1,2,3] )
#        loss_loc = tf.truediv(sum_loss_loc, numb_of_trues) # wrong, but gives a clue of weighting loc and conf
#        loss_loc_without_nans_mask = tf.logical_not(tf.is_nan(loss_loc))#filter nan's!
#        loss_loc_without_nans_mask = tf.Print(loss_loc_without_nans_mask, [loss_loc_without_nans_mask], message="This is loss: ", summarize=1000)
#        loss_loc = tf.boolean_mask(loss_loc, loss_loc_without_nans_mask)

 #       self.mloss_loc = 0.5 * tf.reduce_mean(loss_loc) # 0.5 factor?