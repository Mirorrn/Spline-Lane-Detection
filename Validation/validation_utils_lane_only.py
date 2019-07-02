import random

import numpy as np
import pickle
from cv2 import polylines, circle
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def sigmoid(value):
    if -value > np.log(np.finfo(type(value)).max):
        return 0.0
    a = np.exp(-value)
    return 1.0/ (1.0 + a)

def plot_redy_line(x,y, config):
    y_f = y
   # y_f, x_t = x.copy(), y_f.copy() # for ratoation
    x_t = np.array(x)* config.img_h
    y_f = np.array(y_f)* config.img_h
    return x_t, y_f

def iterate_and_validate_predictions_cells(prediction, gt, cfg):

    y_tr_roc = []
    y_pr_roc = []
    error = []

    for i, s in enumerate(prediction):
        for a in s:
            a_pre, b_pre, c_pre, conf, = a

            x = np.arange(-cfg.grich_anchor_pt, cfg.grich_anchor_pt, 1 / cfg.num_of_samples)
            conf_true = gt[i, k, l, a, -1]

            if conf_true:
                y_tr_roc.append(True)
                a_pre = a_pre * cfg.a_range + cfg.a_shift
                b_pre = b_pre * cfg.b_range + cfg.b_shift
                c_pre = c_pre * cfg.c_range + cfg.c_shift
                y_f = np.round(a_pre * x ** 2 + b_pre * x + c_pre)
                error.append((y_f - gt[i, k, l, a, :-1])**2) ## square error
            else:
                y_tr_roc.append(False)

            conf = sigmoid(conf)
            y_pr_roc.append(conf)

    y_tr_roc = np.asarray(y_tr_roc)
    y_pr_roc = np.asarray(y_pr_roc)
    error = np.asarray(error)

    return y_tr_roc, y_pr_roc, error

def class_report(y_tr_roc, y_pr_class, threshold):
    y_pr_class = y_pr_class.copy()
    for i,s in enumerate(y_pr_class):
        if s >= threshold:
            y_pr_class[i] = True
        else:
            y_pr_class[i] = False
    print(classification_report(y_tr_roc, y_pr_class))

def rgb_color(a):
    if a == 0:
        return (255,0,0)
    if a == 1:
        return (0,255,0)
    if a == 2:
        return (0,0,255)

def plot_image(pred, config, with_print= False):
    filename_cent = config.DIR + config.experiment_name + '/cluster_cent.sav'
    cluster_cent = pickle.load(open(filename_cent, 'rb'))
    plot_image = np.zeros((config.img_w, config.img_w), dtype=np.float32)
    if with_print:
        print('-----------------------New Prediction---------------------------------------')
    for a in range(config.num_prediction_cells):
        a_ko, b, c,min_x,mid_x,  max_x, conf = pred[a, :]
        # sc = sigmoid(sc)
        conf = sigmoid(conf)


        if conf >= config.conf_thr:

            a_ko = a_ko #* config.a_range + config.a_shift
            b = b #* config.b_range + config.b_shift
            c = c #* config.c_range + config.c_shift
            x_pred = np.array([min_x,mid_x, max_x])
            y_f = np.array([a_ko,b, c])
            #y_f = a_ko * x_pred ** 2 + b * x_pred + c
            y_f, x_t = plot_redy_line(y_f, x_pred, config)
            y_f = y_f + cluster_cent[a][0]
            x_t = x_t + cluster_cent[a][1]

            #y_f = y_f + config.img_h //2
            #x_t = x_t + config.img_h //2
            f = np.array([x_t, y_f]).T


            if with_print:
                #print('Score ' + ' a: ' + str(a_ko) + ' b: ' + str(b) + ' c: ' + str(c) + ' conf: ' + str(
                #    conf)+ 'min: ' +str(min_x)+ 'max: ' +str(max_x))  # + ' Fit correct:' + str(fit_t) + ' Fit wrong:' + str(fit_f))
               print(' conf: ' + str(conf)+ 'min: ' +str(x_t[0])+ 'max: ' +str(x_t[-1]))  # + ' Fit correct:' + str(fit_t) + ' Fit wrong:' + str(fit_f))
               plot_image = polylines(plot_image, np.int32([f]), 0, 1, thickness=1)
               #plot_image = circle(plot_image, (int(x_t[0]), 100) , 3, (1, 1, 1), thickness=1, lineType=8,
              #                 shift=0)
               #plot_image = circle(plot_image, (int(x_t[-1]), 100), 10, (1, 1, 1), thickness=1, lineType=8,
               #                     shift=0)
             #  plot_image = circle(plot_image, (y_f[2], x_t[2]), 6, (1, 1, 1), thickness=1, lineType=8,
              #              shift=0)
    if with_print:
        print('----------------------------------------------------------------------------')

    return plot_image

def plot_image3d(grid, config, is_prediction, with_print= False):
    x = np.arange(-config.grich_anchor_pt, config.grich_anchor_pt, 1 / config.num_of_samples)

    fig = plt.figure()
    if is_prediction:
        fig.canvas.set_window_title('Prediction')
    else:
        fig.canvas.set_window_title('Ground_Truth')
    ax = fig.gca(projection='3d')
   # ax.grid(True)

    grid_test = np.arange(0, config.img_w, 16)
    ax.set_xticks(grid_test, minor=False)
    ax.set_yticks(grid_test, minor=False)
   # ax.set_yticks(grid_test, minor=True)
    ax.xaxis.grid(True, which='major')
    ax.yaxis.grid(True, which='major')
    #ax.set_zticks(grid_test, minor=False)

    # count ground thruth
    counter_gt = 0
    # count pred
    counter_pred = 0

    if with_print:
        print('-----------------------New Prediction---------------------------------------')

    for k in range(0, config.grid_size):
        for l in range(0, config.grid_size):
            for a in range(config.num_prediction_cells):
                if is_prediction:

                    a_ko, b, c, conf = grid[k, l, a, :]
                    conf = sigmoid(conf)

                    if conf >= config.conf_thr:
                        a_ko = a_ko * config.a_range + config.a_shift
                        b = b * config.b_range + config.b_shift
                        c = c * config.c_range + config.c_shift
                        conf = sigmoid(conf)

                        y_fill = np.round(a_ko * x ** 2 + b * x + c)
                        x_t, y_fill = plot_redy_line(x, y_fill, l, k, config)
                        f = np.array([x_t, y_fill]).T

                        plot_image = np.zeros((config.img_w, config.img_w), dtype=np.float32)
                        if with_print:
                            print('Score ' + ' a: ' + str(a_ko) + ' b: ' + str(b) + ' c: ' + str(c) + ' conf: ' + str(
                                conf))  # + ' Fit correct:' + str(fit_t) + ' Fit wrong:' + str(fit_f))

                        plot_image = polylines(plot_image, np.int32([f]), 0, 1, thickness=1)
                        indices = np.where(plot_image != 0)
                        z = np.ones_like(indices[1]) * a
                        ax.plot(indices[1], indices[0], z, label='parametric curve')
                       # ax.legend()
                        #ax.plot(all_lines[1][1], all_lines[1][0], all_lines[1][2])  # , projection='3d')
                        counter_pred += 1
                else:
                    if grid[k, l, a,-1]:
                        x_t, y_f = plot_redy_line(x, grid[k, l, a,:-1], l, k, config)
                        z = np.ones_like(x_t) * a
                        ax.plot(y_f, x_t, z, label='parametric curve')
                        counter_gt += 1
                        #ax.legend()
    print('Ground thruth counted: ' + str(counter_gt))
    print('Prediction counted: ' + str(counter_pred))
    if with_print:
        print('----------------------------------------------------------------------------')

    return 0

def iou(line1, line2):
    # assume that the line1 and 2 are [0, 1] images
    sum_line1 = np.sum(line1)   # with assumption sum is just only a count
    sum_line2 = np.sum(line2)
    inter_sum = np.sum(line1 *line2)
    union_sum = sum_line1 + sum_line2 - inter_sum
    if union_sum == 0.:
        return 0.
    else:
        return np.divide(inter_sum, union_sum)# return the iou

def ev_iou_based_assignment(row_ind, col_ind, assignment_matrix, gt_grid_index, pred_grid_index, iou_thr, cfg):
    new_gt_grid_index = []
    new_pred_grid_index = []
    for row_io, col_io in zip(row_ind, col_ind):
        if assignment_matrix[row_io, col_io] >= iou_thr:
            new_pred_grid_index.append(row_io)
            new_gt_grid_index.append(col_io)

    TP = len(new_pred_grid_index)
    FP = len(pred_grid_index) - TP
    FN = len(gt_grid_index) - TP

    precission = np.divide(TP, (TP + FP))
    recall = np.divide(TP, (TP + FN))
    prod = precission * recall
    sum_prec_rc = precission + recall
    f1 = np.divide(2*(prod), sum_prec_rc)

    # for debug reason plot matched gridbox
    if 0:
        not_matched_gt = [x for x in range(len(gt_grid_index)) if x not in new_gt_grid_index] # look for not matched indexs
        not_matched_pred = [x for x in range(len(pred_grid_index)) if x not in new_pred_grid_index]  # look for not matched indexs
        plotgrid_gt = np.zeros((cfg.grid_size, cfg.grid_size, 1), dtype=np.float32)
        plotgrid_assigned = np.zeros((cfg.grid_size, cfg.grid_size, cfg.num_prediction_cells), dtype=np.float32)
        f, axarr = plt.subplots(1, 2)
        for k in not_matched_gt:
            r, c = gt_grid_index[k][:2]
            plotgrid_gt[r, c] = 1
        axarr[0].imshow(plotgrid_gt[:, :, 0], cmap='gray')  # only for debug
        axarr[0].set_title('FN', color='0.7')

        for k in not_matched_pred:
            r, c = pred_grid_index[k][:2]
            plotgrid_assigned[r, c] = 1
        axarr[1].imshow(plotgrid_assigned[:, :, 0], cmap='gray')  # only for debug
        axarr[1].set_title('FP', color='0.7')
##
        for s in row_ind:
            r, c ,a=pred_grid_index[s][0:3]
            plotgrid_assigned[r, c] = 1
            axarr[1].imshow(plotgrid_assigned[:, :, 0], cmap='gray')  # only for debug
            axarr[1].set_title('Predicted assigment', color='0.7')

        plt.show()

    return [precission, recall, f1] ,[TP, FP, FN]

def grid_based_eval_with_iou(gt, pred, cfg, conf_thr= .51, debug = False):
    x = np.arange(-cfg.grich_anchor_pt, cfg.grich_anchor_pt, 1 / cfg.num_of_samples)


    gt_grid_index = []
    pred_grid_index = []
    gt_image_list = []
    # go through gt save all lines as list
    for k in range(0, cfg.grid_size):
        for l in range(0, cfg.grid_size):
            for a in range(cfg.num_prediction_cells):
                if gt[k, l, a, -1]:

                    x_t, y_f = plot_redy_line(x, gt[k, l, a, :-1], l, k ,cfg)
                    f = np.array([x_t, y_f]).T
                   # here are only the y values, the x values are linear from -anchor point to anchor point
                    gt_grid_index.append([k, l, a])
                    gt_image = np.zeros((cfg.img_w, cfg.img_h),
                                        dtype=np.float32)  # row # col https://en.wikipedia.org/wiki/Index_notation
                    gt_image = polylines(gt_image, np.int32([f]), 0, 1, thickness=cfg.thickness_of_lanes)
                    gt_image_list.append(gt_image)

    assignment_row = [] # this will be the assignment matrix
    for k in range(0, cfg.grid_size):
        for l in range(0, cfg.grid_size):
            for a in range(cfg.num_prediction_cells):

                a_pre, b_pre, c_pre, conf = pred[k, l, a, :]
                conf = sigmoid(conf)

                if conf >= conf_thr:
                    pred_grid_index.append([k, l, a])
                    assignment_col = []

                    a_pre = a_pre * cfg.a_range + cfg.a_shift
                    b_pre = b_pre * cfg.b_range + cfg.b_shift
                    c_pre = c_pre * cfg.c_range + cfg.c_shift

                    pred_image = np.zeros((cfg.img_w, cfg.img_h), dtype=np.float32)
                    y_fill = np.round(a_pre * x ** 2 + b_pre * x + c_pre)

                    x_t, y_fill = plot_redy_line(x, y_fill, l, k, cfg)

                    f2 = np.array([x_t, y_fill]).T
                    pred_image = polylines(pred_image, np.int32([f2]), 0, 1, thickness=cfg.thickness_of_lanes)

                    for gt_img in gt_image_list:
                        assignment_col.append(iou(gt_img, pred_image))
                    assignment_row.append(assignment_col)

    assignment_matrix = 1 - np.asarray(assignment_row)
    if assignment_matrix.ndim == 2:
        row_ind, col_ind = linear_sum_assignment(assignment_matrix)
        assignment_matrix = 1 - assignment_matrix
        # for debug reason plot matched gridbox
        if debug:
            plotgrid_gt = np.zeros((cfg.grid_size, cfg.grid_size, 1), dtype=np.float32)
            plotgrid_assigned = np.zeros((cfg.grid_size, cfg.grid_size, 1), dtype=np.float32)
            f, axarr = plt.subplots(1, 2)
            for k in range(0, cfg.grid_size):
                for l in range(0, cfg.grid_size):
                    for a in range(cfg.num_prediction_cells):
                        if gt[k, l, a, -1]:
                            plotgrid_gt[k, l] = 1
                            axarr[0].imshow(plotgrid_gt[:, :, 0], cmap='gray')  # only for debug
                            axarr[0].set_title('Ground Thruth', color='0.7')

            for s in row_ind:
                r, c=pred_grid_index[s][0:2]
                plotgrid_assigned[r, c] = 1
                axarr[1].imshow(plotgrid_assigned[:, :, 0], cmap='gray')  # only for debug
                axarr[1].set_title('Predicted assigment', color='0.7')

            plt.show()

        small = ev_iou_based_assignment(row_ind, col_ind, assignment_matrix, gt_grid_index, pred_grid_index, cfg.small_iou_Thr, cfg)
        medium = ev_iou_based_assignment(row_ind, col_ind, assignment_matrix, gt_grid_index, pred_grid_index, cfg.medium_iou_Thr, cfg)
        big = ev_iou_based_assignment(row_ind, col_ind, assignment_matrix, gt_grid_index, pred_grid_index, cfg.big_iou_Thr, cfg)
        # count for coefidenz matrix
        return small, medium, big
    else:
        return None

def nms(pred, cfg):
    x = np.arange(-cfg.grich_anchor_pt, cfg.grich_anchor_pt, 1 / cfg.num_of_samples)

   # pred_grid_index = []
    pred_grid_image_list = []
    # go through gt save all lines as list
    for k in range(0, cfg.grid_size):
        for l in range(0, cfg.grid_size):
            for a in range(cfg.num_prediction_cells):
                a_pre, b_pre, c_pre, conf = pred[k, l, a, :]
                conf = sigmoid(conf)

                if conf >= cfg.conf_thr:
                    a_pre = a_pre * cfg.a_range + cfg.a_shift
                    b_pre = b_pre * cfg.b_range + cfg.b_shift
                    c_pre = c_pre * cfg.c_range + cfg.c_shift

                    pred_image = np.zeros((cfg.img_w, cfg.img_h), dtype=np.float32)
                    y_fill = np.round(a_pre * x ** 2 + b_pre * x + c_pre)
                    x_t, y_fill = plot_redy_line(x, y_fill, l, k, cfg)
                    f = np.array([x_t, y_fill]).T
                    pred_image = polylines(pred_image, np.int32([f]), 0, 1, thickness=cfg.thickness_of_lanes)

                    pred_grid_image_list.append([k, l, a, conf, pred_image])

    pred_grid_image_list.sort(key=lambda b: b[3], reverse=True) # sort for conf for nms

    lenght_pred = len(pred_grid_image_list)
    for i in range(lenght_pred):
        pred_i = pred_grid_image_list[i]
        for j in range(i + 1, lenght_pred):
            boxj = pred_grid_image_list[j]
            if iou(pred_i[-1], boxj[-1]) > cfg.nms_iou_thrs:
                pred[boxj[0], boxj[1], boxj[2]].fill(0.)
            # boxj.c = 0.0

    return pred