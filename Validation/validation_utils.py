import random

import numpy as np
from cv2 import polylines, line, LINE_AA, bitwise_and, sumElems, resize
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import euclidean
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from cu__grid_cell.data_gen import data_gen


scale_size_y =   (1640 -1) / 368
scale_size_x =  (590 -1) / 368

M = np.array([[scale_size_y, 0, 0],
                          [0, scale_size_x, 0],
                          [0, 0, 1.]])
M=M[0:2]


def sigmoid(value):
    if -value > np.log(np.finfo(type(value)).max):
        return 0.0
    a = np.exp(-value)
    return 1.0/ (1.0 + a)

def plot_redy_line(x,grid, col, row, config):
    y_f = grid
    y_f, x_t = x.copy(), y_f.copy() # for ratoation
    x_t = np.array(x_t + col * (config.grid_cel_size)  + config.grich_anchor_pt)
    y_f = np.array(y_f + row * (config.grid_cel_size) + config.grich_anchor_pt)
    return x_t, y_f

def iterate_and_validate_predictions_cells(prediction, gt, cfg):
    x = np.arange(-cfg.grich_anchor_pt, cfg.grich_anchor_pt + 1, 1)
    y_tr_roc = []
    y_pr_roc = []
    error = []
    error_endpoints = []
   # gt = gt[-1]
   # gt = np.reshape(gt, (cfg.grid_size, cfg.grid_size,
   #                      cfg.num_prediction_cells, cfg.cell_size_grid))
    gt = gt[0]
    for i, s in enumerate(prediction):
        for k in range(0, cfg.grid_size):
            for l in range(0, cfg.grid_size):
                for a in range(cfg.num_prediction_cells):
                    a_pre, b_pre, c_pre, conf = s[k,l, :]

                    conf_true = gt[i, k, l, -1]

                    if conf_true:
                        y_tr_roc.append(True)
                        a_pre = a_pre * cfg.a_range #+ cfg.a_shift
                        b_pre = b_pre * cfg.b_range #+ cfg.b_shift
                        c_pre = c_pre * cfg.c_range #+ cfg.c_shift
                        y_fill = np.round(a_pre * x ** 2 + b_pre * x + c_pre)

                        x_t = gt[i, k, l, 0:cfg.grid_cel_size + 1]
                        y_f = gt[i, k, l, cfg.grid_cel_size + 1:
                                            2 * (cfg.grid_cel_size) + 2]
   #                     #y_f = np.round(a_pre * x ** 2 + b_pre * x + c_pre)
                        error.append((y_f -y_fill)**2) ## square error
                        error_endpoints.append([(y_f[0] -y_fill[0])**2, (y_f[-1] -y_fill[-1])**2])
                    else:
                        y_tr_roc.append(False)

                    conf = sigmoid(conf)
                    y_pr_roc.append(conf)

    y_tr_roc = np.asarray(y_tr_roc)
    y_pr_roc = np.asarray(y_pr_roc)
    error = np.asarray(error)
    error_endpoints = np.asarray(error_endpoints)
    return y_tr_roc, y_pr_roc, error, error_endpoints

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

def plot_image(pred, config, with_print= False, plot_image = []):
    if plot_image == []:
        plot_image = np.zeros((config.img_w, config.img_w), dtype=np.float32)

    x = np.arange(-config.grich_anchor_pt, config.grich_anchor_pt +1, 1)

    if with_print:
        print('-----------------------New Prediction---------------------------------------')
    #pred = pred[-1]
    #pred = np.reshape(pred, (config.grid_size, config.grid_size, config.num_prediction_cells, 4))

    for k in range(0, config.grid_size):
        for l in range(0, config.grid_size):
            for a in range(config.num_prediction_cells):
                a_ko, b, c, conf = pred[k, l, a, :]
                # sc = sigmoid(sc)
                conf = sigmoid(conf)
                if conf >= config.conf_thr:

                    a_ko = a_ko * config.a_range #+ config.a_shift
                    b = b * config.b_range #+ config.b_shift
                    c = c * config.c_range #+ config.c_shift

                    y_f = np.round(a_ko * x ** 2 + b * x + c)
                    y_f, x_t = x.copy(), y_f.copy()
                    x_t = np.array(x_t + l * config.grid_cel_size + config.grich_anchor_pt)
                    y_f = np.array(y_f + k * config.grid_cel_size + config.grich_anchor_pt)
                    f = np.array([x_t, y_f]).T

                    if with_print:
                        print('Score ' + ' a: ' + str(a_ko) + ' b: ' + str(b) + ' c: ' + str(c) + ' conf: ' + str(
                            conf))  # + ' Fit correct:' + str(fit_t) + ' Fit wrong:' + str(fit_f))


                    color = rgb_color(a)
                    #t = int( a + 1)
                    #plot_image = polylines(plot_image, np.int32([f]), 0, 1, (0,0,0), thickness=1 )
                    plot_image = polylines(plot_image, np.int32([f]), 0, 1, thickness=9)
                    #plot_image = polylines(plot_image, np.int32([f]), isClosed=0, color=color, thickness=1)
    if with_print:
        print('----------------------------------------------------------------------------')

    return plot_image

def draw_grid(img, line_color=(1, 1, 1), thickness=1, type_=LINE_AA, pxstep=16):
    '''(ndarray, 3-tuple, int, int) -> void
    draw gridlines on img
    line_color:
        BGR representation of colour
    thickness:
        line thickness
    type:
        8, 4 or cv2.LINE_AA
    pxstep:
        grid line frequency in pixels
    '''
    # TODO: make this comp. only ones at the beginning!
    x = 0
    y = 0
    while x < img.shape[1]:
        line(img, (x, 0), (x, img.shape[0]), color=line_color, lineType=type_, thickness=thickness)
        x += pxstep

    while y < img.shape[0]:
        line(img, (0, y), (img.shape[1], y), color=line_color, lineType=type_, thickness=thickness)
        y += pxstep

    line(img, (367, 0), (367, img.shape[0]), color=line_color, lineType=type_, thickness=thickness)
    line(img, (0, 367), (img.shape[1], 367), color=line_color, lineType=type_, thickness=thickness)

def plot_image3d(grid, config, is_prediction, with_print= False):
    x = np.arange(-config.grich_anchor_pt, config.grich_anchor_pt +1, 1)

    fig = plt.figure(0)
    if is_prediction:
        fig.canvas.set_window_title('Prediction')
    else:
        fig.canvas.set_window_title('Ground_Truth')
    ax = fig.gca(projection='3d')
   # ax.grid(True)

    grid_test = np.arange(0, config.img_w, 16)
    ax.set_xticks(grid_test, minor=False)
    ax.set_yticks(grid_test, minor=False)
    xx, yy = np.meshgrid(grid_test, grid_test)
    ax.plot(xx, yy,0, marker='.', color='k', linestyle='none')
   # ax.set_yticks(grid_test, minor=True)
    ax.xaxis.grid(True, which='major')
    ax.yaxis.grid(True, which='major')

    ax.set_xlabel('x Achse')
    ax.set_ylabel('y Achse')
    ax.set_zlabel('z Achse')
    #ax.set_zticks(grid_test, minor=False)

    heatmap = np.zeros([config.grid_size,config.grid_size])
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
                    heatmap[k, l] = conf
                    if conf >= config.conf_thr:
                        a_ko = a_ko * config.a_range #+ config.a_shift
                        b = b * config.b_range #+ config.b_shift
                        c = c * config.c_range #+ config.c_shift
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
                  #  grid = grid[-1]
                    grid = np.reshape(grid, (config.grid_size, config.grid_size,
                                             config.num_prediction_cells, config.cell_size_grid))
                    if grid[k, l, a,-1]:
                        x_t =  grid[k, l, a, 0:config.grid_cel_size + 1]
                        y_f = grid[k, l, a, config.grid_cel_size + 1:
                                                                2*(config.grid_cel_size) + 2]
                        x_t, y_f = plot_redy_line(x_t, y_f, l, k, config)
                        z = np.ones_like(x_t) * a
                        ax.plot(y_f, x_t, z, label='parametric curve')
                        counter_gt += 1
                        #ax.legend()

    print('Ground thruth counted: ' + str(counter_gt))
    print('Prediction counted: ' + str(counter_pred))
    if with_print:
        print('----------------------------------------------------------------------------')

    plt.figure(2)
    # fig = plt.figure()
    plt.xlabel('x Achse')
    plt.ylabel('y Achse')
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')

    plt.show()

    return 0

def iou(line1, line2):
    # assume that the line1 and 2 are [0, 1] images
    sum_line1 = np.sum(line1)   # with assumption sum is just only a count
    sum_line2 = np.sum(line2)

    inter_sum = sumElems(bitwise_and(line1,line2))[0]
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
    FP = pred_grid_index - TP
    FN = gt_grid_index - TP

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
    x = np.arange(-cfg.grich_anchor_pt, cfg.grich_anchor_pt +1, 1)

   # gt = gt[-1]
    gt = np.reshape(gt, (cfg.grid_size, cfg.grid_size,
                         cfg.num_prediction_cells, cfg.cell_size_grid))
   # pred = pred[-1]
   # pred = np.reshape(pred, (cfg.grid_size, cfg.grid_size,
   #                      cfg.num_prediction_cells, 4))

    gt_grid_index = []
    pred_grid_index = []
    gt_image_list = []
    # go through gt save all lines as list
    for k in range(0, cfg.grid_size):
        for l in range(0, cfg.grid_size):
            for a in range(cfg.num_prediction_cells):
                if gt[k, l, a, -1]:
                    x_t = gt[k, l, a, 0:cfg.grid_cel_size + 1]
                    y_f = gt[k, l, a, cfg.grid_cel_size + 1:
                                        2 * (cfg.grid_cel_size) + 2]
                    x_t, y_f = plot_redy_line(x_t, y_f, l, k, cfg)

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

                    a_pre = a_pre * cfg.a_range #+ cfg.a_shift
                    b_pre = b_pre * cfg.b_range #+ cfg.b_shift
                    c_pre = c_pre * cfg.c_range #+ cfg.c_shift

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
    x = np.arange(-cfg.grich_anchor_pt, cfg.grich_anchor_pt +1, 1)

   # pred_grid_index = []
    pred_grid_image_list = []
    pred = pred
    pred = np.reshape(pred, (cfg.grid_size, cfg.grid_size,
                         cfg.num_prediction_cells, 4))
    # go through gt save all lines as list
    for k in range(0, cfg.grid_size):
        for l in range(0, cfg.grid_size):
            for a in range(cfg.num_prediction_cells):
                a_pre, b_pre, c_pre, conf = pred[k, l, a, :]
                conf = sigmoid(conf)
                if conf >= cfg.conf_thr:
                    a_pre = a_pre * cfg.a_range #+ cfg.a_shift
                    b_pre = b_pre * cfg.b_range #+ cfg.b_shift
                    c_pre = c_pre * cfg.c_range #+ cfg.c_shift

                  #  pred_image = np.zeros((cfg.img_w, cfg.img_h), dtype=np.float32)
                    y_fill = np.round(a_pre * x ** 2 + b_pre * x + c_pre)
                    x_t, y_fill = plot_redy_line(x, y_fill, l, k, cfg)
                    f = np.array([x_t, y_fill]).T
                    f_mean = np.mean(f, axis=0)
                  #  pred_image = polylines(pred_image, np.int32([f]), 0, 1, thickness=cfg.thickness_of_lanes)
                 #   pred_image = resize(pred_image, (92, 92))
                    pred_grid_image_list.append([k, l, a, conf, f, f_mean])


    pred_grid_image_list.sort(key=lambda b: b[3], reverse=True) # sort for conf for nms




    lenght_pred = len(pred_grid_image_list)
    for i in range(lenght_pred):
        pred_i = pred_grid_image_list[i]
        for j in range(i + 1, lenght_pred):
            boxj = pred_grid_image_list[j]
            euclid_dist = np.linalg.norm(pred_i[-1]-boxj[-1])
            if euclid_dist < 25.:
                pred_image1 = np.zeros((cfg.img_w, cfg.img_h), dtype=np.float32)
                pred_image2 = np.zeros((cfg.img_w, cfg.img_h), dtype=np.float32)

                pred_image1 = polylines(pred_image1, np.int32([pred_i[-2]]), 0, 1, thickness=cfg.thickness_of_lanes)
                pred_image2 = polylines(pred_image2, np.int32([boxj[-2]]), 0, 1, thickness=cfg.thickness_of_lanes)

                if iou(pred_image1, pred_image2) > cfg.nms_iou_thrs:
                    pred[boxj[0], boxj[1], boxj[2]].fill(0.)
            # boxj.c = 0.0


    return pred

def concatenate_cells(grid, cfg, prediction = False, conf_th = 0):
    if not conf_th:
        conf_th = cfg.conf_thr
    lanes = []
    x = np.arange(-cfg.grich_anchor_pt, cfg.grich_anchor_pt + 1, 1)

    if len(grid.shape) < 4:
        if grid.shape[-1] == cfg.cell_size_grid * cfg.num_prediction_cells:
            grid = np.reshape(grid, (cfg.grid_size, cfg.grid_size,
                                     cfg.num_prediction_cells, cfg.cell_size_grid))
        else:
            grid = np.reshape(grid, (cfg.grid_size, cfg.grid_size,
                                     cfg.num_prediction_cells, 4))
    row_gather = np.empty((cfg.grid_size,cfg.grid_size, 8), dtype=object)
    first_row = True
    # go through gt save all lines as list
    if prediction:
        for k in reversed(range(0, cfg.grid_size)):
            for l in range(0, cfg.grid_size):
                col_candidates = []
                for a in range(cfg.num_prediction_cells):
                    a_pre, b_pre, c_pre, conf = grid[k, l, a, :]
                    conf = sigmoid(conf)
                    if conf >= conf_th:
                        a_pre = a_pre * cfg.a_range  # + cfg.a_shift
                        b_pre = b_pre * cfg.b_range  # + cfg.b_shift
                        c_pre = c_pre * cfg.c_range  # + cfg.c_shift

                       # pred_image = np.zeros((cfg.img_w, cfg.img_h), dtype=np.float32)
                        y_fill = np.round(a_pre * x ** 2 + b_pre * x + c_pre)
                        x_t, y_fill = plot_redy_line(x, y_fill, l, k, cfg)
                        row_gather[k, l, 0] = k
                        row_gather[k, l, 1] = l
                        row_gather[k, l, 2] = list(x_t)
                        row_gather[k, l, 3] = list(y_fill)
                        row_gather[k, l, 4] = x_t[0]
                        row_gather[k, l, 5] = y_fill[0]
                        row_gather[k, l, 6] = x_t[-1]
                        row_gather[k, l, 7] = y_fill[-1]

            if k != cfg.grid_size-1:
                # testi = row_gather[k+1,:][:,-1]
                row_first = np.where(row_gather[k + 1, :, 0])[0]
                row_second = np.where(row_gather[k, :, 0])[0]
                if row_first.any() and row_second.any():
                    row_first_endpoints = row_gather[k + 1, row_first]
                    row_second_endpoints = row_gather[k, row_second]
                    row_first_endpoints = row_first_endpoints[:, np.newaxis]
                    distanz_matrix = np.sqrt(
                        np.sum((row_second_endpoints[:, 6:8] - row_first_endpoints[:, :, 4:6]) ** 2,
                               axis=-1).astype(np.float64))
                    row_ind, col_ind = linear_sum_assignment(distanz_matrix)

                    thresholded_dist = np.where(distanz_matrix[row_ind, col_ind] <= 40.)

                    indeces_second = row_second_endpoints[col_ind[thresholded_dist]]
                    indeces_first = row_first_endpoints[row_ind[thresholded_dist], 0]

                    for s, f in zip(indeces_second, indeces_first):
                        row_gather[s[0], s[1], 2].extend(f[2])
                     #   row_gather[f[0], f[1], :].fill(None)

                        row_gather[s[0], s[1], 3].extend(f[3])
                        row_gather[f[0], f[1], :].fill(None)
    else:
        for k in reversed(range(0, cfg.grid_size)):
            for l in range(0, cfg.grid_size):
                for a in range(cfg.num_prediction_cells):
                    if grid[k, l, a, -1]:
                        x_t = grid[k, l, a, 0:cfg.grid_cel_size + 1]
                        y_f = grid[k, l, a, cfg.grid_cel_size + 1:
                                            2 * (cfg.grid_cel_size) + 2]
                        x_t, y_f = plot_redy_line(x_t, y_f, l, k, cfg)
                        row_gather[k, l, 0] = k
                        row_gather[k, l, 1] = l
                        row_gather[k, l, 2] = list(x_t)
                        row_gather[k, l, 3] = list(y_f)
                        row_gather[k, l, 4] = x_t[0]
                        row_gather[k, l, 5] = y_f[0]
                        row_gather[k, l, 6] = x_t[-1]
                        row_gather[k, l, 7] = y_f[-1]

            if k != cfg.grid_size - 1:
                #testi = row_gather[k+1,:][:,-1]
                row_first = np.where(row_gather[k+1,:,0])[0]
                row_second = np.where(row_gather[k, :,0])[0]
                if row_first.any() and row_second.any():
                    row_first_endpoints = row_gather[k+1,row_first]
                    row_second_endpoints = row_gather[k, row_second]
                    row_first_endpoints = row_first_endpoints[:, np.newaxis]
                    distanz_matrix = np.sqrt(np.sum((row_second_endpoints[:,6:8] - row_first_endpoints[:,:,4:6])**2, axis = -1).astype(np.float64))
                    row_ind, col_ind = linear_sum_assignment(distanz_matrix)

                    thresholded_dist = np.where(distanz_matrix[row_ind, col_ind] <= 60.)

                    indeces_second = row_second_endpoints[col_ind[thresholded_dist]]
                    indeces_first = row_first_endpoints[row_ind[thresholded_dist],0]

                    for s,f in zip(indeces_second, indeces_first):
                        row_gather[s[0], s[1], 2].extend(f[2])
                        row_gather[f[0], f[1], :].fill(None)

                        row_gather[s[0], s[1], 3].extend(f[3])
                        row_gather[f[0], f[1], :].fill(None)

    lanes_gather = np.where(row_gather[:, :,0])
    for i in range(len(lanes_gather[0])):

        lane_tmp = np.asarray(list(row_gather[lanes_gather][:,2:4][i]))
        if (lane_tmp.shape[1]) > (cfg.num_of_samples + 1) * 3:
            lanes.append(lane_tmp)

    lanes.sort(key=lambda b: b.shape[1], reverse=True)
    lanes = lanes[0:4]

    # testi =
    #lanes = np.array(lanes)
 #   endpoints = []
 #   starts = []
 #   for lan in lanes:
 #       endpoints.append([lan[0,-1],lan[1,-1]]) ## x,y
 #       starts.append([lan[0,0],lan[1,0]])

 #   endpoints = np.array(endpoints)
 #   starts = np.array(starts)
 #   starts = starts[:, np.newaxis]

 #   diff  = np.sqrt(np.sum(np.array(endpoints - starts)**2, axis = -1).astype(np.float64))
 #   np.fill_diagonal(diff, 999999)
 #   row_ind, col_ind = linear_sum_assignment(diff)

 #   thresholded_dist = np.where(diff[row_ind, col_ind] <= 22.7)
 #   if thresholded_dist:
 #       row_ind_thres = row_ind[thresholded_dist]
 #       col_ind_thres = col_ind[thresholded_dist]
 #   for s,f in zip(row_ind_thres, col_ind_thres):
 #       testi = np.where[col_ind_thres == s]
 #       lanes[s]
    return lanes

def grid_based_eval_with_iou(gt, pred, cfg, conf_thr= 0):
    if not conf_thr:
        conf_thr = cfg.conf_thr
    # 1640 x 590
    assignment_row = []
    gt_image_list = []
    for g in gt:
        #pred_image = np.zeros((1640, 590), dtype=np.float32)
        gt_image = np.zeros((590, 1640 ), dtype=np.float32)

        original_points = g

        o = np.array(original_points)
        ones = np.ones_like(o[:, 0])
        ones = ones[..., None]
        original_points = np.concatenate((o, ones),
                                            axis=1)  # we reuse 3rd column in completely different way here, it is hack for matmul with M
        original_points = np.matmul(M, original_points.T).T  # transpose for multiplikation

        g = original_points  # t
        #f = g.T
        gt_image = polylines(gt_image, np.int32([g]), 0, 1, thickness=30)
        gt_image_list.append(gt_image)
  #      plt.imshow(gt_image, cmap='gray')

  #  plt.show()

    for p in pred:
        pred_image = np.zeros((590, 1640 ), dtype=np.float32)
        #pred_image = np.zeros((368, 368), dtype=np.float32)
        f = p

        original_points = f

        o = np.array(original_points).T
        ones = np.ones_like(o[:, 0])
        ones = ones[..., None]
        original_points = np.concatenate((o, ones),
                                            axis=1)  # we reuse 3rd column in completely different way here, it is hack for matmul with M
        original_points = np.matmul(M, original_points.T).T  # transpose for multiplikation

        f = original_points  # t


        pred_image = polylines(pred_image, np.int32([f]), 0, 1, thickness=30)
        assignment_col = []

        for gt_img in gt_image_list:
            assignment_col.append(iou(gt_img, pred_image))
        assignment_row.append(assignment_col)

    assignment_matrix = np.asarray(assignment_row)

    if assignment_matrix.ndim == 2: #Todo: CAARE HERE!
        row_ind, col_ind = linear_sum_assignment(1.-assignment_matrix)



        small = ev_iou_based_assignment(row_ind, col_ind, assignment_matrix, assignment_matrix.shape[1], assignment_matrix.shape[0],
                                        cfg.small_iou_Thr, cfg)
        medium = ev_iou_based_assignment(row_ind, col_ind, assignment_matrix,assignment_matrix.shape[1], assignment_matrix.shape[0],
                                         cfg.medium_iou_Thr, cfg)
        big = ev_iou_based_assignment(row_ind, col_ind, assignment_matrix, assignment_matrix.shape[1], assignment_matrix.shape[0], cfg.big_iou_Thr,
                                      cfg)
        return small, medium, big
    else:
        return None



if __name__ == "__main__":
    import timeit
    #   start = timeit.default_timer()
    from config import *
    config = Config()
    a = data_gen(dataset=config.CU_val_hdf5_path, config = config)
    generator = a.batch_gen()
    x, y = next(generator)
    y = y[0]
    y = y[0]
    start = timeit.default_timer()
    y = nms(y, config)
    lanes = concatenate_cells(y, config)
    grid_based_eval_with_iou(lanes, lanes, config)
    #stop = timeit.default_timer()
    #print(stop - start)
    print('first-done!')
    x, y = next(generator)