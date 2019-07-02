
import os
path = os.getcwd()
from cu__grid_cell.data_gen import data_gen
from cu__grid_cell.preparation import preparation
import numpy as np
from cu__grid_cell.Validation.validation_utils import plot_image, grid_based_eval_with_iou, plot_image3d, nms, concatenate_cells
import matplotlib.pyplot as plt
import cv2




def sigmoid(x):
    return 1. / (1. + np.exp(-x))

batch = 2
model_obj = preparation(testing = True)
config = model_obj.config
a = data_gen(dataset=config.CU_test6_curve_hdf5_path, batchsize=batch, config=config, augment=False)
generator = a.batch_gen(test=True)
x_img, y, gt_image, gt_lanes = next(generator)
y = y[0]
concatenate_cells(y[0], config)
prediction = model_obj.predict(x_img)


scale_size_y =   (1640 -1) / config.img_w
scale_size_x =  (590 -1) /config.img_h

M = np.array([[scale_size_y, 0, 0],
                          [0, scale_size_x, 0],
                          [0, 0, 1.]])
M=M[0:2]


if config.splitted:
    lok = prediction[-2]
    conf = prediction[-1]
    prediction = np.concatenate([lok, conf], axis=-1)
#elif config.staged:
#    prediction = prediction[-1]

for i, s in enumerate(prediction):

    s = nms(s, config)
    plt.figure(1)
    #f, axarr = plt.subplots(1, 2)


    #axarr[0].imshow(gt_image[i,:,:,::-1].astype(np.uint8))
    #axarr[0].set_title('Ground Thruth', color='0.7')
    for a in gt_lanes[i]:
        gt_image[i] = cv2.polylines(gt_image[i], np.int32([a]), isClosed=0, color=(0, 255, 0), thickness=10)
    lanes_pred = concatenate_cells(s, config, prediction=True)
    original_points =  lanes_pred
    for j, o in enumerate(original_points):
        o = np.array(o).T
        ones = np.ones_like(o[:, 0])
        ones = ones[..., None]
        original_points[j] = np.concatenate((o, ones),
                                            axis=1)  # we reuse 3rd column in completely different way here, it is hack for matmul with M
        original_points[j] = np.matmul(M, original_points[j].T).T  # transpose for multiplikation

    lanes = original_points  # take only coords!
    for a in lanes:
        gt_image[i] = cv2.polylines(gt_image[i], np.int32([a]), isClosed=0,color=(0,0,255), thickness=10)

    #pred_img = plot_image(s, config, with_print=True, plot_image =x_img[i,:,:])

    plt.imshow(gt_image[i,:,:,::-1].astype(np.uint8))
  #  plt.set_title('Predicted', color='0.7')

    # now 3d plot

    plot_image3d(s, config, True, with_print=False)
  #  plot_image3d(y[i], config, False, with_print=False)

plt.show()

test = 0