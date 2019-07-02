
import os
path = os.getcwd()
from cu__grid_cell.data_gen import data_gen
from cu__grid_cell.preparation import preparation
import numpy as np
from cu__grid_cell.Validation.validation_utils_lane_only import plot_image, grid_based_eval_with_iou, plot_image3d, nms
import matplotlib.pyplot as plt



def sigmoid(x):
    return 1. / (1. + np.exp(-x))

batch = 1
model_obj = preparation()
config = model_obj.config
a = data_gen(dataset=config.CU_val_hdf5_path, batchsize=batch, config=config)
generator = a.batch_gen()
x_img, y = next(generator)
prediction = model_obj.predict(x_img)

prediction = prediction[-1]

for i, s in enumerate(prediction):
    s = nms(s, config)
    #iou = grid_based_eval_with_iou(y[i], s, config)
    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(x_img[i,:,:,0], cmap='gray')
    axarr[0].set_title('Ground Thruth', color='0.7')
    pred_img = plot_image(s, config, with_print=True)
    axarr[1].imshow(pred_img)
    axarr[1].set_title('Predicted', color='0.7')

    # now 3d plot

    plot_image3d(s, config, True, with_print=False)
    plot_image3d(y[i], config, False, with_print=False)

plt.show()

test = 0