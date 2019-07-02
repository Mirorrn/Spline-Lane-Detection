
import os
path = os.getcwd()

from sklearn.metrics import roc_curve, auc
from cu__grid_cell.data_gen import *
from cu__grid_cell.Validation.validation_utils import *
from cu__grid_cell.preparation import *

def print_roc(y_tr_roc, y_pr_roc):
    fpr, tpr, thresholds = roc_curve(y_tr_roc, y_pr_roc)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC (Fläche = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Falsch-Positiv-Rate')
    plt.ylabel('Richtig-Positiv-Rate')
    #plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")

    # create the axis of thresholds (scores)
    ax2 = plt.gca().twinx()
    ax2.plot(fpr, thresholds, markeredgecolor='r', linestyle='dashed', color='r')
    ax2.set_ylabel('Schwellwert', color='r')
    # ax2.set_ylim([thresholds[-1],thresholds[0]])
    # ax2.set_xlim([fpr[0],fpr[-1]])
    plt.show()

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

batch = 1000

model_obj = preparation()
config = model_obj.config

a  = data_gen(dataset=config.CU_val_hdf5_path,shuffle=False, augment=False, config = config, batchsize = batch, percentage_of_data=config.val_sample_percentage)# care loads howl dataset into ram!

generator = a.batch_gen_inplace()
x_img, y = next(generator)

prediction = model_obj.predict(x_img)

if config.splitted:
    lok = prediction[-2]
    conf = prediction[-1]
    prediction = np.concatenate([lok, conf], axis=-1)

elif len(prediction.shape) > 5:
    prediction = prediction[-1]
else:
    pred_data = prediction

y_tr_roc, y_pr_roc, error, error_endpoints = iterate_and_validate_predictions_cells(prediction, y, config)


class_report(y_tr_roc, y_pr_roc, 0.51)

mean_error = np.mean(error)
error_endpoints = np.mean(error_endpoints, axis=1)
print('Mean error:' + str(mean_error) )


print('Mean error endpoints:' + str(error_endpoints) )
print_roc(y_tr_roc, y_pr_roc)

print("Finished!")