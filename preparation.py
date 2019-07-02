import threading
import warnings
from multiprocessing.pool import ThreadPool
import concurrent.futures
import cu__grid_cell.grid_cell_model as sm
from glob import glob
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger, TensorBoard, TerminateOnNaN, Callback
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras.optimizers import Adam,SGD
from config import *
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from keras.utils.training_utils import multi_gpu_model
import multiprocessing
from cu__grid_cell.custom_loss import loss
from cu__grid_cell.staged_optimizer import MultiSGD, Adam_lr_mult
import itertools
import math

import pickle


#PIK = "DATA_small.dat"
#PIK = "DATA_small_2.dat"

PIK = "valDATA.dat"

#PIK = "Data_SMALL_1.dat"
#PIK = "valDATA_2.dat"
from cu__grid_cell.Validation.validation_utils import plot_image, grid_based_eval_with_iou, plot_image3d, nms, \
    concatenate_cells

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # removes annoying tensorflow warnings

def validate_data(pred, gt, cfg):
    small = []
    medium = []
    big = []
    for i, s in enumerate(pred):

        s = nms(s, cfg)
        lanes_pred = concatenate_cells(s, cfg, prediction=True)

        eval = grid_based_eval_with_iou(gt[i], lanes_pred, cfg)
        #lanes_gt = concatenate_cells(gt[i], cfg)

        if eval:
            small.append(eval[0])
            medium.append(eval[1])
            big.append(eval[2])
      #  else:
       #     small.append(0.)
       #     medium.append(0.)
       #     big.append(0.)
    #print("")

    return small, medium, big

def startval(pred, gt, cfg,TRAINING_LOG, epoche):
    number_of_testdata = len(pred)
    numb_of_prozesses = 2
    split_data = np.linspace(0, number_of_testdata, numb_of_prozesses + 1, dtype=np.int32)
    pool = ThreadPool(processes=numb_of_prozesses)
    async_result = []
    small = []
    medium = []
    big = []

    import timeit
    start = timeit.default_timer()

    if len(pred.shape) > 5:
        pred_data = np.array_split(pred[-1], numb_of_prozesses)
    else:
        pred_data = np.array_split(pred, numb_of_prozesses)

    gt_data = np.array_split(gt, numb_of_prozesses)

    future_to_DATA = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=numb_of_prozesses) as executor:
        for i in range(numb_of_prozesses):
            #         print(prediction.shape)

            future_to_DATA[executor.submit(validate_data, pred_data[i], gt_data[i], cfg)] = 'Thread' + str(i)

        for future in concurrent.futures.as_completed(future_to_DATA):
            data_id = future_to_DATA[future]
            try:
                data = future.result()
            except Exception as exc:
                print('%r generated an exception: %s' % (data_id, exc))
            else:
                print('Load data from worker:' + str(data_id))

                small.append(data[0])
                medium.append(data[1])
                big.append(data[2])

    print("Predicted Val_data!")
    stop = timeit.default_timer()

    print('TIME FOR EVAL:')
    print(stop - start)
    small = list(itertools.chain.from_iterable(small))
    medium = list(itertools.chain.from_iterable(medium))
    big = list(itertools.chain.from_iterable(big))

    thre = [.51, ]  # .55, .6,]

    print('Start Validation with threshold: ' + str(thre))
    model_metrics = []
    header = ["Epoche", "Conf_thr", "Small_Precission", "Small_Recall", "Small_F1",
              "Medium_Precission", "Medium_Recall", "Medium_F1",
              "Big_Precission", "Big_Recall", "Big_F1",
              ]

    #
    try:
        c_small = np.sum(np.asarray(small)[:, 1], axis=0)
        small_precission = np.divide(c_small[0], (c_small[0] + c_small[1]))
        small_recall = np.divide(c_small[0], (c_small[0] + c_small[2]))
        small_f1 = np.divide(2 * (small_precission * small_recall), (small_precission + small_recall))

        c_medium = np.sum(np.asarray(medium)[:, 1], axis=0)
        medium_precission = np.divide(c_medium[0], (c_medium[0] + c_medium[1]))
        medium_recall = np.divide(c_medium[0], (c_medium[0] + c_medium[2]))
        medium_f1 = np.divide(2 * (medium_precission * medium_recall), (medium_precission + medium_recall))

        c_big = np.sum(np.asarray(big)[:, 1], axis=0)
        bi_precission = np.divide(c_big[0], (c_big[0] + c_big[1]))
        big_recall = np.divide(c_big[0], (c_big[0] + c_big[2]))
        big_f1 = np.divide(2 * (bi_precission * big_recall), (bi_precission + big_recall))

        model_metrics += [(epoche, 1,
                           small_precission, small_recall, small_f1,
                           medium_precission, medium_recall, medium_f1,
                           bi_precission, big_recall, big_f1)]
    except:
        print('Weak model')

    if model_metrics:
        model_metrics = pd.DataFrame(model_metrics, columns=header)
        if not os.path.isfile('%s.val.csv' % TRAINING_LOG):
            with open('%s.val.csv' % TRAINING_LOG, 'a') as f:
                model_metrics.to_csv(f, sep="\t", float_format='%.4f', mode='a')
        else:
            with open('%s.val.csv' % TRAINING_LOG, 'a') as f:
                model_metrics.to_csv(f, sep="\t", float_format='%.4f', mode='a', header=False)
    else:
        print("did not save eval!")

class preparation:
    def __init__(self,testing=False, exp_id='', epoch=None):
        self.ex_name = Config.experiment_name
        self.DIR = Config.DIR
        self.splitted = Config.splitted
        self.config = Config()
        self.metrics_id = self.ex_name + "/" + exp_id + 'metric' if exp_id \
            else self.ex_name + "/" + self.ex_name + '_metric'
        self.weights_id = self.ex_name + "/" + exp_id + 'weights'if exp_id  \
            else self.ex_name + "/" + self.ex_name + '_weights'
        self.WEIGHT_DIR = self.DIR + self.weights_id
        self.WEIGHTS_SAVE = 'weights.{epoch:04d}.h5'
        self.TRAINING_LOG = self.DIR + self.ex_name + "/" + exp_id + 'training_log'
        self.TEST_LOG= self.DIR + self.ex_name + "/" + exp_id + 'test_log'
        self.LOGS_DIR_keras = self.DIR + self.metrics_id + 'log.csv'
        self.LOGS_DIR_tensorboard = self.DIR + self.metrics_id
        # load previous weights or vgg19 if this is the first run
        self.last_epoch, self.wfile = self.get_last_epoch_and_weights_file(epoch)

        if self.splitted:
            self.model = sm.get_train_model_splitted(self.config)
#            self.test_model = sm.get_test_model_splitted(self.config)
        elif self.config.staged:
           # if testing:
           #     self.model = sm.get_testing_model_staged(self.config)

           # else:
            self.model = sm.get_training_model_staged(self.config)
        else:
            self.model = sm.get_training_model(self.config)


        self.lr_mult = sm.get_lrmult(self.model)
        if self.wfile is not None:
            print("Loading %s ..." % self.wfile)
            if self.config.multi_gpu:
                self.model = multi_gpu_model(self.model, gpus=2)
            self.model.load_weights(self.wfile, by_name=True)
        else:
            if not self.config.mobile:
                if self.config.vgg19:
                    print("Loading vgg19 weights...")

                    vgg_model = VGG19(include_top=False, weights='imagenet')

                    from_vgg = dict()
                    from_vgg['conv1_1'] = 'block1_conv1'
                    from_vgg['conv1_2'] = 'block1_conv2'
                    from_vgg['conv2_1'] = 'block2_conv1'
                    from_vgg['conv2_2'] = 'block2_conv2'
                    from_vgg['conv3_1'] = 'block3_conv1'
                    from_vgg['conv3_2'] = 'block3_conv2'
                    from_vgg['conv3_3'] = 'block3_conv3'
                    from_vgg['conv3_4'] = 'block3_conv4'
                    from_vgg['conv4_1'] = 'block4_conv1'
                    from_vgg['conv4_2'] = 'block4_conv2'
                    from_vgg['conv4_3'] = 'block4_conv3'
                    from_vgg['conv4_4'] = 'block4_conv4'
                    from_vgg['conv5_1'] = 'block4_conv3'
                    from_vgg['conv5_2'] = 'block4_conv4'


                    for layer in self.model.layers:
                        if layer.name in from_vgg:
                            vgg_layer_name = from_vgg[layer.name]
                            layer.set_weights(vgg_model.get_layer(vgg_layer_name).get_weights())
                            print("Loaded VGG19 layer: " + vgg_layer_name)
                else:
                    vgg_model = VGG16(include_top=False, weights='imagenet')

                    from_vgg = dict()
                    from_vgg['conv1_1'] = 'block1_conv1'
                    from_vgg['conv1_2'] = 'block1_conv2'
                    from_vgg['conv2_1'] = 'block2_conv1'
                    from_vgg['conv2_2'] = 'block2_conv2'
                    from_vgg['conv3_1'] = 'block3_conv1'
                    from_vgg['conv3_2'] = 'block3_conv2'
                    from_vgg['conv3_3'] = 'block3_conv3'
                    from_vgg['conv4_1'] = 'block4_conv1'
                    from_vgg['conv4_2'] = 'block4_conv2'
                    from_vgg['conv4_3'] = 'block4_conv3'
                    from_vgg['conv5_1'] = 'block5_conv1'
                    from_vgg['conv5_2'] = 'block5_conv2'

                    for layer in self.model.layers:
                        if layer.name in from_vgg:
                            vgg_layer_name = from_vgg[layer.name]
                            layer.set_weights(vgg_model.get_layer(vgg_layer_name).get_weights())
                            print("Loaded VGG16 layer: " + vgg_layer_name)
                #    from_vgg = dict()
                #    from_vgg['conv1_1'] = 'block1_conv1'
                #    from_vgg['conv1_2'] = 'block1_conv2'
                #    from_vgg['conv2_1'] = 'block2_conv1'
                #    from_vgg['conv2_2'] = 'block2_conv2'

                   # for layer in self.model.layers:
                   #     if layer.name in from_vgg:
                   #         layer.trainable = False
                   #         vgg_layer_name = from_vgg[layer.name]
                   #         print("Frozen layer: " + vgg_layer_name)

            if self.config.multi_gpu:
                self.model = multi_gpu_model(self.model, gpus=2)



    def get_last_epoch_and_weights_file(self, epoch):

        os.makedirs(self.WEIGHT_DIR, exist_ok=True)

        if epoch is not None and epoch != '': #override
            return int(epoch),  self.WEIGHT_DIR + '/' + self.WEIGHTS_SAVE.format(epoch=epoch)

        files = [file for file in glob(self.WEIGHT_DIR + '/weights.*.h5')]
        files = [file.split('/')[-1] for file in files]
        epochs = [file.split('.')[1] for file in files if file]
        epochs = [int(epoch) for epoch in epochs if epoch.isdigit() ]
        if len(epochs) == 0:
            if 'weights.best.h5' in files:
                return -1, self.WEIGHT_DIR + '/weights.best.h5'
        else:
            ep = max([int(epoch) for epoch in epochs])
            return ep, self.WEIGHT_DIR + '/' + self.WEIGHTS_SAVE.format(epoch=ep)
        return 0, None

    def step_decay(self, epoch):
        drop = 0.9
        epochs_drop = 1.
        lrate = self.config.learn_rate * math.pow(drop,
        math.floor((epoch)/epochs_drop))
        print("Epoch:", epoch, "Learning rate:", lrate)
        return lrate

    def launchTensorBoard(self):
        os.system(
            "firefox http://ehealth-System-Product-Name:6006 \
                   && tensorboard --logdir=" + self.LOGS_DIR_tensorboard)

    def prep_for_training(self, config, train_obj, val_obj, loss_function_obj, epoch=None ):
        self.config = config
        self.train_obj = train_obj  # care! only class obj
        self.val_obj = val_obj    # care only data here
        #self.iterations_per_epoch =  int(self.train_obj.epoch /4. / self.config.batch_size)
        self.iterations_per_epoch =  int(np.floor(self.train_obj.epoch /4. / self.config.batch_size))

        print('loading Validation Data to RAM!')
        # get validation hack
        val_gen = val_obj.batch_gen_inplace(val=True)
        if os.path.isfile(PIK):
            with open(PIK, "rb") as f:
                print('loading File')
                temp_data = pickle.load(f)
        else:
            temp_data = next(val_gen)
            with open(PIK, "wb") as f:
                pickle.dump(temp_data, f,protocol=4)
                print('Saved data')

        self.val_data = temp_data[0:2]
        self.validation_data = temp_data[2]
        print('Val Data Loaded to RAM')

        self.loss_function_obj = loss_function_obj

        checkpoint = ModelCheckpoint(self.WEIGHT_DIR + '/' + self.WEIGHTS_SAVE, monitor='loss', verbose=1, save_best_only=True,
                                         save_weights_only=True, mode='min', period=1)
        csv_logger = CSVLogger(self.LOGS_DIR_keras, append=True, separator=';')
        tb = TensorBoard(log_dir=self.LOGS_DIR_tensorboard, write_graph=True, write_images=False, write_grads=False,
                         batch_size=config.batch_size)

        print("Weight decay policy...")
        for i in range(0, 40, 1):
            self.step_decay(i)

        lrate = LearningRateScheduler(self.step_decay)


        self.callbacks_list = [lrate, checkpoint, csv_logger, tb]

        if self.config.staged:

            self.loss_function_obj = loss_function_obj
       #     self.custom_los_obj1 = loss(config)
       #     self.custom_los_obj2 = loss(config)
           # self.custom_los_obj3 = loss(config)
           # self.custom_los_obj4 = loss(config)
           # self.custom_los_obj5 = loss(config)


            losses = {
                "output_stage1_L1": self.loss_function_obj.loss,
             #   "output_stage2_L1": self.custom_los_obj1.loss,
         #       "output_stage3_L1": self.custom_los_obj2.loss,
   #             "output_stage4_L1": self.custom_los_obj3.loss,
     #           "output_stage5_L1": custom_los_obj4.loss,
     #           "output_stage6_L1": custom_los_obj5.loss
            }
            metrics = {
                "output_stage1_L1": [ self.loss_function_obj.loc_loss, self.loss_function_obj.confidence_loss],
          #      "output_stage2_L1": [self.custom_los_obj1.loc_loss, self.custom_los_obj1.confidence_loss],
          #      "output_stage3_L1": [self.custom_los_obj2.loc_loss, self.custom_los_obj2.confidence_loss],
   #             "output_stage4_L1": [self.custom_los_obj3.loc_loss, self.custom_los_obj3.confidence_loss],
     #           "output_stage5_L1": [custom_los_obj4.loc_loss, custom_los_obj4.confidence_loss],
     #           "output_stage6_L1": [custom_los_obj5.loc_loss, custom_los_obj5.confidence_loss]
            }
            multisgd = MultiSGD(lr=config.learn_rate, momentum=config.momentum, nesterov=False,
                                lr_mult=self.lr_mult, decay=0.0)

            #sgd = SGD(lr=config.learn_rate, momentum=config.momentum, nesterov=False)

          #  adam_lr = Adam_lr_mult(lr=config.learn_rate, multipliers=self.lr_mult, amsgrad=True)
          #  from keras.utils import plot_model
          #  plot_model(self.model, to_file='model.png')
            self.model.compile(loss=losses,
                            metrics=metrics,
                            optimizer=multisgd)
        else:
            losses = {
                "object_detection": loss_function_obj.loss,
                "segmentation": loss_function_obj.loss_nb,
            }
            metrics = {
                "object_detection": [loss_function_obj.loc_loss, loss_function_obj.confidence_loss],
                "segmentation": loss_function_obj.loss_segmentation,
            }
            self.model.compile(loss=losses,
                            metrics=metrics,
                            optimizer=Adam(lr=config.learn_rate))

    def prep_for_training_splitted(self, config, train_obj, val_obj, loss_function_obj, epoch=None ):
        self.config = config
        self.train_obj = train_obj  # care! only class obj
        self.val_obj = val_obj    # care only data here
        #self.iterations_per_epoch =  int(self.train_obj.epoch /4. / self.config.batch_size)
        self.iterations_per_epoch =  int(np.floor(self.train_obj.epoch /4. / self.config.batch_size))

        print('loading Validation Data to RAM!')
        # get validation hack
        val_gen = val_obj.batch_gen_inplace(val=True)
        if os.path.isfile(PIK):
            with open(PIK, "rb") as f:
                print('loading File')
                temp_data = pickle.load(f)
        else:
            temp_data = next(val_gen)
            with open(PIK, "wb") as f:
                pickle.dump(temp_data, f,protocol=4)
                print('Saved data')

        self.val_data = temp_data[0:2]
        self.validation_data = temp_data[2]
        print('Val Data Loaded to RAM')

        self.loss_function_obj = loss_function_obj

        checkpoint = ModelCheckpoint(self.WEIGHT_DIR + '/' + self.WEIGHTS_SAVE, monitor='loss', verbose=1, save_best_only=True,
                                         save_weights_only=True, mode='min', period=1)
        csv_logger = CSVLogger(self.LOGS_DIR_keras, append=True, separator=';')
        tb = TensorBoard(log_dir=self.LOGS_DIR_tensorboard, write_graph=True, write_images=False, write_grads=False,
                         batch_size=config.batch_size)

        print("Weight decay policy...")
        for i in range(0, 40, 1):
            self.step_decay(i)

        lrate = LearningRateScheduler(self.step_decay)


        self.callbacks_list = [lrate, checkpoint, csv_logger, tb]


        self.loss_function_obj1 = loss_function_obj
        self.loss_function_obj2 = loss_function_obj
        self.custom_los_obj1 = loss(config)
        #self.custom_los_obj2 = loss(config)
       # self.custom_los_obj3 = loss(config)
       # self.custom_los_obj4 = loss(config)
       # self.custom_los_obj5 = loss(config)


        losses = {
            "output_stage1_L0": self.loss_function_obj1.loss_LOK,
            "output_stage1_L1": self.loss_function_obj1.loss_KONF,
            "output_stage2_L0": self.custom_los_obj1.loss_LOK,
            "output_stage2_L1": self.custom_los_obj1.loss_KONF,
#            "output_stage3_L1": self.custom_los_obj2.loss,
#             "output_stage4_L1": self.custom_los_obj3.loss,
 #           "output_stage5_L1": custom_los_obj4.loss,
 #           "output_stage6_L1": custom_los_obj5.loss
        }
        metrics = {
            "output_stage1_L0": [self.loss_function_obj.loc_loss],
            "output_stage1_L1":  [self.loss_function_obj.confidence_loss, self.loss_function_obj.lossTRUE,self.loss_function_obj.lossFALSE],
            "output_stage2_L0": [self.loss_function_obj.loc_loss],
            "output_stage2_L1": [self.loss_function_obj.confidence_loss, self.loss_function_obj.lossTRUE,
                                 self.loss_function_obj.lossFALSE],
#             "output_stage2_L1": [self.custom_los_obj1.loc_loss, self.custom_los_obj1.confidence_loss],
#             "output_stage3_L1": [self.custom_los_obj2.loc_loss, self.custom_los_obj2.confidence_loss],
#             "output_stage4_L1": [self.custom_los_obj3.loc_loss, self.custom_los_obj3.confidence_loss],
 #           "output_stage5_L1": [custom_los_obj4.loc_loss, custom_los_obj4.confidence_loss],
 #           "output_stage6_L1": [custom_los_obj5.loc_loss, custom_los_obj5.confidence_loss]
        }
        multisgd = MultiSGD(lr=config.learn_rate, momentum=config.momentum, nesterov=False,
                            lr_mult=self.lr_mult, decay=0.0)
        self.model.compile(loss=losses,
                        #metrics=metrics,
                        optimizer=multisgd)

    def validate_hole_lane_multiprozessed(self, epoche):
        cfg = self.config
        prediction = self.model.predict(self.val_data[0])

        number_of_testdata = len(self.val_data[0])
        numb_of_prozesses = 4
        split_data = np.linspace(0, number_of_testdata, numb_of_prozesses + 1, dtype=np.int32)
        # pool = ThreadPool(processes=numb_of_prozesses)

        testi = []
        async_result = []
        small = []
        medium = []
        big = []

        import timeit
        start = timeit.default_timer()
        if self.config.splitted:
            lok = prediction[-2]
            conf = prediction[-1]
            pred_data = np.concatenate([lok, conf], axis=-1)
            pred_data = np.array_split(pred_data, numb_of_prozesses)
        elif len(prediction) > 1:
            pred_data = np.array_split(prediction, numb_of_prozesses)
        else:
            pred_data = np.array_split(prediction, numb_of_prozesses)

        gt_data = np.array_split(self.validation_data, numb_of_prozesses)

        future_to_DATA = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=numb_of_prozesses) as executor:
            for i in range(numb_of_prozesses):
                #         print(prediction.shape)

                future_to_DATA[executor.submit(validate_data, pred_data[i], gt_data[i], cfg)] = 'Thread' + str(i)

            for future in concurrent.futures.as_completed(future_to_DATA):
                data_id = future_to_DATA[future]
                try:
                    data = future.result()
                except Exception as exc:
                    print('%r generated an exception: %s' % (data_id, exc))
                else:
                    print('Load data from worker:' + str(data_id))

                    small.append(data[0])
                    medium.append(data[1])
                    big.append(data[2])

        print("Predicted Val_data!")

        stop = timeit.default_timer()

        print('TIME FOR EVAL:')
        print(stop - start)
        small = list(itertools.chain.from_iterable(small))
        medium = list(itertools.chain.from_iterable(medium))
        big = list(itertools.chain.from_iterable(big))

        thre = [.51, ]  # .55, .6,]

        print('Start Validation with threshold: ' + str(thre))
        model_metrics = []
        header = ["Epoche", "Conf_thr", "Small_Precission", "Small_Recall", "Small_F1",
                  "Medium_Precission", "Medium_Recall", "Medium_F1",
                  "Big_Precission", "Big_Recall", "Big_F1",
                  ]

        #
        try:
            c_small = np.sum(np.asarray(small)[:, 1], axis=0)
            small_precission = np.divide(c_small[0], (c_small[0] + c_small[1]))
            small_recall = np.divide(c_small[0], (c_small[0] + c_small[2]))
            small_f1 = np.divide(2 * (small_precission * small_recall), (small_precission + small_recall))

            c_medium = np.sum(np.asarray(medium)[:, 1], axis=0)
            medium_precission = np.divide(c_medium[0], (c_medium[0] + c_medium[1]))
            medium_recall = np.divide(c_medium[0], (c_medium[0] + c_medium[2]))
            medium_f1 = np.divide(2 * (medium_precission * medium_recall), (medium_precission + medium_recall))

            c_big = np.sum(np.asarray(big)[:, 1], axis=0)
            bi_precission = np.divide(c_big[0], (c_big[0] + c_big[1]))
            big_recall = np.divide(c_big[0], (c_big[0] + c_big[2]))
            big_f1 = np.divide(2 * (bi_precission * big_recall), (bi_precission + big_recall))

            print('[PREDICTION Epoch: ]'+ str(epoche) + '| F1(0,3): '+ str(small_f1) + 'F1(0,5): '+ str(medium_f1) +  'F1(0,7): '+ str(big_f1))
            model_metrics += [(epoche, 1,
                               small_precission, small_recall, small_f1,
                               medium_precission, medium_recall, medium_f1,
                               bi_precission, big_recall, big_f1)]
        except:
            print('Weak model')

        if model_metrics:
            model_metrics = pd.DataFrame(model_metrics, columns=header)
            if not os.path.isfile('%s.val.csv' % self.TRAINING_LOG):
                with open('%s.val.csv' % self.TRAINING_LOG, 'a') as f:
                    model_metrics.to_csv(f, sep="\t", float_format='%.4f', mode='a')
            else:
                with open('%s.val.csv' % self.TRAINING_LOG, 'a') as f:
                    model_metrics.to_csv(f, sep="\t", float_format='%.4f', mode='a', header=False)
        else:
            print("did not save eval!")

    def predict(self, data):
        return self.model.predict(data)  # makes a x,y prediction!

    def get_working_DIR(self):
        return self.DIR + self.ex_name

    def validate(self, epoche):
        cfg = self.config
        prediction = self.model.predict(self.val_data[0])
        thre = [.51,] # .55, .6,]
        print('Start Validation with threshold: ' + str(thre))
        model_metrics = []
        header = ["Epoche", "Conf_thr","Small_Precission", "Small_Recall", "Small_F1",
                                         "Medium_Precission", "Medium_Recall", "Medium_F1",
                                         "Big_Precission", "Big_Recall", "Big_F1",
                                         ]
        for t in thre: # TODO: change this very leasy approach!
            small = []
            medium = []
            big = []
            for i, s in enumerate(tqdm(prediction[-1])):
                s = nms(s, cfg)
                #[small, medium, big] -> [[precission, recall, f1] ,[TP, FP, FN]]
                eval = grid_based_eval_with_iou(self.val_data[1][-1][i], s, cfg, conf_thr=t)
                if eval:
                    small.append(eval[0])
                    medium.append(eval[1])
                    big.append(eval[2])

            try:
              #  metrics_small = np.mean(np.asarray(small)[:,0], axis=0)
              #  metrics_medium = np.mean(np.asarray(medium)[:, 0], axis=0)
              #  metrics_big = np.mean(np.asarray(big)[:, 0], axis=0)

                c_small = np.sum(np.asarray(small)[:,1], axis=0)
                small_precission = np.divide( c_small[0],  (c_small[0] + c_small[1]))
                small_recall = np.divide( c_small[0],  (c_small[0] + c_small[2]))
                small_f1 = np.divide(2*(small_precission* small_recall ), (small_precission + small_recall))

                c_medium = np.sum(np.asarray(medium)[:, 1], axis=0)
                medium_precission = np.divide( c_medium[0],  (c_medium[0] + c_medium[1]))
                medium_recall = np.divide( c_medium[0],  (c_medium[0] + c_medium[2]))
                medium_f1 = np.divide(2*(medium_precission* medium_recall ), (medium_precission + medium_recall))

                c_big = np.sum(np.asarray(big)[:, 1], axis=0)
                bi_precission = np.divide(c_big[0], (c_big[0] + c_big[1]))
                big_recall = np.divide(c_big[0], (c_big[0] + c_big[2]))
                big_f1 = np.divide(2 * (bi_precission * big_recall), (bi_precission + big_recall))

                model_metrics += [(epoche, t,
                                   small_precission, small_recall, small_f1,
                                   medium_precission, medium_recall, medium_f1,
                                   bi_precission, big_recall, big_f1)]

               # model_metrics += [(epoche, t, metrics_small[0], metrics_small[1], metrics_small[2],
               #                    metrics_medium[0], metrics_medium[1], metrics_medium[2],
               #                    metrics_big[0], metrics_big[1], metrics_big[2],
               #                    small_precission, small_recall, small_f1,
               #                    medium_precission, medium_recall, medium_f1,
               #                    bi_precission, big_recall, big_f1)]
            except:
                print('Weak model')


       # model_metrics = thre # , metrics_medium, metrics_big, cases_conter_small, cases_conter_medium, cases_conter_big)
        if model_metrics:
            model_metrics = pd.DataFrame(model_metrics, columns=header)
            if not os.path.isfile('%s.val.csv' % self.TRAINING_LOG):
                with open('%s.val.csv' % self.TRAINING_LOG, 'a') as f:
                    model_metrics.to_csv(f, sep="\t", float_format='%.4f', mode='a')
            else:
                with open('%s.val.csv' % self.TRAINING_LOG, 'a') as f:
                    model_metrics.to_csv(f, sep="\t", float_format='%.4f', mode='a',header=False)
        else:
            print("did not save eval!")

    def validate_hole_lane(self, epoche):
        cfg = self.config
        prediction = self.model.predict(self.val_data[0])
        thre = [.51, ]  # .55, .6,]
        print('Start Validation with threshold: ' + str(thre))
        model_metrics = []
        header = ["Epoche", "Conf_thr", "Small_Precission", "Small_Recall", "Small_F1",
                  "Medium_Precission", "Medium_Recall", "Medium_F1",
                  "Big_Precission", "Big_Recall", "Big_F1",
                  ]
        for t in thre:  # TODO: change this very leasy approach!
            small = []
            medium = []
            big = []
            for i, s in enumerate(tqdm(prediction[-1])):
                s = nms(s, cfg)
                lanes_pred = concatenate_cells(s, cfg, prediction=True)
                lanes_gt = concatenate_cells(self.val_data[1][-1][i], cfg)
                eval = grid_based_eval_with_iou(lanes_gt, lanes_pred, cfg)
                if eval:
                    small.append(eval[0])
                    medium.append(eval[1])
                    big.append(eval[2])


            try:
                c_small = np.sum(np.asarray(small)[:, 1], axis=0)
                small_precission = np.divide(c_small[0], (c_small[0] + c_small[1]))
                small_recall = np.divide(c_small[0], (c_small[0] + c_small[2]))
                small_f1 = np.divide(2 * (small_precission * small_recall), (small_precission + small_recall))

                c_medium = np.sum(np.asarray(medium)[:, 1], axis=0)
                medium_precission = np.divide(c_medium[0], (c_medium[0] + c_medium[1]))
                medium_recall = np.divide(c_medium[0], (c_medium[0] + c_medium[2]))
                medium_f1 = np.divide(2 * (medium_precission * medium_recall), (medium_precission + medium_recall))

                c_big = np.sum(np.asarray(big)[:, 1], axis=0)
                bi_precission = np.divide(c_big[0], (c_big[0] + c_big[1]))
                big_recall = np.divide(c_big[0], (c_big[0] + c_big[2]))
                big_f1 = np.divide(2 * (bi_precission * big_recall), (bi_precission + big_recall))

                model_metrics += [(epoche, t,
                                   small_precission, small_recall, small_f1,
                                   medium_precission, medium_recall, medium_f1,
                                   bi_precission, big_recall, big_f1)]
            except:
                print('Weak model')
        if model_metrics:
            model_metrics = pd.DataFrame(model_metrics, columns=header)
            if not os.path.isfile('%s.val.csv' % self.TRAINING_LOG):
                with open('%s.val.csv' % self.TRAINING_LOG, 'a') as f:
                    model_metrics.to_csv(f, sep="\t", float_format='%.4f', mode='a')
            else:
                with open('%s.val.csv' % self.TRAINING_LOG, 'a') as f:
                    model_metrics.to_csv(f, sep="\t", float_format='%.4f', mode='a', header=False)
        else:
            print("did not save eval!")


    def test_val(self, val_obj, batch):
        cfg = self.config
        val_gen = val_obj.batch_gen_inplace(val=True)
        numb_batches =  int(np.floor(val_obj.epoch / batch))
        small = []
        medium = []
        big = []

        for i in tqdm(range(numb_batches)):
            val_data_temp = next(val_gen)

            self.val_data = val_data_temp[0:2]
            self.validation_data = val_data_temp[2]

            prediction = self.model.predict(self.val_data[0])

            number_of_testdata = len(self.val_data[0])
            numb_of_prozesses= 4
            split_data = np.linspace(0, number_of_testdata, numb_of_prozesses+1, dtype=np.int32)
           # pool = ThreadPool(processes=numb_of_prozesses)


            import timeit
            start = timeit.default_timer()
            if self.config.splitted:
                lok = prediction[-2]
                conf =prediction[-1]
                pred_data = np.concatenate([lok,conf], axis = -1)
                pred_data = np.array_split(pred_data, numb_of_prozesses)
            elif self.config.staged:
                pred_data = np.array_split(prediction,numb_of_prozesses )
            else:
                pred_data = np.array_split(prediction,numb_of_prozesses )

            gt_data = np.array_split(self.validation_data, numb_of_prozesses)

            future_to_DATA = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=numb_of_prozesses) as executor:
                for i in range(numb_of_prozesses):
           #         print(prediction.shape)

                    future_to_DATA[executor.submit(validate_data ,pred_data[i], gt_data[i], cfg)] = 'Thread'+str(i)

                for future in concurrent.futures.as_completed(future_to_DATA):
                    data_id = future_to_DATA[future]
                    try:
                        data = future.result()
                    except Exception as exc:
                        print('%r generated an exception: %s' % (data_id, exc))
                    else:
                        print('Load data from worker:' +str(data_id))

                        small.append(data[0])
                        medium.append(data[1])
                        big.append(data[2])

        print("Predicted Val_data!")


        stop = timeit.default_timer()

        print('TIME FOR EVAL:')
        print(stop - start)
        small = list(itertools.chain.from_iterable(small))
        medium = list(itertools.chain.from_iterable(medium))
        big = list(itertools.chain.from_iterable(big))


        thre = [.51, ]  # .55, .6,]


        print('Start Validation with threshold: ' + str(thre))
        model_metrics = []
        header = ["Epoche", "Conf_thr", "Small_Precission", "Small_Recall", "Small_F1",
                  "Medium_Precission", "Medium_Recall", "Medium_F1",
                  "Big_Precission", "Big_Recall", "Big_F1",
                  ]

#
        try:
            c_small = np.sum(np.asarray(small)[:, 1], axis=0)
            small_precission = np.divide(c_small[0], (c_small[0] + c_small[1]))
            small_recall = np.divide(c_small[0], (c_small[0] + c_small[2]))
            small_f1 = np.divide(2 * (small_precission * small_recall), (small_precission + small_recall))

            c_medium = np.sum(np.asarray(medium)[:, 1], axis=0)
            medium_precission = np.divide(c_medium[0], (c_medium[0] + c_medium[1]))
            medium_recall = np.divide(c_medium[0], (c_medium[0] + c_medium[2]))
            medium_f1 = np.divide(2 * (medium_precission * medium_recall), (medium_precission + medium_recall))

            c_big = np.sum(np.asarray(big)[:, 1], axis=0)
            bi_precission = np.divide(c_big[0], (c_big[0] + c_big[1]))
            big_recall = np.divide(c_big[0], (c_big[0] + c_big[2]))
            big_f1 = np.divide(2 * (bi_precission * big_recall), (bi_precission + big_recall))

            model_metrics += [(0, 1,
                               small_precission, small_recall, small_f1,
                               medium_precission, medium_recall, medium_f1,
                               bi_precission, big_recall, big_f1)]
        except:
            print('Weak model')

        if model_metrics:
            model_metrics = pd.DataFrame(model_metrics, columns=header)
            if not os.path.isfile('%s.val.csv' % self.TEST_LOG):
                with open('%s.val.csv' % self.TEST_LOG, 'a') as f:
                    model_metrics.to_csv(f, sep="\t", float_format='%.4f', mode='a')
            else:
                with open('%s.val.csv' % self.TEST_LOG, 'a') as f:
                    model_metrics.to_csv(f, sep="\t", float_format='%.4f', mode='a', header=False)
        else:
            print("did not save eval!")

        def validate_hole_lane_multiprozessed(self, epoche):
            cfg = self.config
            prediction = self.model.predict(self.val_data[0])

            number_of_testdata = len(self.val_data[0])
            numb_of_prozesses = 4
            split_data = np.linspace(0, number_of_testdata, numb_of_prozesses + 1, dtype=np.int32)
            # pool = ThreadPool(processes=numb_of_prozesses)

            testi = []
            async_result = []
            small = []
            medium = []
            big = []

            import timeit
            start = timeit.default_timer()
            if self.config.splitted:
                lok = prediction[-2]
                conf = prediction[-1]
                pred_data = np.concatenate([lok, conf], axis=-1)
                pred_data = np.array_split(pred_data, numb_of_prozesses)
            elif self.config.staged:
                pred_data = np.array_split(prediction[-1], numb_of_prozesses)
            else:
                pred_data = np.array_split(prediction, numb_of_prozesses)

            gt_data = np.array_split(self.validation_data, numb_of_prozesses)

            future_to_DATA = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=numb_of_prozesses) as executor:
                for i in range(numb_of_prozesses):
                    #         print(prediction.shape)

                    future_to_DATA[executor.submit(validate_data, pred_data[i], gt_data[i], cfg)] = 'Thread' + str(i)

                for future in concurrent.futures.as_completed(future_to_DATA):
                    data_id = future_to_DATA[future]
                    try:
                        data = future.result()
                    except Exception as exc:
                        print('%r generated an exception: %s' % (data_id, exc))
                    else:
                        print('Load data from worker:' + str(data_id))

                        small.append(data[0])
                        medium.append(data[1])
                        big.append(data[2])

            print("Predicted Val_data!")

            stop = timeit.default_timer()

            print('TIME FOR EVAL:')
            print(stop - start)
            small = list(itertools.chain.from_iterable(small))
            medium = list(itertools.chain.from_iterable(medium))
            big = list(itertools.chain.from_iterable(big))

            thre = [.51, ]  # .55, .6,]

            print('Start Validation with threshold: ' + str(thre))
            model_metrics = []
            header = ["Epoche", "Conf_thr", "Small_Precission", "Small_Recall", "Small_F1",
                      "Medium_Precission", "Medium_Recall", "Medium_F1",
                      "Big_Precission", "Big_Recall", "Big_F1",
                      ]

            #
            try:
                c_small = np.sum(np.asarray(small)[:, 1], axis=0)
                small_precission = np.divide(c_small[0], (c_small[0] + c_small[1]))
                small_recall = np.divide(c_small[0], (c_small[0] + c_small[2]))
                small_f1 = np.divide(2 * (small_precission * small_recall), (small_precission + small_recall))

                c_medium = np.sum(np.asarray(medium)[:, 1], axis=0)
                medium_precission = np.divide(c_medium[0], (c_medium[0] + c_medium[1]))
                medium_recall = np.divide(c_medium[0], (c_medium[0] + c_medium[2]))
                medium_f1 = np.divide(2 * (medium_precission * medium_recall), (medium_precission + medium_recall))

                c_big = np.sum(np.asarray(big)[:, 1], axis=0)
                bi_precission = np.divide(c_big[0], (c_big[0] + c_big[1]))
                big_recall = np.divide(c_big[0], (c_big[0] + c_big[2]))
                big_f1 = np.divide(2 * (bi_precission * big_recall), (bi_precission + big_recall))

                model_metrics += [(epoche, 1,
                                   small_precission, small_recall, small_f1,
                                   medium_precission, medium_recall, medium_f1,
                                   bi_precission, big_recall, big_f1)]
            except:
                print('Weak model')

            if model_metrics:
                model_metrics = pd.DataFrame(model_metrics, columns=header)
                if not os.path.isfile('%s.val.csv' % self.TRAINING_LOG):
                    with open('%s.val.csv' % self.TRAINING_LOG, 'a') as f:
                        model_metrics.to_csv(f, sep="\t", float_format='%.4f', mode='a')
                else:
                    with open('%s.val.csv' % self.TRAINING_LOG, 'a') as f:
                        model_metrics.to_csv(f, sep="\t", float_format='%.4f', mode='a', header=False)
            else:
                print("did not save eval!")




    def validate_hole_lane_multi(self, epoche):
        cfg = self.config
        prediction = self.model.predict(self.val_data[0])
        t = threading.Thread(target=startval, args=(prediction, self.validation_data,cfg,self.TRAINING_LOG, epoche))
        t.start()

    def train(self):
        print("last_epoch:", self.last_epoch)
        #train_gen = self.train_obj.batch_gen()
        train_gen = self.train_obj
        tensorb = False
        for epoch in range(self.last_epoch, self.config.epochs):
            # train for one iteration
            self.model.fit_generator(train_gen,
                               #steps_per_epoch=1,
                                verbose=1,
                              #  steps_per_epoch= self.iterations_per_epoch,
                                #use_multiprocessing=False,
                                epochs=epoch + 1,
                                callbacks=self.callbacks_list,
                                initial_epoch=epoch,
                                validation_data=self.val_data,
                                validation_steps=1,
                                max_queue_size=20,
                                workers=2
                                )

     #       if not tensorb:
    #            try:
    #                t = threading.Thread(target=self.launchTensorBoard, args=([]))
    #                t.start()
    #            except:
    #                print("")
    #            tensorb = True

          #          'Start VALIDATION!___________________________________________________________________________________')

            self.validate_hole_lane_multiprozessed(epoch)
        #self.validate(3)
        print("Finished Training!")