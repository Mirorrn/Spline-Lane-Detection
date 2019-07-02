import copy
import pickle

import h5py
from cu__grid_cell.augmenter import *
from operator import itemgetter
import json
import math
import warnings
from scipy.optimize import linear_sum_assignment
import scipy.ndimage as ndimage
from scipy.interpolate import CubicSpline
from scipy import interpolate
import keras

class DataGenerator(keras.utils.Sequence):
    def __init__(self, config, dataset, batchsize=None, shuffle=True, augment=True, percentage_of_data=1.):
        self.config = config
        self.image_w = config.img_w
        self.image_h = config.img_h

        try:
            self.h5 = h5py.File(dataset, "r")
        except:
            print('Set dataset?')

        self.datum = (self.h5['dataset'], self.h5['images'])
        self.keys = list(self.datum[0].keys())  # for ids [0,len())
        self.shuffle = shuffle
        self.augment = augment
        self.epoch = len(self.keys)

        print('Used Data: ' + str(int(len(self.keys) * percentage_of_data)) + ' out of ' + str(self.epoch))
        self.epoch = int(len(self.keys) * percentage_of_data)
        self.keys = np.array( list(self.datum[0].keys())[: self.epoch] ) # for ids [0,len())

        # if batchsize == 'all':
        #    self.batch_size = self.epoch

        if batchsize:
            if batchsize == 'all':
                self.batch_size = self.epoch
            else:
                self.batch_size = batchsize
        else:
            self.batch_size = config.batch_size
            print('Set Batch_size from config!')

        if self.config.onlyLane:
            try:
                filename = self.config.DIR + self.config.experiment_name + '/cluster.sav'
                filename_cent = self.config.DIR + self.config.experiment_name + '/cluster_cent.sav'
                self.cluster_model = pickle.load(open(filename, 'rb'))
                self.cluster_cent = pickle.load(open(filename_cent, 'rb'))
            except:
                print('create cluster file first!')
                self.cluster_model = None
                self.cluster_cent = None
        else:
            self.cell_size_grid = (
                                          self.config.grid_cel_size + 1) * 4 + 5 + 1  # [x0,x16], [y0,y16], [aktuell grid_cell], [next grid_cell], [conf]
            self.grid_image_x = np.zeros((self.image_h, self.image_w, 1), dtype=np.float32)
            self.draw_grid_x(self.grid_image_x, config)
            self.grid_image = np.zeros((self.image_h, self.image_w, 1), dtype=np.float32)
            self.draw_grid(self.grid_image, config)
        self.epoch_counter = 1
        self.split_data()

    def split_data(self):
        if self.shuffle:
            np.random.shuffle(self.keys)
        self.keys_splitted = np.array_split(self.keys, 4)
        self.key = self.keys_splitted[self.epoch_counter-1]
        self.epoch =  int(len(self.key))

    def draw_grid_x(self, img, config, line_color=(1, 1, 1), thickness=1, type_=cv2.LINE_AA, pxstep=16):
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
     #   while x < img.shape[1]:
     #       cv2.line(img, (x, 0), (x, img.shape[0]), color=line_color, lineType=type_, thickness=thickness)
     #       x += pxstep

        while y < img.shape[0]:
            cv2.line(img, (0, y), (img.shape[1], y), color=line_color, lineType=type_, thickness=thickness)
            y += pxstep

      #  cv2.line(img, (367, 0), (367, img.shape[0]), color=line_color, lineType=type_, thickness=thickness)
        cv2.line(img, (0, config.img_w -1), (img.shape[1], 367), color=line_color, lineType=type_, thickness=thickness)
    def draw_grid(self, img,config, line_color=(1, 1, 1), thickness=1, type_=cv2.LINE_AA, pxstep=16):
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
            cv2.line(img, (x, 0), (x, img.shape[0]), color=line_color, lineType=type_, thickness=thickness)
            x += pxstep

        while y < img.shape[0]:
            cv2.line(img, (0, y), (img.shape[1], y), color=line_color, lineType=type_, thickness=thickness)
            y += pxstep

        cv2.line(img, (config.img_w-1, 0), (config.img_w-1, img.shape[0]), color=line_color, lineType=type_, thickness=thickness)
        cv2.line(img, (0, config.img_w-1), (img.shape[1], config.img_w-1), color=line_color, lineType=type_, thickness=thickness)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.epoch / self.config.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.key[index * self.batch_size:(index + 1) * self.batch_size]


        # Find list of IDs
     #   list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.batch_gen_inplace(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.epoch_counter = self.epoch_counter + 1
        if self.epoch_counter < 5:
            print('aktuell splitt set: ' +str(self.epoch_counter - 1))
            self.key = self.keys_splitted[self.epoch_counter - 1]
            self.epoch = int(len(self.key))
        else:
            print('shuffle and splitt new dataset: ' + str(self.epoch_counter - 1))
            self.epoch_counter = 1
            self.split_data()


    def batch_gen_inplace(self, indexes):

        final_images = np.ndarray(shape=(self.batch_size, self.image_h, self.image_w, 3), dtype=np.float32)

        grid_batch = np.ndarray((self.batch_size, self.config.grid_size, self.config.grid_size,
                                 self.config.num_prediction_cells, self.cell_size_grid),
                                dtype=np.float16)

        for i, key in enumerate(indexes):
            image, lanes = self.read_data(key)
            image, lanes = self.transform_data(image, lanes)
            grid = self.gen_spline_to_grid(lanes, debug=True)
            final_images[i] = image
            grid[:, :, :, -9] = i
            grid[:, :, :, -5] = i
            grid_batch[i] = grid
        if self.config.splitted:
            staged_batch = []
            grid_batch = np.reshape(grid_batch, (self.batch_size, self.config.grid_size, self.config.grid_size,
                                                 self.config.num_prediction_cells * self.cell_size_grid))

            for st in range(self.config.stages):
                staged_batch.append(grid_batch)
                staged_batch.append(grid_batch)
            # staged_batch.append(grid_batch)

            return final_images, staged_batch
        elif self.config.staged:
            staged_batch = []
            grid_batch = np.reshape(grid_batch, (self.batch_size, self.config.grid_size, self.config.grid_size,
                                                 self.config.num_prediction_cells * self.cell_size_grid))

            for st in range(self.config.stages):
                staged_batch.append(grid_batch)
            #  staged_batch.append(grid_batch)

            return final_images, staged_batch
        else:
            return final_images, grid_batch

    def gen_spline_to_grid(self, lanes, val=False, debug=False):
       # correction_image = np.zeros((self.image_h, self.image_w, 1), dtype=np.float32)
        grid = np.zeros((self.config.grid_size, self.config.grid_size, self.config.num_prediction_cells,
                         self.cell_size_grid), dtype=np.float32)
       # numb_lane = len(lanes)

        if val:
            gather_lanes = []
        for laneid, l in enumerate(lanes):
            x_l = l[:, 0]
            y_l = l[:, 1]

            if (len(x_l) > 1):
                #  import timeit
                #   start = timeit.default_timer()
                image = np.zeros((self.image_w, self.image_h, 1), dtype=np.float32)
                f = np.array([x_l, y_l]).T  # (x,y) standard notation!

                image = cv2.polylines(image, np.int32([f]), 0, 1, thickness=1, )
                overlapp_img = np.logical_and(self.grid_image_x, image)
                indezes = np.array(np.where(overlapp_img[:, :, 0] != [0]))
                _, idx = np.unique(indezes[0, :], return_index=True)
                indezes = indezes[:, idx]
                indezes = indezes[:, indezes[0, :].argsort()]
                x_for_spline = indezes[0]  # swap y to x for a better fit, because of many vertical lines

                if (len(x_for_spline)) >= 2:

                    cs = np.poly1d(np.polyfit(y_l, x_l, 3))
                   # gather_indeces_list = []
                   # end_reached = False
                    if val:
                        x_range_hole = np.arange(x_for_spline[0], x_for_spline[-1], 1)
                        gather_lanes.append(np.array([cs(x_range_hole), x_range_hole]).T)
                    for i in range(len(x_for_spline) - 1):
                        x_cell_first = x_for_spline[i] // self.config.grid_cel_size
                        x_range = np.arange(x_cell_first * (self.config.grid_cel_size),
                                            (x_cell_first + 1) * (self.config.grid_cel_size) + 1, 1)
                        x_weights = np.ones_like(x_range, dtype=np.float32)
                        y_fill = cs(x_range)
                        y_cell_first = []
                        for j in range(0, self.config.grid_size):
                            y_axis = j * self.config.grid_cel_size
                            found_pxl = np.sum(image[
                                               x_cell_first * self.config.grid_cel_size: x_cell_first * self.config.grid_cel_size + self.config.grid_cel_size,
                                               y_axis: y_axis + self.config.grid_cel_size])
                            if found_pxl:
                                y_cell_first.append(j)
                            # for i in self.config.mean_y_fill:
                        #     y_cell_first = np.array(y_cell_first)
                        mean_y_fill = np.nanmean(y_fill)
                        y_cell_first_mean = int(mean_y_fill // self.config.grid_cel_size)

                        if y_cell_first_mean > self.config.grid_size-1:
                            y_cell_first_mean = self.config.grid_size-1
                        if y_cell_first_mean < 0:
                            y_cell_first_mean = 0
                        y_cell_first = [y_cell_first_mean]


                        for y_cell in y_cell_first:
                            for a in range(self.config.num_prediction_cells):
                                if grid[x_cell_first, y_cell, a, -1] == 0.:

                                    norm_offset = - (
                                            x_cell_first * self.config.grid_cel_size + self.config.grich_anchor_pt)
                                    # x_weights[int(x_for_spline[i] + norm_offset)] = 10.
                                    # x_weights[int(x_for_spline[i + 1] - (x_cell_second * self.config.grid_cel_size + self.config.grich_anchor_pt))] = 10.
                                    x_weights[0] = 1.5
                                    x_weights[-1] = 1.5

                                    #                         non_nans = np.count_nonzero(~np.isnan(y_fill))  # count nans to threshold extrapolation
                                    #                         non_nans_quo = (non_nans / (self.config.grid_cel_size +1))
                                    #                         if non_nans_quo < .6:
                                    #                             break
                                    norm_x = (x_range + norm_offset)
                                    norm_y = (
                                            y_fill - (y_cell * self.config.grid_cel_size + self.config.grich_anchor_pt))

                                    grid[x_cell_first, y_cell, a,
                                    0:self.config.grid_cel_size + 1] = norm_x  # 0-16, 16-32... we want 17 values!
                                    grid[x_cell_first, y_cell, a, self.config.grid_cel_size + 1:
                                                                  2 * (
                                                                          self.config.grid_cel_size + 1)] = norm_y  # shift by 1 to fill 17 values and not 16 # TODO: fix this cause ugly
                                    grid[x_cell_first, y_cell, a, 2 * (self.config.grid_cel_size + 1):
                                                                  3 * (self.config.grid_cel_size + 1)] = x_weights

                                    if y_cell == y_cell_first_mean:
                                        grid[x_cell_first, y_cell, a, -1] = 1.0  # for classification!
                                    else:
                                        grid[x_cell_first, y_cell, a, -1] = 1.  # for classification!

                                    break

        if val:
            return grid, gather_lanes
        else:
            return grid

    def read_data(self, key):
        entry = self.datum[0][str(key)]
       # dataset_entry = json.loads(entry.value)
        lanes = list(json.loads(entry.attrs['meta']).items())
        lanes = lanes[0][1]
        lanes_as_array = []
        for l in lanes:
            l = np.array(l, dtype=np.float32)
            lanes_as_array.append(l)
        lanes = lanes_as_array
        img = self.datum[1][str(key)].value

        if len(img.shape) == 2 and img.shape[1] == 1:
            img = cv2.imdecode(img, flags=-1)  # decode to array
        else:
            print('something went wrong with decode')

        return img, lanes

    def transform_data(self, img, lanes):

        aug = AugmentSelection.random() if self.augment else AugmentSelection.unrandom()
        img, label = Transformer.transform(img, lanes, self.config, aug)

        return img, lanes