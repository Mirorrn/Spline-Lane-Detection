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

from imgaug import augmenters as iaa


class data_gen:

    def __init__(self, config, dataset,batchsize=None, shuffle = True, augment = True, percentage_of_data= 1.):

        self.config = config
        self.image_w = config.img_w
        self.image_h = config.img_h

        try:
            self.h5 = h5py.File(dataset, "r")
        except:
            print('Set dataset?')

        self.datum =  (self.h5['dataset'], self.h5['images'])
        self.keys = list(self.datum[0].keys()) # for ids [0,len())
        self.shuffle = shuffle
        self.augment = augment
        self.epoch = len(self.keys)

        print('Used Data: ' + str(int(len(self.keys) * percentage_of_data)) + ' out of ' + str(self.epoch))
        self.epoch = int(len(self.keys) * percentage_of_data)
        self.keys = list(self.datum[0].keys())[: self.epoch] # for ids [0,len())

        #if batchsize == 'all':
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
            self.cell_size_grid = (self.config.grid_cel_size + 1)*4  + 5 + 1 # [x0,x16], [y0,y16], [aktuell grid_cell], [next grid_cell], [conf]
            self.grid_image_x = np.zeros((self.image_h, self.image_w, 1), dtype=np.float32)
            self.draw_grid_x(self.grid_image_x, config)
            self.grid_image = np.zeros((self.image_h, self.image_w, 1), dtype=np.float32)
            self.draw_grid(self.grid_image, config)

    def gen_random_lines(self):

        image = np.zeros((self.image_h, self.image_w, 1), dtype=np.float32)

        c = 32.

        a = np.random.uniform(-0.1, 0.1, 1)

        b = np.random.uniform(-3, 3, 1)

        x = np.arange(0, self.image_w, .1)

        y = np.round(a * x ** 2 + b * x + c)
        f = np.array((x, y)).T

        image = cv2.polylines(image, np.int32([f]), 0, 1, thickness=3)

        y2 = np.round(a * (x-64.) ** 2 + b * (x-64.) + c)
        f = np.array((x, y2)).T

        image = cv2.polylines(image, np.int32([f]), 0, 1, thickness=3)

        return a, b, c, image

    def lanes_toGrid(self, lanes, debug=False):
        #  TODO: find a better interpolation !
        interpolated_lanes = []

        for l in lanes:
            img = np.zeros((self.image_h, self.image_w, 1), dtype=np.float32)
            img = cv2.polylines(img, np.int32([l]), 0, 1, thickness=1)
            img = img[:, :, 0]
            x = l[:,0]
            y_f = l[:,1]
            z = np.polyfit(x, y_f, 2) # much better way TODO follow this strategy of interpolating

            indices = np.where(img != [0])
            whites = [indices[1], indices[0]]
  #          for i, pixel_h in enumerate(img):
  #              for j,pixel_w in enumerate(pixel_h):
  #                  if pixel_w == 1:
  #                      whites.append([j,i])
            whites = np.array(whites).T
            interpolated_lanes.append(whites)

     #   stop = timeit.default_timer()
     #   print(stop - start)

           # plt.imshow(img, cmap='gray') # only for debug
           # plt.show()
        # end of interpolation
        # now points to grid
        # create list of list and then fill it with corresponding points
        # care sort them by x!
        x, y = self.config.grid_size, self.config.grid_size
        grid =np.empty((x,y),dtype=np.object)
        for i,s in enumerate(grid):
            for j,b in enumerate(s):
                grid[i, j] = []

        temp_copy_grid = np.empty((x, y), dtype=np.object)
        for i, s in enumerate(temp_copy_grid):
            for j, b in enumerate(s):
                temp_copy_grid[i, j] = []

        # TODO: overlapping? Now its only override full grid cells
        for il in interpolated_lanes:
            temp_grid = copy.deepcopy(temp_copy_grid)
            for point in il:
                x, y = point
                grid_x = x // self.config.grid_cel_size
                grid_y = y // self.config.grid_cel_size
                temp_grid[grid_y, grid_x].append(point)

            for s, i in enumerate(grid):
                for g, j in enumerate(i):
                    if len(j) == 0:
                        grid[s, g] = temp_grid[s, g]

        for s,i in enumerate(grid):
            for g, j in enumerate(i):
                if len(j) > 3:  # very critical!
                    j.sort(key=itemgetter(0, 1))  # first x and then y coords, btw cool inplace sort method!
                    grid[s, g] =  [x - (g * self.config.grid_cel_size + self.config.grich_anchor_pt, s * self.config.grid_cel_size + self.config.grich_anchor_pt) for x in j] # normalize to grid cell
                    grid[s, g].append(1.)
                else:
                    grid[s,g] = [0.]  # delete grid cells with fewer as 3 points
                    # debug
      #              print('points deleted')


        # DEBUG print grid elements
        if debug:
            img = np.zeros((self.image_h, self.image_w, 1), dtype=np.float32)
            plotgtid = np.zeros((self.config.grid_size, self.config.grid_size, 1), dtype=np.float32)
            for s,i in enumerate(grid):
                for g, j in enumerate(i):
                    if(j[-1]):
                        temp_j = j[:-1]
                        temp_j = [x + (g * self.config.grid_cel_size + self.config.grich_anchor_pt, s * self.config.grid_cel_size + self.config.grich_anchor_pt) for x in temp_j] # normalize to grid cell
                        img = cv2.polylines(img, np.int32([temp_j]), 0, 1, thickness=1)
                        plotgtid[s,g] = 1

            if self.config.debug_datagen:
                plt.imshow(img[:,:,0], cmap='gray') # only for debug
                plt.show()
            return grid, img

        return grid

    def set_border(self, x):
        if x < 0:
            return 0
        if x > self.image_w:
            return self.image_w

        x_max_border = x / self.config.grid_cel_size
        border_th = (((x_max_border - int(
            x_max_border)) * self.config.grid_cel_size) / self.config.grid_cel_size) < 0.6  # check if border is above th
        if border_th:
            x = int(x_max_border) * self.config.grid_cel_size
        return x

    def rot_90_degree(self, x, y):
        return y, x

    def rotate(self, origin, x, y, angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.

        The angle should be given in radians.
        """
        angle = (angle/180.) * pi
        ox, oy = origin
        px, py = x, y

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        return qx, qy

    def gen_line_to_grid_with_rot(self, lanes, debug=False):
        numb_lane = len(lanes)
        grid = np.zeros((self.config.grid_size, self.config.grid_size, self.config.num_prediction_cells, self.config.num_gridsamples + 1), dtype=np.float32)
        grid_id = np.zeros((numb_lane, self.config.grid_size, self.config.grid_size, self.config.num_prediction_cells, 1), dtype=np.int16)
        grid_id_sum = np.zeros(
            (numb_lane, self.config.grid_size, self.config.grid_size, self.config.num_prediction_cells, 1),
            dtype=np.int16)
        correction_image = np.zeros((self.image_h, self.image_w, 1), dtype=np.float32)
        for laneid, l in enumerate(lanes):
            x = l[:, 0]
            y_f = l[:, 1]
            x, y_f = y_f, x  # rot for a better fit, because of many vertical lines
            x_min = np.min(x)
            x_max = np.max(x)

            x_max = self.set_border(x_max)
            x_min = self.set_border(x_min)
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    coefficients = np.polyfit(x, y_f, 2)
                except np.RankWarning:
                    print("not enought data:" + str(len(x)))    # some lines are just two points...
                    continue
            a, b, c = coefficients[:]
            image = np.zeros((self.image_w, self.image_h, 1), dtype=np.float32)
            x = np.arange(x_min, x_max, 1 / self.config.num_of_samples)
            x_image = np.arange(0, self.config.img_w, 1 / self.config.num_of_samples)
            y_exact = a * x ** 2 + b * x + c
            y_fill = a * x_image ** 2 + b * x_image + c # thats y in rotated coord.s.
            f = np.array([y_exact, x]).T # (x,y) standard notation!

            image = cv2.polylines(image, np.int32([f]), 0, 1 ,thickness=1)
            correction_image = np.logical_or(correction_image, image) # criterium for assigment: first layer for hole struktur, layer above for crossover
          #  import timeit
          #  start = timeit.default_timer()

            for i in range(0, self.config.grid_size):
                for j in range(0, self.config.grid_size):
                    y_axis = i * self.config.grid_cel_size
                    x_axis = j * self.config.grid_cel_size
                    found_pxl = np.sum(image[y_axis: y_axis + self.config.grid_cel_size,    # fixed this!
                                       x_axis: x_axis + self.config.grid_cel_size])  # fast numpy search for pixel
                    if found_pxl >2:
                        for a in range(self.config.num_prediction_cells):
                            if grid[i, j, a, -1] == 0.:
                                section = slice(i * (self.config.num_gridsamples), i *
                                                (self.config.num_gridsamples) + self.config.num_gridsamples)
                                grid[i, j, a, :-1] = np.asarray(y_fill[section] - (x_axis + self.config.grich_anchor_pt))
                                grid[i, j, a, -1] = 1.0
                                grid_id[laneid, i, j, a] = i + 1
                                grid_id_sum[laneid, i, j, a] = found_pxl
                                break

  #     #      # fix for this duplicated TP cases!
        for id in range(0, numb_lane):
            for i in range(0, self.config.grid_size):
                for j in range(0, self.config.grid_size):
                    if grid_id[id, i, j, 0]:
                        check_for_overlapp = np.where(grid_id[id, :, :, 0] == grid_id[id, i, j, 0])
                        indezes = np.array(check_for_overlapp).T
                        if indezes.shape[0] > 1:
                            sum = grid_id_sum[id, indezes[:, 0], indezes[:, 1], indezes[:, 2]]
                            max = np.argmax(sum)
                            # now delete the others!
                            for i, idx in enumerate(indezes):
                                if not i == max:
                                    grid[idx[0], idx[1], 0].fill(0.)
                                    grid_id[id, idx[0], idx[1], idx[2]].fill(0.)

        for id in range(0, numb_lane):
            for i in range(0, self.config.grid_size):
                for j in range(0, self.config.grid_size):
                    for a in range(1, self.config.num_prediction_cells):
                        if grid_id[id, i, j, a]:
                            check_for_overlapp = np.any(np.where(grid_id[id, :, :, 0] == grid_id[id, i, j, a]))
                            if check_for_overlapp:
                                grid[i, j, a].fill(0.)
           # stop = timeit.default_timer()
           # print(stop - start)

        if debug:
            if self.config.debug_datagen:
                plt.imshow(correction_image[:, :, 0], cmap='gray')  # only for debug
                plt.show()

        if debug:
            return grid, correction_image
        return grid

    def gen_line_to_lane(self, lanes, debug=False):
        correction_image = np.zeros((self.image_h, self.image_w, 1), dtype=np.uint8)
        #norm = 0 #self.image_h
        grid = np.zeros((self.config.num_prediction_cells, 9), dtype=np.object)
        if self.config.debug_datagen:
            mean_x_list = []
            mean_y_list = []
            min_x_list = []
            min_y_list = []
            max_x_list = []
            max_y_list = []

        if self.cluster_model:
            #order_vec = np.ones((4), dtype=np.int16) * -1
            temp_assigment_row = []
            for laneid, t in enumerate(lanes):

                x_h = t[:, 0]
                y_h = t[:, 1]
                mean_x = np.mean(np.array(x_h))
                mean_y = np.mean(np.array(y_h))
                temp_assigment_row.append(self.cluster_model.transform(np.array([mean_x, mean_y]).reshape(1, -1))[0])
            assignment_matrix = np.asarray(temp_assigment_row)
            row_ind, col_ind = linear_sum_assignment(assignment_matrix)

        for laneid, l in enumerate(lanes):

            x = l[:, 0]
            y_f = l[:, 1]

            if len(x) > 2:
                x, y_f = y_f, x  # rot for a better fit, because of many vertical lines
                # TODO: CORRECT THIS!!!!!!!!!!
                #min_x = 0 if  x[0] < 0 else  x[0]
                #max_x = 368 if  x[-1] > 368 else  x[-1]
                #middle = [x[len(l)//2], y_f[len(l)//2]]
                #min_y = 368 if y_f[0] > 368 else y_f[0]
                #max_y = 368 if y_f[-1] > 368 else y_f[-1]
                mean_x = np.mean(np.array(x))
                mean_y = np.mean(np.array(y_f))

                image = np.zeros((self.image_w, self.image_h, 1), dtype=np.float32)

                f = np.array([y_f, x]).T  # (x,y) standard notation!

                image = cv2.polylines(image, np.int32([f]), 0, 1, thickness=1)
                indices = np.where(image[:,:,0] != [0])
                whites = [indices[0], indices[1]] #[y, x] swap axis -> [x, y]

               # correction_image[whites] = 1.0

               # plt.imshow(correction_image[:,:,0], cmap='gray')  # only for debug
              #  plt.show()

                whites = np.array(whites).T

            #    whites_index = np.unique(whites[:, 1],return_index = True)[1]
                y_min_interp = whites[0, 1]
                y_max_interp = whites[-1, 1]
                y_mid_interp = whites[whites.shape[0] //2, 1]
             #   y_sample_interp = np.round(np.linspace(y_min_interp, y_max_interp, 6, dtype=float))
             #   y_filtered = whites[:, 1]
             #   y_indices = np.where(np.isin(y_filtered[whites_index], y_sample_interp))[0]

  #              if len(y_indices) <10:
  #                  test = 'wtf'


                x_min_interp= whites[0, 0]
                x_max_interp = whites[-1, 0]
                x_mid_interp = whites[whites.shape[0] // 2, 0]
             #   x_sample_interp = whites[:, 0]
             #   x_sample_interp = x_sample_interp[whites_index]
             #   x_sample_interp = x_sample_interp[y_indices]



                correction_image = np.logical_or(correction_image,
                                                 image)  # criterium for assigment: first layer for hole struktur, layer above for crossover

                if self.cluster_model:

                    grid[col_ind[row_ind[laneid]],0] = (x_min_interp  - self.cluster_cent[col_ind[row_ind[laneid]]][
                                                                0])/ self.image_h
                    grid[col_ind[row_ind[laneid]],1] = (x_mid_interp  - self.cluster_cent[col_ind[row_ind[laneid]]][
                                                                0])/ self.image_h
                    grid[col_ind[row_ind[laneid]],2] = (x_max_interp - self.cluster_cent[col_ind[row_ind[laneid]]][
                                                                0]) / self.image_h
                    grid[col_ind[row_ind[laneid]],3] = (y_min_interp - self.cluster_cent[col_ind[row_ind[laneid]]][
                                                                1]) / self.image_h
                    grid[col_ind[row_ind[laneid]],4] = (y_mid_interp - self.cluster_cent[col_ind[row_ind[laneid]]][
                                                                1]) / self.image_h
                    grid[col_ind[row_ind[laneid]],5] = (y_max_interp - self.cluster_cent[col_ind[row_ind[laneid]]][
                                                                1]) / self.image_h
                    grid[col_ind[row_ind[laneid]],6] = mean_x #/ self.image_h
                    grid[col_ind[row_ind[laneid]],7] = mean_y #/ self.image_h
                    grid[col_ind[row_ind[laneid]],8] = 1.0
              #  else:
               #     grid[col_ind[row_ind[laneid]], 0:10] = (y_sample_interp -
               #                                             self.cluster_cent[col_ind[row_ind[laneid]]][
               #                                                 0]) / self.image_h
               #     grid[col_ind[row_ind[laneid]], 10:20] = (x_sample_interp -
               #                                              self.cluster_cent[col_ind[row_ind[laneid]]][
               #                                                  1]) / self.image_h
               #     grid[col_ind[row_ind[laneid]],20] = mean_x #/ self.image_h
                #    grid[col_ind[row_ind[laneid]],21] = mean_y #/ self.image_h
                ##    grid[col_ind[row_ind[laneid]],22] = 1.0

                if self.config.debug_datagen:
                    mean_x_list.append(mean_x)
                    mean_y_list.append(mean_y)
                    min_x_list.append(x_min_interp)
                    min_y_list.append(y_min_interp)
                    max_x_list.append(x_max_interp)
                    max_y_list.append(y_max_interp)

        if debug:
            if self.config.debug_datagen:
                fully = correction_image.copy()
                fully = np.array(fully, dtype=np.uint8)
                for x_mean,y_mean,x_min,y_min in zip(mean_x_list, mean_y_list, min_x_list, min_y_list):
                    fully = cv2.circle(fully, (int(y_mean),int(x_mean)), 5, (1,1,1), thickness=1, lineType=8, shift=0)
                    fully = cv2.circle(fully, (int(y_min), int(x_min)), 3, (1, 1, 1), thickness=1, lineType=8, shift=0)
                plt.imshow(fully[:, :, 0], cmap='gray')  # only for debug
                plt.show()

        if debug:
            return grid, correction_image
        return grid

    def gen_line_to_grid(self, lanes, debug=False):
        numb_lane = len(lanes)
        grid = np.zeros((self.config.grid_size, self.config.grid_size, self.config.num_prediction_cells,
                         self.config.num_gridsamples + 1), dtype=np.float32)
        grid_id = np.zeros(
            (numb_lane, self.config.grid_size, self.config.grid_size, self.config.num_prediction_cells, 1),
            dtype=np.int16)
        grid_id_sum = np.zeros(
            (numb_lane, self.config.grid_size, self.config.grid_size, self.config.num_prediction_cells, 1),
            dtype=np.int16)
        correction_image = np.zeros((self.image_h, self.image_w, 1), dtype=np.float32)
        nb_grid = np.zeros((self.config.grid_size, self.config.grid_size, 8), dtype=np.int16)
        for laneid, l in enumerate(lanes):
            x = l[:, 0]
            y_f = l[:, 1]
            x, y_f = y_f, x  # rot for a better fit, because of many vertical lines
            x_min = np.min(x)
            x_max = np.max(x)

            x_max = self.set_border(x_max)
            x_min = self.set_border(x_min)
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    coefficients = np.polyfit(x, y_f, 2)
                except np.RankWarning:
                    print("not enought data:" + str(len(x)))  # some lines are just two points...
                    continue
            a, b, c = coefficients[:]
            image = np.zeros((self.image_w, self.image_h, 1), dtype=np.float32)
            x = np.arange(x_min, x_max, 1 / self.config.num_of_samples)
            x_image = np.arange(0, self.config.img_w, 1 / self.config.num_of_samples)
            y_exact = a * x ** 2 + b * x + c
            y_fill = a * x_image ** 2 + b * x_image + c  # thats y in rotated coord.s.

            f = np.array([y_exact, x]).T  # (x,y) standard notation!

            image = cv2.polylines(image, np.int32([f]), 0, 1, thickness=1)
            correction_image = np.logical_or(correction_image,
                                             image)  # criterium for assigment: first layer for hole struktur, layer above for crossover
            #  import timeit
            #  start = timeit.default_timer()
            temp_grid = np.zeros((self.config.grid_size, self.config.grid_size, 1), dtype=np.float32)
            for i in range(0, self.config.grid_size):
                for j in range(0, self.config.grid_size):
                    y_axis = i * self.config.grid_cel_size
                    x_axis = j * self.config.grid_cel_size
                    found_pxl = np.sum(image[y_axis: y_axis + self.config.grid_cel_size,  # fixed this!
                                       x_axis: x_axis + self.config.grid_cel_size])  # fast numpy search for pixel
                    if found_pxl > 2:
                        for a in range(self.config.num_prediction_cells):
                            if grid[i, j, a, -1] == 0.:
                                section = slice(i * (self.config.num_gridsamples), i *
                                                (self.config.num_gridsamples) + self.config.num_gridsamples)
                                grid[i, j, a, :-1] = np.asarray(
                                    y_fill[section] )#- (x_axis + self.config.grich_anchor_pt))
                                grid[i, j, a, -1] = 1.0
                                grid_id[laneid, i, j, a] = i + 1
                                temp_grid[i, j] = 1.0
                                grid_id_sum[laneid, i, j, a] = found_pxl
                                break

            footprint = np.array([[1, 1, 1],
                                  [1, 0, 1],
                                  [1, 1, 1]])

            results = []
            ndimage.generic_filter(temp_grid[:, :, 0], self.test_func, mode='reflect', footprint=footprint,
                                   extra_arguments=(results,))
            results = np.array(results)
            results = results.reshape(self.config.grid_size, self.config.grid_size, 8)
            nb_grid = np.logical_or(nb_grid, results) * 1
        #     #      # fix for this duplicated TP cases!
 #       for id in range(0, numb_lane):
 #           for i in range(0, self.config.grid_size):
 #               for j in range(0, self.config.grid_size):
 #                   if grid_id[id, i, j, 0]:
 #                       check_for_overlapp = np.where(grid_id[id, :, :, 0] == grid_id[id, i, j, 0])
 #                       indezes = np.array(check_for_overlapp).T
 #                       if indezes.shape[0] > 1:
 #                           sum = grid_id_sum[id, indezes[:, 0], indezes[:, 1], indezes[:, 2]]
 #                           max = np.argmax(sum)
 #                           # now delete the others!
 #                           for i, idx in enumerate(indezes):
 #                               if not i == max:
  #                                  grid[idx[0], idx[1], 0].fill(0.)
 #                                   grid_id[id, idx[0], idx[1], idx[2]].fill(0.)

        for id in range(0, numb_lane):
            for i in range(0, self.config.grid_size):
                for j in range(0, self.config.grid_size):
                    for a in range(1, self.config.num_prediction_cells):
                        if grid_id[id, i, j, a]:
                            check_for_overlapp = np.any(np.where(grid_id[id, :, :, 0] == grid_id[id, i, j, a]))
                            if check_for_overlapp:
                                grid[i, j, a].fill(0.)

        # stop = timeit.default_timer()
        # print(stop - start)

        if debug:
            if self.config.debug_datagen:
                plt.imshow(grid[:, :, 0, -1], cmap='gray')  # only for debug
                plt.show()

        if debug:
            return grid, correction_image, nb_grid
        return grid

    def test_func(self, values, out):
        out.append(values)
        return 0

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
        cv2.line(img, (0, config.img_w-1), (img.shape[1],  config.img_w-1), color=line_color, lineType=type_, thickness=thickness)

    def draw_grid(self, img, config, line_color=(1, 1, 1), thickness=1, type_=cv2.LINE_AA, pxstep=16):
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

    def gen_spline_to_grid(self, lanes,val=False, debug=False):
        correction_image = np.zeros((self.image_h, self.image_w, 1), dtype=np.float32)
        grid = np.zeros((self.config.grid_size, self.config.grid_size, self.config.num_prediction_cells,
                         self.cell_size_grid), dtype=np.float32)
        numb_lane = len(lanes)
    #    grid_id = np.zeros(
    #        (numb_lane, self.config.grid_size, self.config.grid_size, self.config.num_prediction_cells, 1),
    #        dtype=np.int16)
    #    nb_grid = np.zeros((self.config.grid_size, self.config.grid_size, 3), dtype=np.int16)

        if val:
            gather_lanes= []
        for laneid, l in enumerate(lanes):
            x_l = l[:, 0]
            y_l = l[:, 1]

            if(len(x_l) > 1):
              #  import timeit
             #   start = timeit.default_timer()
                image = np.zeros((self.image_w, self.image_h, 1), dtype=np.float32)
                f = np.array([x_l, y_l]).T  # (x,y) standard notation!

                image = cv2.polylines(image, np.int32([f]), 0, 1, thickness=1, )
                overlapp_img = np.logical_and(self.grid_image_x,image)
                indezes = np.array(np.where(overlapp_img[:,:,0] != [0]))
                _, idx = np.unique(indezes[0,:], return_index=True)
                indezes = indezes[:,idx]
                indezes = indezes[:, indezes[0, :].argsort()]
                x_for_spline = indezes[0]   # swap y to x for a better fit, because of many vertical lines
            #    y_for_spline = indezes[1]   # swap x to y

            #    overlapp_img = np.logical_and(self.grid_image, image)
            #    indezes_for_grid = np.array(np.where(overlapp_img[:, :, 0] != [0]))
            #    indezes_for_grid = indezes_for_grid[:, indezes_for_grid[0, :].argsort()]
            #    x_for_grd = indezes_for_grid[0]  # swap y to x for a better fit, because of many vertical lines
            #    y_for_grid = indezes_for_grid[1]  # swap x to y

              #  stop = timeit.default_timer()
              #  print(stop - start)
                if(len(x_for_spline)) >= 2:

                   # cs = CubicSpline(x_for_spline, y_for_spline,bc_type='natural' ,extrapolate=False)#, kind='quadratic')
                    cs = np.poly1d(np.polyfit(y_l, x_l, 3))
                    gather_indeces_list = []
                    end_reached = False
                    if val:
                        x_range_hole = np.arange(x_for_spline[0],x_for_spline[-1] , 1)
                        gather_lanes.append(np.array([cs(x_range_hole),x_range_hole]).T)
                    for i in range(len(x_for_spline) -1):
                        x_cell_first = x_for_spline[i] // self.config.grid_cel_size
                        x_range = np.arange(x_cell_first * (self.config.grid_cel_size),
                                            (x_cell_first + 1) * (self.config.grid_cel_size) + 1, 1)
                        x_weights = np.ones_like(x_range, dtype=np.float32)
                        y_fill = cs(x_range)
                        y_cell_first = []
                        for j in range(0, self.config.grid_size):
                            y_axis = j * self.config.grid_cel_size
                            found_pxl = np.sum(image[x_cell_first*self.config.grid_cel_size: x_cell_first*self.config.grid_cel_size + self.config.grid_cel_size,
                                               y_axis: y_axis + self.config.grid_cel_size])
                            if found_pxl:
                                y_cell_first.append(j)
                            #for i in self.config.mean_y_fill:
                   #     y_cell_first = np.array(y_cell_first)
                        mean_y_fill = np.nanmean(y_fill)
                        y_cell_first_mean = int(mean_y_fill  // self.config.grid_cel_size)

                        if y_cell_first_mean > self.config.grid_size-1:
                            y_cell_first_mean = self.config.grid_size-1
                        if y_cell_first_mean < 0:
                            y_cell_first_mean = 0
                        y_cell_first = [y_cell_first_mean]
                       # if((i +1) >= (len(x_for_spline) -1)):
                       #     end_reached = True
                       # else:
                       #     x_cell_second = x_for_spline[i + 1] // self.config.grid_cel_size

                        #    x_range_second = np.arange(x_cell_second * (self.config.grid_cel_size),
                         #                       (x_cell_second + 1) * (self.config.grid_cel_size) + 1, 1)

                          #  y_fill_second = cs(x_range_second)
                           # mean_y_fill_second = np.nanmean(y_fill_second)
                            #y_cell_second = int(mean_y_fill_second // self.config.grid_cel_size)

                           # if y_cell_second > 22:
                           #     y_cell_second = 22
                           # if y_cell_second < 0:
                           #     y_cell_second = 0

                            #a_second =0
                            #for a in range(self.config.num_prediction_cells):
                             #   if grid[x_cell_second, y_cell_second, a, -1]:
                            #        a_second = a


                        for y_cell in y_cell_first:
                            for a in range(self.config.num_prediction_cells):
                                if grid[x_cell_first, y_cell, a, -1] == 0.:

                                    norm_offset = - (x_cell_first * self.config.grid_cel_size + self.config.grich_anchor_pt)
                                   # x_weights[int(x_for_spline[i] + norm_offset)] = 10.
                                   # x_weights[int(x_for_spline[i + 1] - (x_cell_second * self.config.grid_cel_size + self.config.grich_anchor_pt))] = 10.
                                    x_weights[0] = 1.5
                                    x_weights[-1] = 1.5

           #                         non_nans = np.count_nonzero(~np.isnan(y_fill))  # count nans to threshold extrapolation
           #                         non_nans_quo = (non_nans / (self.config.grid_cel_size +1))
           #                         if non_nans_quo < .6:
           #                             break
                                    norm_x =  (x_range + norm_offset)
                                    norm_y = (y_fill - (y_cell * self.config.grid_cel_size + self.config.grich_anchor_pt))



                                    grid[x_cell_first, y_cell, a, 0:self.config.grid_cel_size + 1] = norm_x # 0-16, 16-32... we want 17 values!
                                    grid[x_cell_first, y_cell, a, self.config.grid_cel_size + 1:
                                                                        2*(self.config.grid_cel_size+1)] = norm_y # shift by 1 to fill 17 values and not 16 # TODO: fix this cause ugly
                                    grid[x_cell_first, y_cell, a, 2 * (self.config.grid_cel_size +1):
                                                                        3 * (self.config.grid_cel_size +1)] = x_weights
                                   # if not end_reached:
                                   #     x_range_second_for_diff = np.arange(x_cell_first * (self.config.grid_cel_size),
                                   #                 (x_cell_second + 1) * (self.config.grid_cel_size) +1, 1)
                                    ##    y_fill_second_diff = cs(x_range_second_for_diff)
                                    #    norm_x_second_diff = (x_range_second_for_diff - (
                                    #            x_cell_first * self.config.grid_cel_size + self.config.grich_anchor_pt))
                                    #    norm_y_second_diff = (y_fill_second_diff - (
                                    #            y_cell_first * self.config.grid_cel_size + self.config.grich_anchor_pt))#

                                        #c1, c2, c3 = np.polyfit(norm_x_second_diff, norm_y_second_diff, 2)
                                        #dif_poly = 2 * c1 * x_range + c2
                                        #grid[x_cell_first, y_cell_first, a, -23:-6] = dif_poly

                                        #grid[x_cell_first, y_cell_first, a, -6:-1] = [y_cell_first,0., x_cell_second, y_cell_second, a_second]
                                    #    print(str(i)+ '  ' + str(x_cell_first)+ '  ' + str(y_cell_first) + '  ' +str(x_cell_second) + '  '+ str(y_cell_second)+ '  ')
                                    if y_cell == y_cell_first_mean:
                                        grid[x_cell_first, y_cell, a, -1] = 1.0 # for classification!
                                    else:
                                        grid[x_cell_first, y_cell, a, -1] = 1.  # for classification!
                                    #grid_id[laneid, x_cell_first, y_cell, a] = x_cell_first + 1
                                    #nb_grid[x_cell_first, y_cell] = 1
                                    break
            #            temp_grid = np.zeros((self.config.grid_size, self.config.grid_size, 1), dtype=np.float32)
            #            for i in range(len(x_for_grd)):
            #                x_cell_grid = x_for_grd[i] // self.config.grid_cel_size
            #                y_cell_gric = y_for_grid[i] // self.config.grid_cel_size
            #                temp_grid[x_cell_grid, y_cell_gric] = 1

                    correction_image = np.logical_or(correction_image,
                                                     image)  # criterium for assigment: first layer for hole struktur, layer above for crossover

         #           plt.imshow(temp_grid[:, :, 0], cmap='gray')  # only for debug
         #           plt.show()


                  #  footprint = np.array([[1, 1, 1],
                  #                        [1, 0, 1],
                  #                        [1, 1, 1]])

                  #  results = []
                  #  ndimage.generic_filter(temp_grid[:, :, 0], self.test_func, mode='reflect', footprint=footprint,
                  #                         extra_arguments=(results,))
                  #  results = np.array(results)
                  #  results = results.reshape(self.config.grid_size, self.config.grid_size, 8)
                  #  nb_grid = np.logical_or(nb_grid, results) * 1



        #for id in range(0, numb_lane):
        #    for i in range(0, self.config.grid_size):
        #        for j in range(0, self.config.grid_size):
        #            for a in range(1, self.config.num_prediction_cells):
        #                if grid_id[id, i, j, a]:
        #                    check_for_overlapp = np.any(np.where(grid_id[id, :, :, 0] == grid_id[id, i, j, a]))
        #                    if check_for_overlapp:
        #                        grid[i, j, a].fill(0.)


       # if debug:
       #     if self.config.debug_datagen:
        #plt.imshow(grid[:, :, 0, -1], cmap='gray')  # only for debug
        #plt.show()

        #if debug:
        #return grid, correction_image* 1., nb_grid
        if val:
            return grid,gather_lanes
        else:
            return grid



    def test_func(self, values, out):
        out.append(values)
        return 0

    def gen_segments(self, lanes, debug=False):
        numb_lane = len(lanes)
        grid = np.zeros((self.config.grid_size, self.config.grid_size, self.config.num_prediction_cells,
                         self.config.num_gridsamples + 1), dtype=np.float32)
        grid_id = np.zeros(
            (numb_lane, self.config.grid_size, self.config.grid_size, self.config.num_prediction_cells, 1),
            dtype=np.int16)
        grid_id_sum = np.zeros(
            (numb_lane, self.config.grid_size, self.config.grid_size, self.config.num_prediction_cells, 1),
            dtype=np.int16)
        correction_image = np.zeros((self.image_h, self.image_w, 1), dtype=np.float32)

        nb_grid = np.zeros((self.config.grid_size, self.config.grid_size,1), dtype=np.int16)
        for laneid, l in enumerate(lanes):
            x = l[:, 0]
            x_min = np.min(x)
            x_max = np.max(x)

            x_max = self.set_border(x_max)
            x_min = self.set_border(x_min)

            y_f = l[:, 1]
            z = np.polyfit(x, y_f, 2)  # much better way TODO follow this strategy of interpolating
            a, b, c = z[:]

            image = np.zeros((self.image_h, self.image_w, 1), dtype=np.float32)
            x = np.arange(x_min, x_max, 1 / self.config.num_of_samples)
            y_exact = np.round(a * x ** 2 + b * x + c)
            x_image = np.arange(0, self.config.img_w, 1 / self.config.num_of_samples)
            y_fill = np.round(a * x_image ** 2 + b * x_image + c)
            f = np.array([x, y_exact]).T  # (x,y) standard notation!

            image = cv2.polylines(image, np.int32([f]), 0, 1, thickness=1)
            correction_image = np.logical_or(correction_image,
                                             image)  # criterium for assigment: first layer for hole struktur, layer above for crossover
            #  import timeit
            #  start = timeit.default_timer()
            temp_grid = np.zeros((self.config.grid_size, self.config.grid_size, 1), dtype=np.float32)
            for i in range(0, self.config.grid_size):
                for j in range(0, self.config.grid_size):
                    y_axis = i * self.config.grid_cel_size
                    x_axis = j * self.config.grid_cel_size
                    found_pxl = np.sum(image[y_axis: y_axis + self.config.grid_cel_size,  # fixed this!
                                       x_axis: x_axis + self.config.grid_cel_size])  # fast numpy search for pixel
                    if found_pxl > 4:
                        for a in range(self.config.num_prediction_cells):
                            if grid[i, j, a, -1] == 0.:
                                grid[i, j, a, :-1] = np.asarray(
                                    y_fill[j * (self.config.num_gridsamples): j * (
                                        self.config.num_gridsamples) + self.config.num_gridsamples]
                                    - (
                                            y_axis + self.config.grich_anchor_pt))  # ToDo check if norm to the grid middle point is better and min max cc
                                grid[i, j, a, -1] = 1.0
                                temp_grid[i,j] = 1.0
                                grid_id[laneid, i, j, a] = j + 1
                                grid_id_sum[laneid, i, j, a] = found_pxl
                                break

            footprint = np.array([[1, 1, 1],
                                  [1, 0, 1],
                                  [1, 1, 1]])

            results = []
            ndimage.generic_filter(temp_grid[:,:,0], self.test_func, mode='reflect', footprint=footprint, extra_arguments=(results,))
            results = np.array(results)
            results = results.reshape(self.config.grid_size, self.config.grid_size, 8)

            nb_grid = np.logical_or(nb_grid,results)*1
        #      # fix for this duplicated TP cases! # TODO tix empty cells! in between lines!
  #      for id in range(0, numb_lane):
  ##          for i in range(0, self.config.grid_size):
  #              for j in range(0, self.config.grid_size):
  #                  if grid_id[id, i, j, 0]:
  #                      check_for_overlapp = np.where(grid_id[id, :, :, 0] == grid_id[id, i, j, 0])
 #                       indezes = np.array(check_for_overlapp).T
 #                       if indezes.shape[0] > 1:
 #                           num = grid_id_sum[id, indezes[:, 0], indezes[:, 1], indezes[:, 2]]
 #                           max = np.argmax(num)
 #                           # now delete the others!
 #                           for i, idx in enumerate(indezes):
 #                               if not i == max:
 #                                   grid[idx[0], idx[1], 0].fill(0.)
 #                                   grid_id[id, idx[0], idx[1], idx[2]].fill(0.)

        for id in range(0, numb_lane):
            for i in range(0, self.config.grid_size):
                for j in range(0, self.config.grid_size):
                    for a in range(1, self.config.num_prediction_cells):
                        if grid_id[id, i, j, a]:
                            check_for_overlapp = np.any(np.where(grid_id[id, :, :, 0] == grid_id[id, i, j, a]))
                            if check_for_overlapp:
                                grid[i, j, a].fill(0.)

    def gen(self, test=False):

        if self.shuffle:
            random.shuffle(self.keys)  # shuffle keys

        for key in self.keys:

            img,lanes = self.read_data(key)

            if test:
                gt_image = img
                gt_lanes = lanes.copy()
            img, lanes_trans = self.transform_data(img,lanes)

            #import timeit
            #start = timeit.default_timer()
            #grid, img = self.gen_line_to_grid_with_rot(lanes, debug=True)
            #grid = self.gen_line_to_grid_with_rot(lanes, debug=False)

            if self.config.onlyLane:
                grid = self.gen_line_to_grid(lanes_trans, debug=True)
            else:
                grid = self.gen_spline_to_grid(lanes_trans, debug=True)
               # grid = self.gen_spline_to_grid(lanes, debug=True)

            #stop = timeit.default_timer()
            #print(stop - start)
            if test:
                yield img, grid, gt_image, gt_lanes
            else:
                yield img, grid

    def read_data(self, key):
        entry = self.datum[0][str(key)]
        dataset_entry = json.loads(entry.value)
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
        img, label = Transformer.transform(img,lanes,self.config, aug)

        return img,lanes

    def batch_gen(self, test=False):

        trainingset_size = self.epoch
        batch_idx_start = 0
        data_gen = self.gen(test=test)
        batch_idx_end = self.batch_size if self.batch_size < trainingset_size else trainingset_size

        while 1:
            if batch_idx_start == batch_idx_end:
                batch_idx_start = 0
                batch_idx_end = self.batch_size if self.batch_size < trainingset_size else trainingset_size
                data_gen = self.gen(test=test)
         #   print(str(batch_idx_end))
            batch_size_round = batch_idx_end - batch_idx_start  # at the end batch can be < batch_size

            final_images = np.ndarray(shape=(batch_size_round, self.image_h, self.image_w, 3), dtype=np.float32)
            if test:
                gt_images = np.ndarray(shape=(batch_size_round, 590, 1640, 3), dtype=np.float32)
                gt_lanes = []
            if not self.config.onlyLane:
                grid_batch = np.ndarray((batch_size_round, self.config.grid_size, self.config.grid_size,
                                         self.config.num_prediction_cells, self.cell_size_grid),
                                        dtype=np.float)
              #  nb_grid_batch = np.ndarray((batch_size_round, self.config.grid_size, self.config.grid_size,
              #                             1),
              #                          dtype=np.float)
            else:
                grid_batch = np.ndarray((batch_size_round,
                                         self.config.num_prediction_cells,
                                         9),
                                        dtype=np.object)  # np.object else for points to grid

            for i in range(batch_size_round):
                if test:
                    image, grid, gt_img, gt_lane = next(data_gen)
                    gt_lanes.append(gt_lane)
                    gt_images[i] = gt_img
                else:
                    image, grid = next(data_gen)
                final_images[i] = image
                grid[:, :, :, -9] = i
                grid[:, :, :, -5] = i
                grid_batch[i] = grid
               # nb_grid_batch[i] = nb_grid

            if self.config.staged:
                staged_batch = []
                grid_batch = np.reshape(grid_batch, (batch_size_round, self.config.grid_size, self.config.grid_size,
                                         self.config.num_prediction_cells *self.cell_size_grid))

                for st in range(self.config.stages):
                    staged_batch.append(grid_batch)
              #  staged_batch.append(grid_batch)
                if test:
                    yield final_images, staged_batch,gt_images, gt_lanes
                else:
                    yield final_images, staged_batch
            else:
                yield final_images, grid_batch

            batch_idx_start = batch_idx_end
            batch_idx_end = batch_idx_end + self.batch_size
            if batch_idx_end > trainingset_size:
                batch_idx_end = trainingset_size

    def batch_gen_inplace(self,val=False, debug=False):
        id = 0
        trainingset_size = self.epoch
        batch_idx_start = 0
        if self.shuffle:
            random.shuffle(self.keys)
        batch_idx_end = self.batch_size if self.batch_size < trainingset_size else trainingset_size

        while 1:

            if batch_idx_start == batch_idx_end:
                batch_idx_start = 0
                batch_idx_end = self.batch_size if self.batch_size < trainingset_size else trainingset_size
                if self.shuffle:
                    random.shuffle(self.keys)
         #   print(str(batch_idx_end))
            batch_size_round = batch_idx_end - batch_idx_start  # at the end batch can be < batch_size

            final_images = np.ndarray(shape=(batch_size_round, self.image_h, self.image_w, 3), dtype=np.float32)

            grid_batch = np.ndarray((batch_size_round, self.config.grid_size, self.config.grid_size,
                                     self.config.num_prediction_cells, self.cell_size_grid),
                                    dtype=np.float)
            if val:
                val_data_batch = np.ndarray((batch_size_round),
                                    dtype=np.object)
                for i, key in enumerate (self.keys[batch_idx_start:batch_idx_end]):
                    image, lanes = self.read_data(key)
                    image, lanes = self.transform_data(image, lanes)
                    grid, val = self.gen_spline_to_grid(lanes,val = True, debug=True)
                    final_images[i] = image
                    val_data_batch[i] = val
                    grid[:, :, :, -9] = i
                    grid[:, :, :, -5] = i
                    grid_batch[i] = grid
                   # nb_grid_batch[i] = nb_grid

            else:
                for i, key in enumerate (self.keys[batch_idx_start:batch_idx_end]):
                    image, lanes = self.read_data(key)
                    image, lanes = self.transform_data(image, lanes)
                    grid = self.gen_spline_to_grid(lanes, debug=True)
                    final_images[i] = image
                    grid[:, :, :, -9] = i
                    grid[:, :, :, -5] = i
                    grid_batch[i] = grid
                   # nb_grid_batch[i] = nb_grid
            if self.config.splitted:
                staged_batch = []
                grid_batch = np.reshape(grid_batch, (self.batch_size, self.config.grid_size, self.config.grid_size,
                                                     self.config.num_prediction_cells * self.cell_size_grid))

                for st in range(self.config.stages):
                    staged_batch.append(grid_batch)
                    staged_batch.append(grid_batch)
                # staged_batch.append(grid_batch)

                if val:
                    yield final_images, staged_batch, val_data_batch
                else:
                    yield final_images, staged_batch

            elif self.config.staged:
                staged_batch = []
                grid_batch = np.reshape(grid_batch, (batch_size_round, self.config.grid_size, self.config.grid_size,
                                         self.config.num_prediction_cells *self.cell_size_grid))

                for st in range(self.config.stages):
                    staged_batch.append(grid_batch)
              #  staged_batch.append(grid_batch)
                if val:
                    yield final_images, staged_batch, val_data_batch
                else:
                    yield final_images, staged_batch
            else:
                yield final_images, grid_batch

            #print('generator yielded a batch %d' % id)
            #id += 1
            batch_idx_start = batch_idx_end
            self.batch_idx_end = batch_idx_end
            batch_idx_end = batch_idx_end + self.batch_size
            if batch_idx_end > trainingset_size:
                batch_idx_end = trainingset_size

if __name__ == "__main__":
    from config import *
    config = Config()
    a = data_gen(dataset=config.CU_val_hdf5_path, config = config, augment=True)
    generator = a.batch_gen_inplace()
    x, y = next(generator)
    print('first-done!')
    x, y = next(generator)

    for k in x:
        plt.imshow(k[:,:,0], cmap='gray')
        plt.show()

