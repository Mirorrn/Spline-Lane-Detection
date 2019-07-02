#!/usr/bin/env python

import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.applications.vgg16 import preprocess_input
from math import cos, sin, pi
from imgaug import augmenters as iaa

import cu__grid_cell.Automold as am
class AugmentSelection:

    def __init__(self, flip=False, augment_light = False,):
        self.flip = flip # shift y-axis
        self.aug_light = augment_light

    @staticmethod
    def random():
        flip = random.uniform(0., 1.) >= 0.5
        aug_light = True
        return AugmentSelection(flip, aug_light)

    @staticmethod
    def unrandom():
        flip = False
        aug_light = False
        return AugmentSelection(flip, aug_light)

    def affine(self, config):
        # the main idea: we will do all image transformations with one affine matrix.
        # this saves lot of cpu and make code significantly shorter
        # same affine matrix could be used to transform joint coordinates afterwards
        # look https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/warp_affine/warp_affine.html


      #  width, height = img_shape
        degree = random.uniform(-1., 1.) * 5 if self.flip else 0.
        #degree = 7.

        A = cos(degree / 180. * pi)
        B = sin(degree / 180. * pi)


        rotate = np.array([[A, -B, 0],
                           [B, A, 0],
                           [0, 0, 1.]])


        scale_size_y =  config.img_w / (1640 -1)
        scale_size_x =  config.img_h / (590 -1)

        center2zero = np.array([[1., 0., -(1640 /2 - 1)],
                                [0., 1., -(590 /2 - 1)],
                                [0., 0., 1.]])


        scale = np.array([[scale_size_y, 0, 0],
                          [0, scale_size_x, 0],
                          [0, 0, 1.]])

        flip = -1. if self.flip else 1.
        flip = np.array([[flip, 0., 0.],
                         [0., 1., 0.],
                         [0., 0., 1.]], dtype=np.float32)

        translatx =  random.uniform(-1., 1.) * 15 if self.flip else 0.
        translaty =  random.uniform(-1., 1.) * 15 if self.flip else 0.

        center2center = np.array([[1., 0., (config.img_w /2 - 1 + translatx)],
                                  [0., 1., (config.img_h /2 - 1 + translaty)],
                                  [0., 0., 1.]])

        # order of combination is reversed
        #combined =   center2center  @ rotate @ scale @ flip @center2zero# @ - matmul
        combined = center2center @ scale @ flip @rotate @ center2zero  # @ - matmul
        # combined = center2center @ center2zero  # @ - matmul

        return combined[0:2]  # 3th row is not important anymore

seq = iaa.Sequential([
    iaa.Add((-10, 10), per_channel=0.5),  # change brightness of images (by -10 to 10 of original value)
    iaa.AddToHueAndSaturation((-20, 20),)
])

class Transformer:
    @staticmethod
    def print_canvas(img, anno, config):  # only for debug

        for a in anno:
            f = a
            img = cv2.polylines(img, np.int32([f]), 0, 1, thickness=1)

        img = img[:, :, ::-1]
        #if config.debug_datagen:
        plt.imshow(img)
        plt.show()

    @staticmethod
    def norm_img_vgg(img):
        #info = np.iinfo(img.dtype)
        img = np.array(img, dtype=np.float32)
        img  = preprocess_input(img) # care bgr -> rgb
        img  = img[..., ::-1] #make rgb to bgr again, because of opencv
        return img

    @staticmethod
    def norm_img_mobile(img):
        #info = np.iinfo(img.dtype)
        img = np.array(img, dtype=np.float32)
        img  = img/127.5 - 1.0 # care bgr -> rgb
        img  = img[..., ::-1] #make rgb to bgr again, because of opencv
        return img

    @staticmethod
    def augment_brightness_camera_images(img):
        img = seq.augment_image(img)
        return img

    @staticmethod
    def transform(img, lanes, config, aug='', for_Test=False):

        if for_Test:
            test_img = img.copy.deepcopy()

        if not aug:
            aug = AugmentSelection.random()

        # warp picture and mask
        M = aug.affine(config)


        img = cv2.warpAffine(img, M, (config.img_w, config.img_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(127, 127, 127))

        if aug.aug_light:
            img = Transformer.augment_brightness_camera_images(img)
            #img = am.add_gravel(img)

        # warp key points
        original_points = lanes
        for i, o in enumerate(original_points):
            ones = np.ones_like(o[:,0])
            ones = ones[...,None]
            original_points[i] = np.concatenate((o, ones), axis=1) # we reuse 3rd column in completely different way here, it is hack for matmul with M
            original_points[i] = np.matmul(M, original_points[i].T).T # transpose for multiplikation

        lanes = original_points # take only coords!

    #    Transformer.print_canvas(img, lanes, config) # only for debbuging
        if not config.mobile:
            img = Transformer.norm_img_vgg(img)
        else:
            img = Transformer.norm_img_mobile(img)

        if for_Test:
            return img, lanes, test_img

        return img, lanes

