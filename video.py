import os
path = os.getcwd()
from cu__grid_cell.data_gen import data_gen
from cu__grid_cell.preparation import preparation
import numpy as np
from cu__grid_cell.Validation.validation_utils import plot_image, grid_based_eval_with_iou, plot_image3d, nms, concatenate_cells
import matplotlib.pyplot as plt
import cv2
from keras.applications.vgg16 import preprocess_input

img_w = 1640
img_h = 590

### KITTI

#img_w = 1242
#img_h = 375

def norm_img_vgg(img):
    # info = np.iinfo(img.dtype)
    img = np.array(img, dtype=np.float32)
    img = preprocess_input(img)  # care bgr -> rgb
    img = img[..., ::-1]  # make rgb to bgr again, because of opencv
    return img

model_obj = preparation(testing = True)
config = model_obj.config

scale_size_y =  config.img_w / (img_w -1)
scale_size_x =  config.img_h / (img_h -1)

M = np.array([[scale_size_y, 0, 0],
                          [0, scale_size_x, 0],
                          [0, 0, 1.]])
M=M[0:2]

scale_size_y =  (img_w -1) / config.img_w
scale_size_x =  (img_h -1) / config.img_h

M2 = np.array([[scale_size_y, 0, 0],
                          [0, scale_size_x, 0],
                          [0, 0, 1.]])
M2=M2[0:2]
cap = cv2.VideoCapture('/home/ehealth/Desktop/test/6/output.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 10.0, (img_w,img_h) )
while(cap.isOpened()):
    ret, frame = cap.read()
    img = cv2.warpAffine(frame, M, (config.img_w, config.img_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                         borderValue=(127, 127, 127))
    img = norm_img_vgg(img)
    pred = prediction = model_obj.predict(img[np.newaxis, :] )
    pred = nms(pred, config)
    lanes_pred = concatenate_cells(pred, config, prediction=True)
    original_points = lanes_pred
    for j, o in enumerate(original_points):
        o = np.array(o).T
        ones = np.ones_like(o[:, 0])
        ones = ones[..., None]
        original_points[j] = np.concatenate((o, ones),
                                            axis=1)  # we reuse 3rd column in completely different way here, it is hack for matmul with M
        original_points[j] = np.matmul(M2, original_points[j].T).T  # transpose for multiplikation

    lanes = original_points  # take only coords!
    for a in lanes:
        frame = cv2.polylines(frame, np.int32([a]), isClosed=0,color=(0,0,255), thickness=8)
    #img = cv2.warpAffine(img, M2, (config.img_w, config.img_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
    #                     borderValue=(127, 127, 127))
    out.write(frame)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()