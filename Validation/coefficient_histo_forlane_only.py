import pickle

from cu__grid_cell.data_gen import *
import matplotlib.pyplot as plt
from sklearn import preprocessing
from config import Config

from sklearn.cluster import KMeans
import pandas as pd
from scipy.optimize import linear_sum_assignment

#K means Clustering
def doKmeans(X, nclust=2):
    model = KMeans(nclust)
    model.fit(X)
    clust_labels = model.predict(X)
    cent = model.cluster_centers_
    return (clust_labels, cent, model)

cfg = Config()
batch = 100

a = data_gen(dataset=cfg.CU_tr_hdf5_path, batchsize=batch, config=cfg)
generator = a.batch_gen()
x_img, y = next(generator)

a_list = []
b_list = []
c_list = []
mean = []
norm = cfg.img_h //2
line = []
x_hole_lane = np.arange(0, cfg.img_h, cfg.grid_cel_size)
for batch in y:
    for pre in batch:
        if pre[-1]:
            x = np.asarray(pre[0:3], dtype=float)  # * cfg.img_h + norm
            y_f = np.asarray(pre[3:6], dtype=float)  # * cfg.img_h + norm

            z = np.polyfit(x, y_f, 2)
            a, b, c = z[:]
            a_list.append(a)
            b_list.append(b)
            c_list.append(c)

            line.append(y_f)
            mean.append([pre[-2], pre[-3]]) # switched x and y for rotation to [x,y]!
            test = 0
           # p = np.poly1d(z)
           # plt.plot(p(x), x, '+')
           # plt.show()


mean= np.array(mean)
clust_labels, cent, model = doKmeans(mean, 4)
kmeans = pd.DataFrame(clust_labels)

fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(mean[:,0], mean[:,1],
                     c=clust_labels,s=50)
ax.set_title('K-Means Clustering')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.colorbar(scatter)

plt.show()
filename_center = cfg.DIR + cfg.experiment_name + '/cluster_cent.sav'
filename = cfg.DIR + cfg.experiment_name + '/cluster.sav'
#pickle.dump(model, open(filename, 'wb'))
#pickle.dump(cent, open(filename_center, 'wb'))
print('MODEL saved!!!!')


a_list = np.array(a_list)
a_range = a_list.max() - a_list.min()
print('self.a_range = ' + str(a_range))
a_shift = a_list.min()
print('self.a_shift = ' + str(a_shift))
a_list = (a_list - a_shift) / (a_range)

b_list = np.array(b_list)
b_range = b_list.max() - b_list.min()
print('self.b_range = ' + str(b_range))
b_shift = b_list.min()
print('self.b_shift = ' + str(b_shift))
b_list = (b_list - b_shift) / (b_range)

c_list = np.array(c_list)
c_range = c_list.max() - c_list.min()
print('self.c_range = ' + str(c_range))
c_shift = c_list.min()
print('self.c_shift = ' + str(c_shift))
c_list = (c_list - c_shift) / (c_range)


line = np.array(line)

#plt.hist(a_list, bins='auto')  # arguments are passed to np.histogram
#plt.title("a koeff")
#plt.show()
#plt.hist(b_list, bins='auto')  # arguments are passed to np.histogram
#plt.title("b koeff")
#plt.show()
#plt.hist(c_list, bins='auto')  # arguments are passed to np.histogram
#plt.title("c koeff")

#plt.hist(line, bins='auto')  # arguments are passed to np.histogram
#plt.title("Line distribution")

#plt.show()



for i,p in enumerate(y[0:10]):
    image = np.zeros((cfg.img_w, cfg.img_h, 1), dtype=np.float32)
    for pre in p:
        if pre[-1]:
            x = np.asarray(pre[0:3], dtype=float) * cfg.img_h + norm
            y_f = np.asarray(pre[3:6], dtype=float) * cfg.img_h + norm

           # z = np.polyfit(x, y_f, 2)
           # a, b, c = z[:]
            #y_f = a * x ** 2 + b * x + c
            f = np.array([x, y_f]).T
            image = cv2.polylines(image, np.int32([f]), 0, 1, thickness=1)
            predicted = model.transform(np.array([pre[-2], pre[-3]]).reshape(1, -1))
            predicted = np.argsort(predicted[0])

            test = model.predict(np.array([pre[-2], pre[-3]]).reshape(1, -1))


            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10, 500)
            fontScale = 1
            fontColor = (1, 1, 1)
            lineType = 2

            cv2.putText(image, str(test[0]),
                        (int(pre[-2]), int(pre[-3])),
                        font,
                        fontScale,
                        fontColor,
                        lineType)

            cv2.putText(image, str(predicted[0]),
                        (int(pre[-2]), int(pre[-3])),
                        font,
                        fontScale,
                        fontColor,
                        lineType)


    fig, axarr\
        = plt.subplots(1, 2)
    axarr[0].imshow(image[:,:,0], cmap='gray')
    axarr[0].set_title('From Grid', color='0.7')
    axarr[1].imshow(x_img[i, :,:,0], cmap='gray')
    axarr[1].set_title('Input Img')
plt.show()