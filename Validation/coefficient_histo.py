from cu__grid_cell.data_gen import *
import matplotlib.pyplot as plt
from sklearn import preprocessing
from config import Config

cfg = Config()
batch = 100

a = data_gen(dataset=cfg.CU_tr_hdf5_path, batchsize=batch, config=cfg)
generator = a.batch_gen_inplace()
x_img, y = next(generator)

#x = np.arange(0, 2* cfg.grich_anchor_pt, 1/cfg.num_of_samples)

#x = np.arange(-cfg.grich_anchor_pt, cfg.grich_anchor_pt, 1/cfg.num_of_samples)
a_list = []
b_list = []
c_list = []
d_list = []
line = []

for batch in y:
    for row in batch:
        for col in row:
            for pre in col:
                if pre[-1]:
                    x_t = pre[0:cfg.grid_cel_size + 1]  # + c * cfg.grid_cel_size + cfg.grich_anchor_pt)
                    y_t = pre[cfg.grid_cel_size + 1:2 * (cfg.grid_cel_size) + 2]
                   # idx = np.logical_not(np.isnan(y_t))
                    a, b, c = np.polyfit(x_t, y_t, 2)
                    a_list.append(a)
                    b_list.append(b)
                    c_list.append(c)
                  #  d_list.append(1)

 #                   line.append(y_f)

                  #  if c > 300:
                  #      p = np.poly1d(z)
                  #      plt.plot(p(x), x, '+')
                  #      plt.show()



a_list = np.expand_dims(np.array(a_list), axis=-1)
a_range = a_list.max(axis=0) - a_list.min(axis=0)
a_range = np.var(a_list,1)
a_mean = np.mean(a_list)
print('self.a_Mean = ' + str(a_mean))
print('self.a_range = ' + str(a_range * 3))
a_shift = a_list.min(axis=0)
print('self.a_shift = ' + str(a_shift[0]))
#a_list = (a_list - a_shift) / (a_range)

b_list = np.expand_dims(np.array(b_list), axis=-1)
#b_range = b_list.max(axis=0) - b_list.min(axis=0)
b_range = np.var(b_list,1)
b_mean = np.mean(b_list)
print('self.b_Mean = ' + str(b_mean))
print('self.b_range = ' + str(b_range * 3))
b_shift = b_list.min(axis=0)
print('self.b_shift = ' + str(b_shift[0]))
#b_list = (b_list - b_shift) / (b_range)

c_mean = np.mean(c_list)
print('self.c_Mean = ' + str(c_mean))

c_list = np.expand_dims(np.array(c_list), axis=-1)
c_range = c_list.max(axis=0) - c_list.min(axis=0)
c_range = np.var(c_list,1)
print('self.c_range = ' + str(c_range * 3))
c_shift = c_list.min(axis=0)
print('self.c_shift = ' + str(c_shift[0]))
#c_list = (c_list - c_shift) / (c_range)

#d_list = np.expand_dims(np.array(d_list), axis=-1)
#d_range = d_list.max(axis=0) - d_list.min(axis=0)
#print('d_range = ' + str(d_range[0]))
#d_shift = d_list.min(axis=0)
#print('d_shift = ' + str(d_shift[0]))
#d_list = (d_list - d_shift) / (d_range)

#line = np.array(line)

plt.hist(a_list, bins='auto')  # arguments are passed to np.histogram
plt.title("a koeff")
plt.show()
plt.hist(b_list, bins='auto')  # arguments are passed to np.histogram
plt.title("b koeff")
plt.show()
plt.hist(c_list, bins='auto')  # arguments are passed to np.histogram
plt.title("c koeff")

#plt.hist(d_list, bins='auto')  # arguments are passed to np.histogram
#plt.title("c koeff")

#plt.hist(line, bins='auto')  # arguments are passed to np.histogram
#plt.title("Line distribution")

plt.show()

# debug if everything is right

for y_i in y[0:100]:
    image = np.zeros((cfg.img_w, cfg.img_h, 1), dtype=np.float32)
    for r,row in enumerate(y_i):
        for c, col in enumerate(row):
            for pre in col:
                if pre[-1]:
                   # y_f = pre[:-1]  #+ (r*cfg.grid_cel_size + cfg.grich_anchor_pt) #+ r + cfg.grich_anchor_pt
                    #if np.any(y_f >= 250):
                    x_t = pre[0:cfg.grid_cel_size+1] # + c * cfg.grid_cel_size + cfg.grich_anchor_pt)
                    y_t = pre[cfg.grid_cel_size + 1:2*(cfg.grid_cel_size) + 2]
                  #  a, b, c = np.polyfit(x_t[idx], y_t[idx], 2)
                  #  y_t = a* x_t**2 + b * x_t + c
                    y_f, x_t = x_t, y_t  # for ratoation
                    x_t = np.array(x_t + c * (cfg.grid_cel_size) + cfg.grich_anchor_pt)
                    y_f = np.array(y_f + r * (cfg.grid_cel_size) + cfg.grich_anchor_pt)

                    f = np.array([x_t, y_f]).T
                    image = cv2.polylines(image, np.int32([f]), 0, 1,thickness=1)

        #fig, axarr = plt.subplots(1, 2)
    plt.imshow(image[:,:,0], cmap='gray')
#        plt.set_title('From Grid', color='0.7')

    #axarr[1].imshow(x_img[0].astype(np.uint8))
    #axarr[1].set_title('Input Img')
    plt.show()

test = 0