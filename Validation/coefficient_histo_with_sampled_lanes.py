from cu__grid_cell.data_gen import *
import matplotlib.pyplot as plt
from sklearn import preprocessing

a = data_gen(dataset = cfg.CU_val_hdf5_path)
generator = a.batch_gen()
x_truth, y = next(generator)


#x = np.arange(-2,2,0.1)
a_list = []
b_list = []
c_list = []
flist = []
# debug if everything is right
image = np.zeros((cfg.img_w, cfg.img_h, 1), dtype=np.float32)
for r,row in enumerate(y[0]):
    for c, col in enumerate(row):
        if col[-1]:
            y_f = col[:-1] #+ r + cfg.grich_anchor_pt
           # z = np.polyfit(x, y_f, 2)
           # a, b, k = z[:]
           # y_f = np.round(a * x ** 2 + b * x + k)
 #           x_t = np.array( x + c*cfg.grid_cel_size + cfg.grich_anchor_pt)
            y_f = [x + (c * cfg.grid_cel_size + cfg.grich_anchor_pt, r * cfg.grid_cel_size + cfg.grich_anchor_pt) for
                      x in y_f]  # normalize to grid cell
            flist.append(y_f)
            image = cv2.polylines(image, np.int32([y_f]), 0, 1,thickness=1)

fig, axarr = plt.subplots(1, 2)
test_img = image[:,:,0]
axarr[0].imshow(test_img, cmap='gray')
axarr[1].imshow(x_truth[0,:,:,0], cmap='gray')
plt.show()



for batch in y:
    for row in batch:
        for col in row:
            if col[-1]:
                x =  np.asarray(col[:-1])[:,0]
                y_f =  np.asarray(col[:-1])[:,1]

                z = np.polyfit(x, y_f, 2)
                p = np.poly1d(z)
                a, b, c = z[:]
                a_list.append(a)
                b_list.append(b)
                c_list.append(c)

         #       import matplotlib.pyplot as plt

         #       xp = np.linspace(-10, 10, 100)
         #       _ = plt.plot(x, y_f, '.', xp, p(xp), '-')

  #              plt.show()

                test = 0


a_list = np.expand_dims(np.array(a_list), axis=-1)
a_range = a_list.max(axis=0) - a_list.min(axis=0)
print('a_range = ' + str(a_range[0]))
a_shift = a_list.min(axis=0)
print('a_shift = ' + str(a_shift[0]))
a_list = (a_list - a_shift) / (a_range)


b_list = np.expand_dims(np.array(b_list), axis=-1)
b_range = b_list.max(axis=0) - b_list.min(axis=0)
print('b_range = ' + str(b_range[0]))
b_shift = b_list.min(axis=0)
print('b_shift = ' + str(b_shift[0]))
b_list = (b_list - b_shift) / (b_range)

c_list = np.expand_dims(np.array(c_list), axis=-1)
c_range = c_list.max(axis=0) - c_list.min(axis=0)
print('c_range = ' + str(c_range[0]))
c_shift = c_list.min(axis=0)
print('c_shift = ' + str(c_shift[0]))
c_list = (c_list - c_shift) / (c_range)


plt.hist(a_list, bins='auto')  # arguments are passed to np.histogram
plt.show()
plt.hist(b_list, bins='auto')  # arguments are passed to np.histogram
plt.show()
plt.hist(c_list, bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram with 'auto' bins")

plt.show()

test = 0