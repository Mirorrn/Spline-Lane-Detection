
import cu__grid_cell.data_gen  as dg
import cu__grid_cell.data_gen_threaded as thr
from cu__grid_cell.custom_loss import *
from cu__grid_cell.preparation import *


model_obj = preparation()



working_path = model_obj.get_working_DIR() # make all prep. work!

config = model_obj.config

val_data_obj = dg.data_gen(dataset=config.CU_val_hdf5_path,shuffle=False, augment=False, config = config, batchsize = 'all', percentage_of_data=config.val_sample_percentage)# care loads howl dataset into ram!

train_data_obj = thr.DataGenerator(dataset = config.CU_tr_hdf5_path,shuffle=True,augment=True, config = config, percentage_of_data=config.train_sample_percentage)

# define model
custom_los_obj = loss(config)

if Config.splitted:
    model_obj.prep_for_training_splitted(config, train_data_obj, val_data_obj, custom_los_obj)
else:
    model_obj.prep_for_training(config, train_data_obj, val_data_obj, custom_los_obj)

model_obj.train()


#def validate(config, model, val_data, validation_steps, metrics_id, epoch):
#    prediction = model.predict(val_data, batch_size=batch)
   # # list all data in history
   # print(history.history.keys())
    # summarize history for accuracy
  #  plt.plot(history.history['loss'])
  #  plt.plot(history.history['val_loss'])
  #  plt.title('model accuracy')
  #  plt.ylabel('accuracy')
  #  plt.xlabel('epoch')
  #  plt.legend(['train', 'test'], loc='upper left')
  #  plt.show()
    # summarize history for loss
  #  plt.plot(history.history['loss'])
  #  plt.plot(history.history['val_loss'])
  #  plt.title('model loss')
  #  plt.ylabel('loss')
  #  plt.xlabel('epoch')
  #  plt.legend(['train', 'test'], loc='upper left')
  #  plt.show()