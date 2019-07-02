
import cu__grid_cell.data_gen  as dg
import cu__grid_cell.data_gen_threaded as thr
from cu__grid_cell.custom_loss import *
from cu__grid_cell.preparation import *



model_obj = preparation()



working_path = model_obj.get_working_DIR() # make all prep. work!

config = model_obj.config

val_data_obj = dg.data_gen(dataset=config.CU_test4_noline_hdf5_path,shuffle=False, augment=False, config = config, batchsize = 50, percentage_of_data=config.val_sample_percentage)# care loads howl dataset into ram!

model_obj.test_val(val_data_obj, 50)

val_data_obj = dg.data_gen(dataset=config.CU_test5_arrow_hdf5_path,shuffle=False, augment=False, config = config, batchsize = 50, percentage_of_data=config.val_sample_percentage)# care loads howl dataset into ram!

model_obj.test_val(val_data_obj, 50)

val_data_obj = dg.data_gen(dataset=config.CU_test6_curve_hdf5_path,shuffle=False, augment=False, config = config, batchsize = 50, percentage_of_data=config.val_sample_percentage)# care loads howl dataset into ram!

model_obj.test_val(val_data_obj, 50)

val_data_obj = dg.data_gen(dataset=config.CU_test8_night_hdf5_path,shuffle=False, augment=False, config = config, batchsize = 50, percentage_of_data=config.val_sample_percentage)# care loads howl dataset into ram!

model_obj.test_val(val_data_obj, 50)
