import numpy as np
import numpy.random as rng
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import Binarizer
import h5py as h5
import matplotlib.pyplot as plt
from my_RBM_tf2_test import RBM
from optimizer import Optimizer
from datasets.bas_data import get_data
import pandas as pd
import datetime
from utils import plot_image_grid, plot_single_image, plot_input_sample

#This needs to be done after running data_collection_a1.py, 
# not really sure how to automate this efficiently without altering the source heavily


########################################################################
#                                   BAS                                #
########################################################################
x_test_bas = get_data(rng,s=2)
x_train_bas = get_data(rng,s=2)


machine = RBM(4, 6,1000,(2,2), 6)
machine.from_saved_model('D:\Desktop\Spring 2024\ML\RBM_NEEDS_TO_WORK\\results\models0609/0609-024458model.h5')
image_shape = (2,2)
weights = np.asarray(machine.weights)
vis_bias = np.asarray(machine.visible_biases)
hid_bias = np.asarray(machine.hidden_biases)
#print(weights)
#plot_image_grid(weights, image_shape, 6)

visible_states_1,visible_probabilities_1,inpt = machine.parallel_sample(inpt = x_test_bas,n_step_MC=10)
#print out the result
#plot_image_grid(x_test_bas, image_shape, 6,name='TestGrid'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.pdf')
plot_image_grid(visible_probabilities_1, image_shape, 6,name='BAS2x2Temp300A'+'.pdf')
#



# machine.from_saved_model('D:\Desktop\Spring 2024\ML\RBM_NEEDS_TO_WORK\\results\models2208/2208-140223model.h5')
# image_shape = (2,2)
# weights = np.asarray(machine.weights)
# #vis_bias = np.asarray(machine.visible_biases)
# hid_bias = np.asarray(machine.hidden_biases)
# #print(weights)
# #plot_image_grid(weights, image_shape, 6)

# visible_states_1,visible_probabilities_1,inpt = machine.parallel_sample(inpt = x_test_bas,n_step_MC=10)
# #print out the result
# #plot_image_grid(x_test_bas, image_shape, 6,name='TestGrid'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.pdf')
# plot_image_grid(visible_probabilities_1, image_shape, 6,name='BAS2x2Temp150A'+'.pdf')
# #


# machine.from_saved_model('D:\Desktop\Spring 2024\ML\RBM_NEEDS_TO_WORK\\results\models0509/0509-214351model.h5')
# image_shape = (2,2)
# weights = np.asarray(machine.weights)
# vis_bias = np.asarray(machine.visible_biases)
# hid_bias = np.asarray(machine.hidden_biases)
# #print(weights)
# #plot_image_grid(weights, image_shape, 6)

# visible_states_1,visible_probabilities_1,inpt = machine.parallel_sample(inpt = x_test_bas,n_step_MC=10)
# #print out the result
# #plot_image_grid(x_test_bas, image_shape, 6,name='TestGrid'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.pdf')
# plot_image_grid(visible_probabilities_1, image_shape, 6,name='1kBAS2x2Temp150'+'.pdf')

# machine.from_saved_model('D:\Desktop\Spring 2024\ML\RBM_NEEDS_TO_WORK\\results\models0509/0509-222121model.h5')
# image_shape = (2,2)
# weights = np.asarray(machine.weights)
# vis_bias = np.asarray(machine.visible_biases)
# hid_bias = np.asarray(machine.hidden_biases)
# #print(weights)
# #plot_image_grid(weights, image_shape, 6)

# visible_states_1,visible_probabilities_1,inpt = machine.parallel_sample(inpt = x_test_bas,n_step_MC=10)
# #print out the result
# #plot_image_grid(x_test_bas, image_shape, 6,name='TestGrid'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.pdf')
# plot_image_grid(visible_probabilities_1, image_shape, 6,name='1kBAS2x2Temp175'+'.pdf')

# machine.from_saved_model('D:\Desktop\Spring 2024\ML\RBM_NEEDS_TO_WORK\\results\models0509/0509-225928model.h5')
# image_shape = (2,2)
# weights = np.asarray(machine.weights)
# vis_bias = np.asarray(machine.visible_biases)
# hid_bias = np.asarray(machine.hidden_biases)
# #print(weights)
# #plot_image_grid(weights, image_shape, 6)

# visible_states_1,visible_probabilities_1,inpt = machine.parallel_sample(inpt = x_test_bas,n_step_MC=10)
# #print out the result
# #plot_image_grid(x_test_bas, image_shape, 6,name='TestGrid'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.pdf')
# plot_image_grid(visible_probabilities_1, image_shape, 6,name='1kBAS2x2Temp200'+'.pdf')

# machine.from_saved_model('D:\Desktop\Spring 2024\ML\RBM_NEEDS_TO_WORK\\results\models0609/0509-233729model.h5')
# image_shape = (2,2)
# weights = np.asarray(machine.weights)
# vis_bias = np.asarray(machine.visible_biases)
# hid_bias = np.asarray(machine.hidden_biases)
# #print(weights)
# #plot_image_grid(weights, image_shape, 6)

# visible_states_1,visible_probabilities_1,inpt = machine.parallel_sample(inpt = x_test_bas,n_step_MC=10)
# #print out the result
# #plot_image_grid(x_test_bas, image_shape, 6,name='TestGrid'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.pdf')
# plot_image_grid(visible_probabilities_1, image_shape, 6,name='1kBAS2x2Temp300'+'.pdf')

# machine.from_saved_model('D:\Desktop\Spring 2024\ML\RBM_NEEDS_TO_WORK\\results\models0609/0609-001503model.h5')
# image_shape = (2,2)
# weights = np.asarray(machine.weights)
# vis_bias = np.asarray(machine.visible_biases)
# hid_bias = np.asarray(machine.hidden_biases)
# #print(weights)
# #plot_image_grid(weights, image_shape, 6)

# visible_states_1,visible_probabilities_1,inpt = machine.parallel_sample(inpt = x_test_bas,n_step_MC=10)
# #print out the result
# #plot_image_grid(x_test_bas, image_shape, 6,name='TestGrid'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.pdf')
# plot_image_grid(visible_probabilities_1, image_shape, 6,name='1kBAS2x2Temp130A'+'.pdf')

# machine.from_saved_model('D:\Desktop\Spring 2024\ML\RBM_NEEDS_TO_WORK\\results\models0609/0609-005233model.h5')
# image_shape = (2,2)
# weights = np.asarray(machine.weights)
# vis_bias = np.asarray(machine.visible_biases)
# hid_bias = np.asarray(machine.hidden_biases)
# #print(weights)
# #plot_image_grid(weights, image_shape, 6)

# visible_states_1,visible_probabilities_1,inpt = machine.parallel_sample(inpt = x_test_bas,n_step_MC=10)
# #print out the result
# #plot_image_grid(x_test_bas, image_shape, 6,name='TestGrid'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.pdf')
# plot_image_grid(visible_probabilities_1, image_shape, 6,name='1kBAS2x2Temp150A'+'.pdf')

# machine.from_saved_model('D:\Desktop\Spring 2024\ML\RBM_NEEDS_TO_WORK\\results\models0609/0609-013003model.h5')
# image_shape = (2,2)
# weights = np.asarray(machine.weights)
# vis_bias = np.asarray(machine.visible_biases)
# hid_bias = np.asarray(machine.hidden_biases)
# #print(weights)
# #plot_image_grid(weights, image_shape, 6)

# visible_states_1,visible_probabilities_1,inpt = machine.parallel_sample(inpt = x_test_bas,n_step_MC=10)
# #print out the result
# #plot_image_grid(x_test_bas, image_shape, 6,name='TestGrid'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.pdf')
# plot_image_grid(visible_probabilities_1, image_shape, 6,name='1kBAS2x2Temp175A'+'.pdf')

# machine.from_saved_model('D:\Desktop\Spring 2024\ML\RBM_NEEDS_TO_WORK\\results\models0609/0609-020729model.h5')
# image_shape = (2,2)
# weights = np.asarray(machine.weights)
# vis_bias = np.asarray(machine.visible_biases)
# hid_bias = np.asarray(machine.hidden_biases)
# #print(weights)
# #plot_image_grid(weights, image_shape, 6)

# visible_states_1,visible_probabilities_1,inpt = machine.parallel_sample(inpt = x_test_bas,n_step_MC=10)
# #print out the result
# #plot_image_grid(x_test_bas, image_shape, 6,name='TestGrid'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.pdf')
# plot_image_grid(visible_probabilities_1, image_shape, 6,name='1kBAS2x2Temp200A'+'.pdf')

# machine.from_saved_model('D:\Desktop\Spring 2024\ML\RBM_NEEDS_TO_WORK\\results\models0609/0609-024458model.h5')
# image_shape = (2,2)
# weights = np.asarray(machine.weights)
# vis_bias = np.asarray(machine.visible_biases)
# hid_bias = np.asarray(machine.hidden_biases)
# #print(weights)
# #plot_image_grid(weights, image_shape, 6)

# visible_states_1,visible_probabilities_1,inpt = machine.parallel_sample(inpt = x_test_bas,n_step_MC=10)
# #print out the result
# #plot_image_grid(x_test_bas, image_shape, 6,name='TestGrid'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.pdf')
# plot_image_grid(visible_probabilities_1, image_shape, 6,name='1kBAS2x2Temp300A'+'.pdf')
# ########

# x_test_bas = get_data(rng,s=3)
# x_train_bas = get_data(rng,s=3)

# machine = RBM(9, 14,1000,(3,3), 14)

# machine.from_saved_model('D:\Desktop\Spring 2024\ML\RBM_NEEDS_TO_WORK\\results\models0609/0609-032225model.h5')
# image_shape = (3,3)
# weights = np.asarray(machine.weights)
# vis_bias = np.asarray(machine.visible_biases)
# hid_bias = np.asarray(machine.hidden_biases)
# #print(weights)
# #plot_image_grid(weights, image_shape, 6)

# visible_states_1,visible_probabilities_1,inpt = machine.parallel_sample(inpt = x_test_bas,n_step_MC=10)
# #print out the result
# plot_image_grid(x_test_bas, image_shape, 14,name='BAS3x3Input'+'.pdf')
# plot_image_grid(visible_probabilities_1, image_shape, 14,name='2kBAS3x3Temp100'+'.pdf')

# machine.from_saved_model('D:\Desktop\Spring 2024\ML\RBM_NEEDS_TO_WORK\\results\models0609/0609-070136model.h5')
# image_shape = (3,3)
# weights = np.asarray(machine.weights)
# vis_bias = np.asarray(machine.visible_biases)
# hid_bias = np.asarray(machine.hidden_biases)
# #print(weights)
# #plot_image_grid(weights, image_shape, 6)

# visible_states_1,visible_probabilities_1,inpt = machine.parallel_sample(inpt = x_test_bas,n_step_MC=10)
# #print out the result
# #plot_image_grid(x_test_bas, image_shape, 6,name='TestGrid'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.pdf')
# plot_image_grid(visible_probabilities_1, image_shape, 14,name='2kBAS3x3Temp200'+'.pdf')

# machine.from_saved_model('D:\Desktop\Spring 2024\ML\RBM_NEEDS_TO_WORK\\results\models0609/0609-103635model.h5')
# image_shape = (3,3)
# weights = np.asarray(machine.weights)
# vis_bias = np.asarray(machine.visible_biases)
# hid_bias = np.asarray(machine.hidden_biases)
# #print(weights)
# #plot_image_grid(weights, image_shape, 6)

# visible_states_1,visible_probabilities_1,inpt = machine.parallel_sample(inpt = x_test_bas,n_step_MC=10)
# #print out the result
# #plot_image_grid(x_test_bas, image_shape, 6,name='TestGrid'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.pdf')
# plot_image_grid(visible_probabilities_1, image_shape, 14,name='2kBAS3x3Temp300'+'.pdf')

# machine.from_saved_model('D:\Desktop\Spring 2024\ML\RBM_NEEDS_TO_WORK\\results\models3008\\3008-012504model.h5')
# image_shape = (3,3)
# weights = np.asarray(machine.weights)
# vis_bias = np.asarray(machine.visible_biases)
# hid_bias = np.asarray(machine.hidden_biases)
# #print(weights)D:\Desktop\Spring 2024\ML\RBM_NEEDS_TO_WORK\results\models3008\3008-012504model.h5
# #plot_image_grid(weights, image_shape, 6)

# visible_states_1,visible_probabilities_1,inpt = machine.parallel_sample(inpt = x_test_bas,n_step_MC=10)
# #print out the result
# #plot_image_grid(x_test_bas, image_shape, 14,name='TestGrid'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.pdf')
# plot_image_grid(visible_probabilities_1, image_shape, 14,name='BAS3x3Temp200A'+'.pdf')

# machine.from_saved_model('D:\Desktop\Spring 2024\ML\RBM_NEEDS_TO_WORK\\results\models0609/0609-141640model.h5')
# image_shape = (3,3)
# weights = np.asarray(machine.weights)
# vis_bias = np.asarray(machine.visible_biases)
# hid_bias = np.asarray(machine.hidden_biases)
# #print(weights)
# #plot_image_grid(weights, image_shape, 6)

# visible_states_1,visible_probabilities_1,inpt = machine.parallel_sample(inpt = x_test_bas,n_step_MC=10)
# #print out the result
# #plot_image_grid(x_test_bas, image_shape, 6,name='TestGrid'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.pdf')
# plot_image_grid(visible_probabilities_1, image_shape, 14,name='2kBAS3x3Temp200A'+'.pdf')
# #
# machine.from_saved_model('D:\Desktop\Spring 2024\ML\RBM_NEEDS_TO_WORK\\results\models0609/0609-174215model.h5')
# image_shape = (3,3)
# weights = np.asarray(machine.weights)
# vis_bias = np.asarray(machine.visible_biases)
# hid_bias = np.asarray(machine.hidden_biases)
# #print(weights)
# #plot_image_grid(weights, image_shape, 6)

# visible_states_1,visible_probabilities_1,inpt = machine.parallel_sample(inpt = x_test_bas,n_step_MC=10)
# #print out the result
# #plot_image_grid(x_test_bas, image_shape, 6,name='TestGrid'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.pdf')
# plot_image_grid(visible_probabilities_1, image_shape, 14,name='2kBAS3x3Temp300A'+'.pdf')
# print('BAS Finished!')

# ########################################################################
# #                                 MNIST                                #
# ########################################################################
# mnist = tf.keras.datasets.mnist

# #Split in test and train
# (x_train_M, y_train_M), (x_test_M, y_test_M) = mnist.load_data()

# #Scale entries between(0,1)
# x_train_M = x_train_M/255
# x_test_M = x_test_M/255

# #Binarize pictures
# binarizer = Binarizer(threshold=0.5)
# x_train_binary_M = np.array([binarizer.fit_transform(slice) for slice in x_train_M])
# x_test_binary_M = np.array([binarizer.fit_transform(slice) for slice in x_test_M])

# #reshape pictures to be vectors and fix datatype
# x_train_binary_M = x_train_binary_M.reshape(x_train_binary_M.shape[0],-1).astype(np.float64)
# x_test_binary_M = x_test_binary_M.reshape(x_test_binary_M.shape[0],-1).astype(np.float64)
# #above is just to get the right model configuration
# np.random.shuffle(x_test_binary_M)
# machine = RBM(x_train_binary_M[0].shape[0], 100,100,(28,28), 128)
# machine.from_saved_model('D:\Desktop\Spring 2024\ML\RBM_NEEDS_TO_WORK\\results\models1207/1207-093542model.h5')
# image_shape_2 = (28,28)
# weights = np.asarray(machine.weights)
# #print(weights)
# #print(weights)
# #plot_image_grid(weights, image_shape_2, 9)
# #update input with visible state
# visible_states_1,visible_probabilities_1,inpt = machine.parallel_sample(inpt = x_test_binary_M[0:20],n_step_MC=20)
# #print out the result
# plot_image_grid(x_test_binary_M[0:20], image_shape_2, 20,name='TestGrid'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.pdf')
# plot_image_grid(visible_probabilities_1[0:20], image_shape_2, 20,name='MNistTemp300C'+'.pdf')

# #plot_image_grid(weights, image_shape_2, 30)

# #
# #D:\Desktop\Spring 2024\ML\RBM_NEEDS_TO_WORK\results\models1207/1207-015546model.h5
# #D:\Desktop\Spring 2024\ML\RBM_NEEDS_TO_WORK\results\models1207/1207-093542model.h5
# machine.from_saved_model('D:\Desktop\Spring 2024\ML\RBM_NEEDS_TO_WORK\\results\models1207/1207-015546model.h5')
# image_shape_2 = (28,28)
# #weights = np.asarray(machine.weights)
# #print(weights)
# #print(weights)
# #plot_image_grid(weights, image_shape_2, 9)
# #update input with visible state
# visible_states_1,visible_probabilities_1,inpt = machine.parallel_sample(inpt = x_test_binary_M[0:20],n_step_MC=20)
# #print out the result
# #plot_image_grid(x_test_binary_M[0:9], image_shape_2, 9,name='TestGrid'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.pdf')
# plot_image_grid(visible_probabilities_1[0:20], image_shape_2, 20,name='MNistTemp200C'+'.pdf')

# #plot_image_grid(weights, image_shape_2, 30)

# #D:\Desktop\Spring 2024\ML\RBM_NEEDS_TO_WORK\results\models1107/1107-025813model.h5
# #D:\Desktop\Spring 2024\ML\RBM_NEEDS_TO_WORK\results\models1107/1107-103438model.h5
# #D:\Desktop\Spring 2024\ML\RBM_NEEDS_TO_WORK\results\models1207/1107-181814model.h5
# machine.from_saved_model('D:\Desktop\Spring 2024\ML\RBM_NEEDS_TO_WORK\\results\models1107/1107-025813model.h5')
# image_shape_2 = (28,28)
# #weights = np.asarray(machine.weights)
# #print(weights)
# #print(weights)
# #plot_image_grid(weights, image_shape_2, 9)
# #update input with visible state
# visible_states_1,visible_probabilities_1,inpt = machine.parallel_sample(inpt = x_test_binary_M[0:20],n_step_MC=20)
# #print out the result
# #plot_image_grid(x_test_binary_M[0:9], image_shape_2, 9,name='TestGrid'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.pdf')
# plot_image_grid(visible_probabilities_1[0:20], image_shape_2, 20,name='MNistTemp100C'+'.pdf')

# machine.from_saved_model('D:/Desktop/Spring 2024/ML/RBM_NEEDS_TO_WORK/results/models1207/1107-181814model.h5')
# image_shape_2 = (28,28)
# #weights = np.asarray(machine.weights)
# #print(weights)
# #print(weights)
# #plot_image_grid(weights, image_shape_2, 9)
# #update input with visible state
# visible_states_1,visible_probabilities_1,inpt = machine.parallel_sample(inpt = x_test_binary_M[0:20],n_step_MC=20)
# #print out the result
# #plot_image_grid(x_test_binary_M[0:9], image_shape_2, 9,name='TestGrid'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.pdf')
# plot_image_grid(visible_probabilities_1[0:20], image_shape_2, 20,name='MNistTemp175C'+'.pdf')

# #plot_image_grid(weights, image_shape_2, 30)

# #

# machine.from_saved_model('D:/Desktop/Spring 2024/ML/RBM_NEEDS_TO_WORK/results/models0707/0607-204230model.h5')
# image_shape_2 = (28,28)
# #weights = np.asarray(machine.weights)
# #print(weights)
# #print(weights)
# #plot_image_grid(weights, image_shape_2, 9)
# #update input with visible state
# visible_states_1,visible_probabilities_1,inpt = machine.parallel_sample(inpt = x_test_binary_M[0:20],n_step_MC=20)
# #print out the result
# #plot_image_grid(x_test_binary_M[0:9], image_shape_2, 9,name='TestGrid'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.pdf')
# plot_image_grid(visible_probabilities_1[0:20], image_shape_2, 20,name='MNistTemp300A'+'.pdf')

# #plot_image_grid(weights, image_shape_2, 30)

# #

# machine.from_saved_model('D:/Desktop/Spring 2024/ML/RBM_NEEDS_TO_WORK/results/models0607/0607-123858model.h5')
# image_shape_2 = (28,28)
# #weights = np.asarray(machine.weights)
# #print(weights)
# #print(weights)
# #plot_image_grid(weights, image_shape_2, 9)
# #update input with visible state
# visible_states_1,visible_probabilities_1,inpt = machine.parallel_sample(inpt = x_test_binary_M[0:20],n_step_MC=20)
# #print out the result
# #plot_image_grid(x_test_binary_M[0:9], image_shape_2, 9,name='TestGrid'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.pdf')
# plot_image_grid(visible_probabilities_1[0:20], image_shape_2, 20,name='MNistTemp200A'+'.pdf')

# #plot_image_grid(weights, image_shape_2, 30)

# #

# machine.from_saved_model('D:/Desktop/Spring 2024/ML/RBM_NEEDS_TO_WORK/results/models0607/0607-001032model.h5')
# image_shape_2 = (28,28)
# #weights = np.asarray(machine.weights)
# #print(weights)
# #print(weights)
# #plot_image_grid(weights, image_shape_2, 9)
# #update input with visible state
# visible_states_1,visible_probabilities_1,inpt = machine.parallel_sample(inpt = x_test_binary_M[0:20],n_step_MC=20)
# #print out the result
# #plot_image_grid(x_test_binary_M[0:9], image_shape_2, 9,name='TestGrid'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.pdf')
# plot_image_grid(visible_probabilities_1[0:20], image_shape_2, 20,name='MNistTemp175A'+'.pdf')

# #plot_image_grid(weights, image_shape_2, 30)

# #

# machine.from_saved_model('D:/Desktop/Spring 2024/ML/RBM_NEEDS_TO_WORK/results/models0607/0507-145855model.h5')
# image_shape_2 = (28,28)
# #weights = np.asarray(machine.weights)
# #print(weights)
# #print(weights)
# #plot_image_grid(weights, image_shape_2, 9)
# #update input with visible state
# visible_states_1,visible_probabilities_1,inpt = machine.parallel_sample(inpt = x_test_binary_M[0:20],n_step_MC=20)
# #print out the result
# #plot_image_grid(x_test_binary_M[0:9], image_shape_2, 9,name='TestGrid'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.pdf')
# plot_image_grid(visible_probabilities_1[0:20], image_shape_2, 20,name='MNistTemp150A'+'.pdf')

# #plot_image_grid(weights, image_shape_2, 30)

# #
# machine.from_saved_model('D:\Desktop\Spring 2024\ML\RBM_NEEDS_TO_WORK\\results\models1107/1107-103438model.h5')
# image_shape_2 = (28,28)
# #weights = np.asarray(machine.weights)
# #print(weights)
# #print(weights)
# #plot_image_grid(weights, image_shape_2, 9)
# #update input with visible state
# visible_states_1,visible_probabilities_1,inpt = machine.parallel_sample(inpt = x_test_binary_M[0:20],n_step_MC=20)
# #print out the result
# #plot_image_grid(x_test_binary_M[0:9], image_shape_2, 9,name='TestGrid'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.pdf')
# plot_image_grid(visible_probabilities_1[0:20], image_shape_2, 20,name='MNistTemp150C'+'.pdf')

# # plot_image_grid(weights, image_shape_2, 30)
# print('MNIST Finished!')
# ########################################################################
# #                                 SMILE                                #
# ########################################################################
# s_train_M = smile = pd.read_csv("D:/Desktop/Spring 2024/ML/RBM_NEEDS_TO_WORK/Restricted-Boltzmann-Machines-master/datasets/Top_Secret/sbde.csv")
# s_test_M = smile = pd.read_csv("D:/Desktop/Spring 2024/ML/RBM_NEEDS_TO_WORK/Restricted-Boltzmann-Machines-master/datasets/Top_Secret/sbde.csv")
# s_train_binary_M = np.array([binarizer.fit_transform(slice) for slice in s_train_M])
# s_test_binary_M = np.array([binarizer.fit_transform(slice) for slice in s_test_M])