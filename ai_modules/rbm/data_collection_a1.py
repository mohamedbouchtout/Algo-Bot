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
from utils import plot_image_grid, plot_single_image, plot_input_sample

##############################################################################################
#                                            BAS TRAINING                                    #
##############################################################################################
x_test_bas = get_data(rng,s=2)
x_train_bas = get_data(rng,s=2)

np.random.shuffle(x_train_bas)
np.random.shuffle(x_test_bas)



data_bas = {"x_train": x_train_bas[:len(x_train_bas)],"x_test": x_test_bas[:len(x_test_bas)]}

x_test_bas3 = get_data(rng,s=3)
x_train_bas3 = get_data(rng,s=3)

np.random.shuffle(x_train_bas3)
np.random.shuffle(x_test_bas3)

data_bas3 = {"x_train": x_train_bas3[:len(x_train_bas3)],"x_test": x_test_bas3[:len(x_test_bas3)]}
#Create a restricted boltzmann machines

#TEST1
machine = RBM(np.size(x_test_bas[0]), len(x_test_bas),20000,(2,2), 1, k=100, n_test_samples=6 ,small_Big=True, NAME='BAS_2x2_110824_T1',l_1=0,non_parallel=True,initial_temperature=1,annealing_decay=0)
optimus = Optimizer(machine, 0.15)
#Train the machine
machine.train(data_bas,optimus)

# #TEST2
machine = RBM(np.size(x_test_bas[0]), len(x_test_bas),20000,(2,2), 1, k=100, n_test_samples=6 ,small_Big=True, NAME='BAS_2x2_010824_T2',l_1=0,non_parallel=True,initial_temperature=1.3,annealing_decay=0)
optimus = Optimizer(machine, 0.15)
#Train the machine
machine.train(data_bas,optimus)

# #TEST3
machine = RBM(np.size(x_test_bas[0]), len(x_test_bas),20000,(2,2), 1, k=100, n_test_samples=6 ,small_Big=True, NAME='BAS_2x2_010824_T3',l_1=0,non_parallel=True,initial_temperature=1.5,annealing_decay=0)
optimus = Optimizer(machine, 0.15)
#Train the machine
machine.train(data_bas,optimus)

# #TEST4
machine = RBM(np.size(x_test_bas[0]), len(x_test_bas),20000,(2,2), 1, k=100, n_test_samples=6 ,small_Big=True, NAME='BAS_2x2_010824_T4',l_1=0,non_parallel=True,initial_temperature=1.75,annealing_decay=0)
optimus = Optimizer(machine, 0.15)
#Train the machine
machine.train(data_bas,optimus)

#TEST5
machine = RBM(np.size(x_test_bas[0]), len(x_test_bas),20000,(2,2), 1, k=100, n_test_samples=6 ,small_Big=True, NAME='BAS_2x2_110824_T5',l_1=0,non_parallel=True,initial_temperature=2,annealing_decay=0)
optimus = Optimizer(machine, 0.15)
#Train the machine
machine.train(data_bas,optimus)

# #TEST6
machine = RBM(np.size(x_test_bas[0]), len(x_test_bas),20000,(2,2), 1, k=100, n_test_samples=6 ,small_Big=True, NAME='BAS_2x2_110824_T5',l_1=0,non_parallel=True,initial_temperature=3,annealing_decay=0)
optimus = Optimizer(machine, 0.15)
#Train the machine
machine.train(data_bas,optimus)

# # #TEST2
machine = RBM(np.size(x_test_bas[0]), len(x_test_bas),20000,(2,2), 1, k=100, n_test_samples=6 ,small_Big=True, NAME='BAS_2x2_010824_T2',l_1=0,non_parallel=True,initial_temperature=1.3,annealing_decay=0.00001311)
optimus = Optimizer(machine, 0.15)
#Train the machine
machine.train(data_bas,optimus)

# # #TEST3
machine = RBM(np.size(x_test_bas[0]), len(x_test_bas),20000,(2,2), 1, k=100, n_test_samples=6 ,small_Big=True, NAME='BAS_2x2_010824_T3',l_1=0,non_parallel=True,initial_temperature=1.5,annealing_decay=0.00002027)
optimus = Optimizer(machine, 0.15)
#Train the machine
machine.train(data_bas,optimus)

#TEST4
machine = RBM(np.size(x_test_bas[0]), len(x_test_bas),20000,(2,2), 1, k=100, n_test_samples=6 ,small_Big=True, NAME='BAS_2x2_010824_T4',l_1=0,non_parallel=True,initial_temperature=1.75,annealing_decay=0.00002798)
optimus = Optimizer(machine, 0.15)
#Train the machine
machine.train(data_bas,optimus)

# # #TEST5
machine = RBM(np.size(x_test_bas[0]), len(x_test_bas),20000,(2,2), 1, k=100, n_test_samples=6 ,small_Big=True, NAME='BAS_2x2_110824_T5',l_1=0,non_parallel=True,initial_temperature=2,annealing_decay=0.00003465)
optimus = Optimizer(machine, 0.15)
#Train the machine
machine.train(data_bas,optimus)
#TEST6
machine = RBM(np.size(x_test_bas[0]), len(x_test_bas),20000,(2,2), 1, k=100, n_test_samples=6 ,small_Big=True, NAME='BAS_2x2_110824_T6',l_1=0,non_parallel=True,initial_temperature=3,annealing_decay=0.00005493)
optimus = Optimizer(machine, 0.15)
#Train the machine
machine.train(data_bas,optimus)
# #3x3
# #TEST1
# machine = RBM(np.size(x_test_bas3[0]), len(x_test_bas3),5000,(3,3), 2, k=100 ,small_Big=True, NAME='BAS_3x3_110824_T1',l_1=0,non_parallel=False,initial_temperature=1,annealing_decay=0)
# optimus = Optimizer(machine, 0.15)
# #Train the machine
# machine.train(data_bas3,optimus)

# #TEST2
# machine = RBM(np.size(x_test_bas3[0]), len(x_test_bas3),5000,(3,3), 2, k=100 ,small_Big=True, NAME='BAS_3x3_010824_T2',l_1=0,non_parallel=False,initial_temperature=1.3,annealing_decay=0)
# optimus = Optimizer(machine, 0.15)
# #Train the machine
# machine.train(data_bas3,optimus)

# #TEST3
# machine = RBM(np.size(x_test_bas3[0]), len(x_test_bas3),5000,(3,3), 2, k=100 ,small_Big=True, NAME='BAS_3x3_010824_T3',l_1=0,non_parallel=False,initial_temperature=1.5,annealing_decay=0)
# optimus = Optimizer(machine, 0.15)
# #Train the machine
# machine.train(data_bas3,optimus)

# #TEST4
# machine = RBM(np.size(x_test_bas3[0]), len(x_test_bas3),5000,(3,3), 2, k=100,small_Big=True, NAME='BAS_3x3_010824_T4',l_1=0,non_parallel=False,initial_temperature=1.75,annealing_decay=0)
# optimus = Optimizer(machine, 0.15)
# #Train the machine
# machine.train(data_bas3,optimus)

# #TEST5
# machine = RBM(np.size(x_test_bas3[0]), len(x_test_bas3),2000,(3,3), 2, k=100 ,small_Big=True, NAME='BAS_3x3_110824_T5',l_1=0,non_parallel=False,initial_temperature=2,annealing_decay=0)
# optimus = Optimizer(machine, 0.15)
# #Train the machine
# machine.train(data_bas3,optimus)

# #TEST6
# machine = RBM(np.size(x_test_bas3[0]), len(x_test_bas3),2000,(3,3), 2, k=100 ,small_Big=True, NAME='BAS_3x3_110824_T6',l_1=0,non_parallel=False,initial_temperature=3,annealing_decay=0)
# optimus = Optimizer(machine, 0.15)
# #Train the machine
# machine.train(data_bas3,optimus)

# #TEST2
# machine = RBM(np.size(x_test_bas3[0]), len(x_test_bas3),5000,(3,3), 2, k=100 ,small_Big=True, NAME='BAS_3x3_010824_T2',l_1=0,non_parallel=False,initial_temperature=1.3,annealing_decay=0.00002622)
# optimus = Optimizer(machine, 0.15)
# #Train the machine
# machine.train(data_bas3,optimus)

# #TEST3
# machine = RBM(np.size(x_test_bas3[0]), len(x_test_bas3),5000,(3,3), 2, k=100 ,small_Big=True, NAME='BAS_3x3_010824_T3',l_1=0,non_parallel=False,initial_temperature=1.5,annealing_decay=0.00004054)
# optimus = Optimizer(machine, 0.15)
# #Train the machine
# machine.train(data_bas3,optimus)

# #TEST4
# machine = RBM(np.size(x_test_bas3[0]), len(x_test_bas3),5000,(3,3), 2, k=100 ,small_Big=True, NAME='BAS_3x3_010824_T4',l_1=0,non_parallel=False,initial_temperature=1.75,annealing_decay=0.00005596)
# optimus = Optimizer(machine, 0.15)
# #Train the machine
# machine.train(data_bas3,optimus)

# #TEST5
# machine = RBM(np.size(x_test_bas3[0]), len(x_test_bas3),2000,(3,3), 2, k=100 ,small_Big=True, NAME='BAS_3x3_010824_T5',l_1=0,non_parallel=False,initial_temperature=2,annealing_decay=0.0001386)
# optimus = Optimizer(machine, 0.15)
# #Train the machine
# machine.train(data_bas3,optimus)
# #TEST6
# machine = RBM(np.size(x_test_bas3[0]), len(x_test_bas3),2000,(3,3), 2, k=100 ,small_Big=True, NAME='BAS_3x3_010824_T6',l_1=0,non_parallel=False,initial_temperature=3,annealing_decay=0.0002197)
# optimus = Optimizer(machine, 0.15)
# #Train the machine
# machine.train(data_bas3,optimus)

print('BAS Training Complete')
###################################################################################################
#                                            MNIST TRAINING                                       #
###################################################################################################
# # #Import mnist dataset
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

# #shuffle data
# np.random.shuffle(x_train_binary_M)
# np.random.shuffle(x_test_binary_M)

# #create dictionary of data
# data_M = {"x_train": x_train_binary_M[:2560],"y_train": y_train_M[:2560],"x_test": x_test_binary_M[:1600],"y_test": y_test_M[:1600]}

# #Create a restricted boltzmann machines
# machine = RBM(x_train_binary_M[0].shape[0], 100,5000,(28,28), 128, k = 20,NAME='MNIST 110824',l_1=0,initial_temperature=1,annealing_decay=0)
# #originally decay rate of 10^-6
# optimus = Optimizer(machine, 0.2,k=0.00000001)
# #Train the machine
# machine.train(data_M,optimus)

# #Create a restricted boltzmann machines
# machine = RBM(x_train_binary_M[0].shape[0], 100,5000,(28,28), 128, k = 20,NAME='MNIST 110824',l_1=0,initial_temperature=1.5,annealing_decay=0)
# #originally decay rate of 10^-6
# optimus = Optimizer(machine, 0.2,k=0.00000001)
# #Train the machine
# machine.train(data_M,optimus)

# #Create a restricted boltzmann machines
# machine = RBM(x_train_binary_M[0].shape[0], 100,5000,(28,28), 128, k = 20,NAME='MNIST 110824',l_1=0,initial_temperature=1.75,annealing_decay=0)
# #originally decay rate of 10^-6
# optimus = Optimizer(machine, 0.2,k=0.00000001)
# #Train the machine
# machine.train(data_M,optimus)

# #Create a restricted boltzmann machines
# machine = RBM(x_train_binary_M[0].shape[0], 100,5000,(28,28), 128, k = 20,NAME='MNIST 110824',l_1=0,initial_temperature=2,annealing_decay=0)
# #originally decay rate of 10^-6
# optimus = Optimizer(machine, 0.2,k=0.00000001)
# #Train the machine
# machine.train(data_M,optimus)

# #Create a restricted boltzmann machines
# machine = RBM(x_train_binary_M[0].shape[0], 100,5000,(28,28), 128, k = 20,NAME='MNIST 110824',l_1=0,initial_temperature=3,annealing_decay=0)
# #originally decay rate of 10^-6
# optimus = Optimizer(machine, 0.2,k=0.00000001)
# #Train the machine
# machine.train(data_M,optimus)

# #Create a restricted boltzmann machines
# machine = RBM(x_train_binary_M[0].shape[0], 100,5000,(28,28), 128, k = 20,NAME='MNIST 110824',l_1=0,initial_temperature=1.5,annealing_decay=0.00008109)
# #originally decay rate of 10^-6
# optimus = Optimizer(machine, 0.2,k=0.00000001)
# #Train the machine
# machine.train(data_M,optimus)

# #Create a restricted boltzmann machines
# machine = RBM(x_train_binary_M[0].shape[0], 100,5000,(28,28), 128, k = 20,NAME='MNIST 110824',l_1=0,initial_temperature=1.75,annealing_decay=0.0001119)
# #originally decay rate of 10^-6
# optimus = Optimizer(machine, 0.2,k=0.00000001)
# #Train the machine
# machine.train(data_M,optimus)

# #Create a restricted boltzmann machines
# machine = RBM(x_train_binary_M[0].shape[0], 100,5000,(28,28), 128, k = 20,NAME='MNIST 010824',l_1=0,initial_temperature=2,annealing_decay=0.0001386)
# #originally decay rate of 10^-6
# optimus = Optimizer(machine, 0.2,k=0.00000001)
# #Train the machine
# machine.train(data_M,optimus)

# #Create a restricted boltzmann machines
# machine = RBM(x_train_binary_M[0].shape[0], 100,5000,(28,28), 128, k = 20,NAME='MNIST 010824',l_1=0,initial_temperature=3,annealing_decay=0.0002197)
# #originally decay rate of 10^-6
# optimus = Optimizer(machine, 0.2,k=0.00000001)
# #Train the machine
# machine.train(data_M,optimus)


# print('MNIST Training Complete')
# #####################################################################################################
# #                                           SMILE                                                   #
# #####################################################################################################
# #read data
# s_train_M  = pd.read_csv("D:/Desktop/Spring 2024/ML/RBM_NEEDS_TO_WORK/Restricted-Boltzmann-Machines-master/datasets/Top_Secret/sbde.csv")
# s_test_M   = pd.read_csv("D:/Desktop/Spring 2024/ML/RBM_NEEDS_TO_WORK/Restricted-Boltzmann-Machines-master/datasets/Top_Secret/sbde.csv")

# s_train_M = s_train_M.to_numpy(dtype = 'float64')
# s_test_M  = s_test_M.to_numpy(dtype ='float64')
# #binarize
# binarizer = Binarizer(threshold=0.5)
# s_train_binary_M = binarizer.fit_transform(s_train_M)
# s_test_binary_M = binarizer.fit_transform(s_test_M)

# #s_train_binary_M = x_train_binary_M.reshape(s_train_binary_M.shape[0],-1).astype(np.float64)
# #s_test_binary_M = x_test_binary_M.reshape(s_test_binary_M.shape[0],-1).astype(np.float64)
# #shuffle
# np.random.shuffle(s_train_binary_M)
# np.random.shuffle(s_test_binary_M)
# #create dictionary
# data_S = {"x_train": s_train_binary_M[:3000],"x_test": s_test_binary_M[:3000]}
# #Create a restricted boltzmann machines
# machine = RBM(s_train_binary_M[0].shape[0], 60 ,5000,(1,60), 128,k=25,NAME='SMILE 170624',l_1=0)
# optimus = Optimizer(machine, 0.25,k=0.0000001)
# #Train the machine
# #machine.train(data_S,optimus)
# print('SMILE Training Complete')