import tensorflow as tf 
import numpy as np 
import datetime
import math
import sys
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import deepdish as dd
from sklearn.neighbors import NearestNeighbors
import multiprocessing.dummy as mp
import yaml
from itertools import product
import os
import pandas as pd
# BAS dataset helpers are only needed when self.small_Big is True (legacy experiments).
# Import them lazily inside train() so this module can be used with other datasets
# (e.g. stock data) without requiring the datasets package to be importable.

'''

class monitoring():
    def reconstruction_cross_e():4 
    
    def ava_sq_error(): 
    
    def pseudo_log_l():
'''

class RBM():
    def __init__(self, visible_dim, hidden_dim, number_of_epochs, picture_shape, batch_size, initial_temperature = 1, temps = [1], annealing_decay = 0, training_algorithm='cd', initializer='glorot', k=1, n_test_samples=500, small_Big=False, NAME='RBM', l_1 = -1E-6, non_parallel = False, initial_gamma=2.0, gamma_decay=0.05):
        self.initial_temperature = initial_temperature
        self.annealing_decay = annealing_decay
        self.non_parallel = non_parallel
        self.small_Big = small_Big
        self._n_epoch = number_of_epochs
        self._v_dim = visible_dim
        self._h_dim = hidden_dim
        self._batch_size = batch_size
        self._picture_shape = picture_shape
        self.n_test_samples = n_test_samples
        self.training_algorithm = training_algorithm
        self.epoch = 1
        self.initializer = initializer
        self.k = k
        self.l_1 = l_1
        self.model = self.model()
        self._current_time = datetime.datetime.now().strftime("%d%m-%H%M%S")
        self._log_dir = 'results/logs/' + datetime.datetime.now().strftime("%d%m") + '/' + datetime.datetime.now().strftime("%H%M%S") + '/train'
        self._file_writer = tf.summary.create_file_writer(self._log_dir)
        self._file_writer.set_as_default()
        self.NAME = NAME
        self.boltzmann = 1
        self.temps = temps
        self.initial_gamma = initial_gamma
        self.gamma_decay = gamma_decay
        #0.00008617


    #@tf.function
    def model(self):
        if self.initializer == 'glorot':
            self.weights = tf.Variable(
                tf.random.normal([self._h_dim, self._v_dim], mean=0.0, stddev=0.1, seed=42, dtype=tf.float64)*np.sqrt(2/(self._h_dim + self._v_dim)),
                tf.float64, name="weights")
        elif self.initializer == 'normal':
            self.weights = tf.Variable(
                tf.random.normal([self._h_dim, self._v_dim], mean=0.0, stddev=0.1, seed=42, dtype=tf.float64),
                tf.float64, name="weights")
        #self.visible_biases = tf.Variable(tf.random.uniform([1, self._v_dim], 0, 0.1,seed = 42,dtype=tf.float64), tf.float64, name="visible_biases")
        self.visible_biases = tf.Variable(tf.random.uniform([1, self._v_dim], 0, 0.1,seed = 42,dtype=tf.float64), tf.float64, name="visible_biases")
        #self.hidden_biases = tf.Variable(tf.random.uniform([1, self._h_dim], 0, 0.1, seed = 42,dtype=tf.float64), tf.float64, name="hidden_biases")
        self.hidden_biases = tf.Variable(tf.zeros([1, self._h_dim], dtype=tf.float64), tf.float64, name="hidden_biases")
        self.model_dict = {'weights': self.weights, 'visible_biases': self.visible_biases, 'hidden_biases': self.hidden_biases}
        self.weight_evo  = [self.weights] 
        self.visible_evo = [self.visible_biases]
        self.hidden_evo  = [self.hidden_biases]
        return

    def update_model(self):
        for key,value in self.model_dict.items():
            setattr(self, key, value)

    def save_model(self):
        global h5direc
        """
        Save the current RBM model as .h5 file dictionary with  keys: {'weights', 'visible_biases', 'hidden_biases' }
        weights.numpy() \ biases.numpy() problems with numpy version no longer necessary have to check if it worked.
        """
        model_dict_save = {'weights': self.weights, 'visible_biases': self.visible_biases,
                           'hidden_biases': self.hidden_biases}
        d_m_folder= 'results/models/' + datetime.datetime.now().strftime("%d%m")
        if not os.path.exists(d_m_folder):
            os.makedirs(d_m_folder)
        h5direc = d_m_folder+'/'+self._current_time+'model.h5'
        return dd.io.save(d_m_folder+'/'+self._current_time+'model.h5', model_dict_save)

    def save_param(self, optimizer, data = None):
        to_save = {}
        to_save = {**to_save,**optimizer.__dict__}
        del to_save['machine']
        if data is not None :
            to_save['data'] = data

        variables = self.__dict__
        not_save = ['_file_writer', 'model', 'visible_biases', 'hidden_biases', 'weights', 'model_dict']
        for key,value in variables.items():
            if key not in not_save:
                to_save[key] = value
        with open('results/models/'+self._current_time+'parameters.yml', 'w') as yaml_file:
            yaml.dump(to_save, stream=yaml_file, default_flow_style=False)
        #display parameters in tensorboard
        as_text_matrix = [[k, str(w)] for k, w in sorted(to_save.items())]
        config_summary = tf.summary.text('TrainConfig', tf.convert_to_tensor(as_text_matrix), step = 1)


    def from_saved_model(self,model_path):
        """
        Build a model from the saved parameters.

        :param model_path: string
                           path of .h5 file containing dictionary of the model with  keys: {'weights', 'visible_biases', 'hidden_biases' }

        :return: loaded model
        """

        model_dict = dd.io.load(model_path)
        self.weights = model_dict['weights'] #.astype(np.float64)
        self.visible_biases = model_dict['visible_biases'] #.astype(np.float64)
        self.hidden_biases = model_dict['hidden_biases'] #.astype(np.float64)

        return self

    def calculate_state(self, probability):
        """
        Given the probability(x'=1|W,b) = sigmoid(Wx+b) computes the next state by sampling from the relative binomial distribution.
        x and x' can be the visible and hidden layers respectively or viceversa.

        :param probability: array, shape(visible_dim) or shape(hidden_dim)

        :return: array , shape(visible_dim) or shape(hidden_dim)
                 0,1 state of each unit in the layer
        """

        s = np.random.binomial(1, probability)
        return s.astype(np.float64)
    
    
    #annealing control function!!!!
    def sigmoid_Temp(self,E):
        Beta = (1/(self.initial_temperature*np.exp(-self.annealing_decay*self.epoch)*self.boltzmann))
        return tf.sigmoid(E*Beta)
    # QUANTUM REPLACEMENT
    def sigmoid_Quantum(self, E):
        # Transverse field Gamma anneals from G0 → 0 just like T anneals
        Gamma = self.initial_gamma * np.exp(-self.gamma_decay * self.epoch)
        Beta  = 1 / (self.initial_temperature * np.exp(-self.annealing_decay * self.epoch) * self.boltzmann)
        
        # Quantum-corrected effective field magnitude
        E_eff = tf.sqrt(tf.square(E) + Gamma**2)
        
        # Quantum Boltzmann activation (Benedetti et al. 2016)
        return 0.5 * (1 + (E / E_eff) * tf.tanh(Beta * E_eff))
    #tempering
    def tempering(self):
        return True
    def sample(self, inpt = [] ,n_step_MC=1,p_0=0.5,p_1=0.5):
        """
        Sample from the RBM with n_step_MC steps markov chain.

        :param inpt: array shape(visible_dim), It is possible to start the markov chain from a given point from the dataset or from a random state

        :param n_step_MC: scalar, number of markov chain steps to sample.

        :return: visible_states_1: array shape(visible_dim) visible state after n_step_MC steps

                 visible_probabilities_1: array shape(visible_dim) probabilities from which visible_states_1 is sampled

                 inpt: array shape(picture), return the tarting point of the markov chain

                 evolution_MC, list containing all the states of the markov chain till the sample
        """
        if len(inpt) == 0:
            #inpt = tf.constant(np.random.randint(2, size=self._v_dim), tf.float32)
            inpt = tf.constant(np.random.choice([0,1], size=self._v_dim,p=[p_0,p_1]), tf.float64)
        hidden_probabilities_0 = tf.sigmoid(tf.add(tf.tensordot(self.weights, inpt,1), self.hidden_biases)) # dimension W + 1 row for biases
        hidden_states_0 = self.calculate_state(hidden_probabilities_0)
        evolution_MC = [inpt]
        for _ in range(n_step_MC): #gibbs update
            visible_probabilities_1 = tf.sigmoid(tf.add(tf.tensordot(hidden_states_0,self.weights,1), self.visible_biases)) # dimension W + 1 row for biases
            visible_states_1 = self.calculate_state(visible_probabilities_1)
            hidden_probabilities_1 = tf.sigmoid(tf.add(tf.tensordot(visible_states_1, tf.transpose(self.weights),1), self.hidden_biases)) # dimension W + 1 row for biases
            hidden_states_1 = self.calculate_state(hidden_probabilities_1)
            hidden_states_0 = hidden_states_1
            evolution_MC.append(visible_states_1.reshape(self._v_dim,))
        return visible_states_1,visible_probabilities_1,inpt,evolution_MC

    def parallel_sample(self, inpt = [] ,n_step_MC=1,p_0=0.5,p_1=0.5, n_chains = 1, save_evolution = False):
        if len(inpt) == 0:
            inpt = np.random.choice([0, 1], size=(n_chains,self._v_dim), p=[p_0, p_1]).astype(np.float64)
        else:
            #check shape
            if len(inpt.shape) != 2:
                inpt = inpt.reshape(1,inpt.shape[0])
        if save_evolution:
            evolution = np.empty((n_step_MC,self._v_dim))
            evolution[0] = inpt
        hidden_probabilities_0 = tf.sigmoid(tf.tensordot(inpt, self.weights, axes=[[1], [1]]) + self.hidden_biases)  # dimension W + 1 row for biases
        hidden_states_0 = self.calculate_state(hidden_probabilities_0)
        for i in range(n_step_MC):  # gibbs update
            visible_probabilities_1 = tf.sigmoid(tf.tensordot(hidden_states_0, self.weights, axes=[[1], [0]]) + self.visible_biases)  # dimension W + 1 row for biases
            visible_states_1 = self.calculate_state(visible_probabilities_1)
            hidden_probabilities_1 = tf.sigmoid(tf.tensordot(visible_states_1, self.weights, axes=[[1], [1]]) + self.hidden_biases)  # dimension W + 1 row for biases
            hidden_states_1 = self.calculate_state(hidden_probabilities_1)
            hidden_states_0 = hidden_states_1
            if save_evolution:
                evolution[i] = visible_states_1
        if save_evolution:
            return visible_states_1, visible_probabilities_1,inpt, evolution
        else:
            return visible_states_1, visible_probabilities_1,inpt

    #@tf.function
    def contr_divergence(self, data_point, L2_l = 0):
        """
        Perform contrastive divergence given a data point.

        :param data_point: array, shape(visible_layer)
                           data point sampled from the batch

        :param L2_l: float, lambda for L2 regularization, default = 0 so no regularization performed

        :return: delta_w: array shape(hidden_dim, visible_dim)
                          Array of the same shape of the weight matrix which entries are the gradients dw_{ij}

                 delta_vb: array, shape(visible_dim)
                           Array of the same shape of the visible_biases which entries are the gradients d_vb_i

                 delta_vb: array, shape(hidden_dim)
                           Array of the same shape of the hidden_biases which entries are the gradients d_hb_i

        """
        #print(data_point)
        hidden_probabilities_0 = self.sigmoid_Temp(tf.add(tf.tensordot(data_point, tf.transpose(self.weights),1), self.hidden_biases)) # dimension W + 1 row for biases
        hidden_states_0 = self.calculate_state(hidden_probabilities_0)
        hidden_states_0_copy = hidden_states_0
        for _ in range(self.k): #gibbs update
            visible_probabilities_1 = self.sigmoid_Temp(tf.add(tf.tensordot(hidden_states_0,self.weights, 1), self.visible_biases))# dimension W + 1 row for biases
            visible_states_1 = self.calculate_state(visible_probabilities_1)
            hidden_probabilities_1 = self.sigmoid_Temp(tf.add(tf.tensordot(visible_states_1, tf.transpose(self.weights),1), self.hidden_biases)) # dimension W + 1 row for biases
            hidden_states_1 = self.calculate_state(hidden_probabilities_1)
            hidden_states_0 = hidden_states_1

        vh_0 = tf.reshape(tf.tensordot(hidden_states_0_copy, data_point, 0), (self._h_dim,self._v_dim))
        vh_1 = tf.reshape(tf.tensordot(hidden_states_1, visible_states_1, 0), (self._h_dim,self._v_dim))
        delta_w = tf.add(vh_0, - vh_1) +L2_l*self.weights
        delta_vb = tf.add(data_point, - visible_states_1) + L2_l*self.visible_biases
        delta_hb = tf.add(hidden_states_0_copy, - hidden_states_1) + L2_l*self.hidden_biases
        return delta_w.numpy(), delta_vb.numpy(), delta_hb.numpy() #, visible_states_1


    #@tf.function
    #04/06/24 converted functions to sigmoid_temp
    def parallel_cd(self, batch):
        hidden_probabilities_0 = self.sigmoid_Temp(tf.tensordot(batch, self.weights, axes=[[1], [1]]) + self.hidden_biases)  # dimension W + 1 row for biases
        hidden_states_0 = self.calculate_state(hidden_probabilities_0)
        hidden_states_0_copy = hidden_states_0.copy()
        gibbs_update_counter = 1
        #global visible_config_cheat 
        for _ in range(self.k):  # gibbs update
            visible_probabilities_1 = self.sigmoid_Temp(tf.tensordot(hidden_states_0, self.weights, axes=[[1], [0]]) + self.visible_biases)  # dimension W + 1 row for biases
            visible_states_1 = self.calculate_state(visible_probabilities_1)
            hidden_probabilities_1 = self.sigmoid_Temp(tf.tensordot(visible_states_1, self.weights, axes=[[1], [1]]) + self.hidden_biases)  # dimension W + 1 row for biases
            hidden_states_1 = self.calculate_state(hidden_probabilities_1)
            hidden_states_0 = hidden_states_1
            gibbs_update_counter += 1
            #visible_config_cheat = visible_states_1
            #if (gibbs_update_counter == 100):
            #    print(gibbs_update_counter," steps of AGS\n", self.k," self.k value")
            #    gibbs_update_counter = 0
            #elif (gibbs_update_counter > 100):
            #    print("something has gone wrong")
        
        vh_0 = tf.tensordot(hidden_probabilities_0, batch, axes=[[0], [0]])
        vh_1 = tf.tensordot(hidden_probabilities_1, visible_states_1, axes=[[0], [0]])
        delta_w = (vh_0 - vh_1) / batch.shape[0]
        delta_vb = np.average(batch - visible_states_1, 0)
        delta_hb = np.average(hidden_probabilities_0 - hidden_probabilities_1, 0)

        #return delta_w.numpy(), delta_vb, delta_hb
        return delta_w.numpy()+self.l_1*np.sign(self.weights.numpy()), delta_vb+self.l_1*np.sign(self.visible_biases.numpy()),delta_hb+self.l_1*np.sign(self.hidden_biases.numpy())


    #def energy(self, visible_config):
    #    hidden_probabilities = tf.sigmoid(tf.add(tf.tensordot(self.weights, visible_config,1), self.hidden_biases)) # dimension W + 1 row for biases
    #    hidden_state = self.calculate_state(hidden_probabilities)
    #    E = -np.inner(visible_config, self.visible_biases) -np.inner(hidden_state,self.hidden_biases) -np.inner(hidden_state, tf.tensordot(self.weights,visible_config,1))#
    #
    #    return E[0]
    
    # this is very costly (dont run this on a whim)
    def True_Z(self,v_all,h_all,Beta):
        #
        Z3 = np.transpose(tf.tensordot(h_all.astype(np.float64),np.transpose(tf.tensordot(v_all.astype(np.float64),np.transpose(self.weights),1)),1))
        Z1 = tf.tensordot(v_all.astype(np.float64),np.transpose(self.visible_biases),1)
        Z2 = tf.tensordot(h_all.astype(np.float64),np.transpose(self.hidden_biases),1)
        
        Z23e = tf.exp((Z3+np.transpose(Z2))*Beta)
        
        Z123e = tf.exp(Z1*Beta)*Z23e

        Z = tf.reduce_sum(Z123e)
        
        return Z
    
    def energy(self, visible_config):
        visible_layer_e = visible_config #new adds
        visible_layer_e1 = np.transpose(visible_layer_e) #new adds
        hidden_probabilities_1 = tf.sigmoid(tf.tensordot(visible_layer_e, self.weights, axes=[[1], [1]]) + self.hidden_biases) # dimension W + 1 row for biases
        hidden_state = self.calculate_state(hidden_probabilities_1)
        E = -np.inner(visible_layer_e, self.visible_biases) -np.inner(hidden_state,self.hidden_biases) 
        -np.inner(hidden_state, np.transpose(tf.tensordot(self.weights,visible_layer_e1,1)))

        return E[0][0]
    
    def energyr(self, v_all,h_all):
 
        visible_layer_e = v_all #new adds
        visible_layer_e1 = np.transpose(visible_layer_e) #new adds
        hidden_state = h_all
        
        
        E3 = -np.transpose(tf.tensordot(h_all.astype(np.float64),np.transpose(tf.tensordot(v_all.astype(np.float64),np.transpose(self.weights),1)),1))
        E1 = -tf.tensordot(v_all.astype(np.float64),np.transpose(self.visible_biases),1)
        E2 = -tf.tensordot(h_all.astype(np.float64),np.transpose(self.hidden_biases),1)
        #print(np.shape(E1),np.shape(E2),np.shape(E3))
        E23 = E3+np.transpose(E2)
        #print(np.shape(E23))
        E_matrix = E1+E23
        #print(np.shape(E_matrix))

        #print(E_matrix)
        Et = -np.sum(E_matrix)

        return Et,E_matrix
    
    def expected_energy(self,visible_config):
        
        visible_layer_e = visible_config #new adds
        visible_layer_e1 = np.transpose(visible_layer_e) #new adds
        hidden_probabilities_1 = tf.sigmoid(tf.tensordot(visible_layer_e, self.weights, axes=[[1], [1]]) + self.hidden_biases) # dimension W + 1 row for biases
        hidden_state = self.calculate_state(hidden_probabilities_1)
        
        E = -np.inner(visible_layer_e, self.visible_biases) -np.inner(hidden_state,self.hidden_biases) 
        -np.inner(hidden_state, tf.tensordot(self.weights,visible_layer_e, axes=[[1], [1]]))
        expo_E = tf.exp(-E)
        tot_EM = tf.divide(tf.multiply(E,expo_E),(1+expo_E))
        tot_E =  tf.reduce_sum(tot_EM)
    
        
        return tot_E,tot_EM
    
    def cost_function_Emirate():
        return True 
    
    def expected_energy_2(self,E1,Z,Beta):
        E = np.asarray(E1)
        expo_E = np.exp(-E*Beta)/Z
        E_ind  =  tf.multiply(E,expo_E)
        Et     =  tf.reduce_sum(E_ind)
        
        return Et
            
    def thermo_entropy(self,E1,Z,Beta):
        S = 0
        E = np.asarray(E1)
        p_v_h = np.array(tf.exp(-E*Beta)/Z)
        log_p_v_h = np.log(p_v_h)
        S_ind = -tf.multiply(p_v_h,log_p_v_h)
        S = tf.reduce_sum(S_ind)
        return self.boltzmann*S, self.boltzmann*S_ind
    
    #def thermo_entropy_2(self, data,Z):
    ###    visible_layer_e = data #new adds
    #    visible_layer_e1 = np.transpose(visible_layer_e) #new adds
    #    hidden_probabilities_1 = tf.sigmoid(tf.tensordot(visible_layer_e, self.weights, axes=[[1], [1]]) + self.hidden_biases) # dimension W + 1 row for biases
    #    hidden_state = self.calculate_state(hidden_probabilities_1)
    #    E = -np.inner(visible_layer_e, self.visible_biases) -np.inner(hidden_state,self.hidden_biases) -np.inner(hidden_state, tf.tensordot(self.weights,visible_layer_e, axes=[[1], [1]]))
    #    #                                                                                                          np.inner(j, tf.tensordot(self.weights,i,1)
    #    p_v_h = tf.divide(tf.exp(-E),Z)
    #    log_p_v_h = np.log(p_v_h)
    #    S = -tf.reduce_sum(tf.multiply(p_v_h,log_p_v_h))
    #    
    #    return S
    
    def reconstruction_cross_entropy(self,test_points, plot=False):
        """
        Compute the reconstruction cross entropy = - \Sum_[i=1]^d z_i log(p(z_i)) + (1-z_i) log(1-p(z_i)) where i
        is the i-th component of the reconstructed vector and p(z_i) = sigmoid(Wx+b)_i.

        :param test_point: array like
                           Random point sampled from the test set
        :param plot: bool
                    if True plot the reconstruction togheter with the sampled test point for comparison
        :return: scalar
                Reconstruction cross entropy
        """
        
        r_ce_list=[]
        for vec in test_points: 
            reconstruction,prob,_,_ = self.sample(inpt=vec)
            #tf.where is needed to have 0*-\infty = 0
            r_ce = tf.multiply(reconstruction, tf.where(tf.math.is_inf(tf.math.log(prob)),np.zeros_like(tf.math.log(prob)),tf.math.log(prob))) \
                   + tf.multiply((1-reconstruction), tf.where(tf.math.is_inf(tf.math.log(1-prob)),np.zeros_like(tf.math.log(1-prob)), tf.math.log(1-prob)))
            r_ce_list.append(-tf.reduce_sum(r_ce,1)[0])

        if plot:
            reconstruction_plot= self.sample(inpt=test_points[1,:])[0]
            fig, axes = plt.subplots(nrows=1, ncols=2)
            axes[0].imshow(test_points[1,:].reshape(self._picture_shape),cmap='Greys')
            axes[0].set_title("Original Image")
            axes[1].imshow(np.asarray(reconstruction_plot).reshape(self._picture_shape), cmap='Greys')
            axes[1].set_title("Reconstruction")
            plt.show(block=False)
            plt.pause(3)
            plt.close()

        return np.average(r_ce_list)

    def recon_c_e(self,test_points):
        reconstruction, prob, _ = self.parallel_sample(test_points)
        loss = tf.keras.backend.binary_crossentropy(test_points, prob).numpy()
        loss = np.sum(loss,1)
        return np.average(loss)

    def average_squared_error(self, test_points):
        """
        Compute the mean squared error between a test vector and its reconstruction performed by the RBM, ||x - z||^2.  

        :param test_point: array, shape(visible_dim)
                           data point to test the reconstruction
        :return: sqr: float
                      error
        """
        ase_list=[]
        reconstruction, prob, _ = self.parallel_sample(test_points)
        as_e = tf.pow(test_points - reconstruction, 2)
        sqr = tf.reduce_sum(as_e, 1) / self._v_dim
        return np.mean(sqr)

    def free_energy(self,test_point):
        """
        Compute the free energy of the RBM, it is useful to compute the pseudologlikelihood.
        F(v) = - log \sum_h e^{-E(v,h)} = -bv - \sum_i \log(1 + e^{c_i + W_i v}) where v= visible state, h = hidden state,
        b = visible_biases, c = hidden_biases, W_i = i-th column of the weights matrix
        :param test_point: array, shape(visible_dim)
                           random point sampled from the test set
        :return: scalar
        """
        bv = tf.tensordot(test_point, tf.transpose(self.visible_biases),1)
        wx_b = tf.tensordot(self.weights,test_point,1) + self.hidden_biases
        hidden_term = tf.reduce_sum(tf.math.log(1+tf.math.exp(wx_b)))
        return np.asarray(-hidden_term -bv)[0]

    
    def pseudo_log_likelihood(self, test_point):
        i = np.random.randint(0,self._v_dim,1)
        test_point_flip = test_point.copy()
        test_point_flip[i] = np.logical_not(test_point_flip[i])
        fe_test = self.free_energy(test_point)
        fe_flip = self.free_energy(test_point_flip)
        pseudo = self._v_dim * tf.math.log(tf.sigmoid(fe_flip-fe_test))

        return pseudo
############################################################################################
    def KL_div_BIG_SMALL(self,v_all,h_all,data):
        #Z = self.True_Z(v_all,h_all,1)
        v = np.asarray(v_all)
        d = np.asarray(data)
        E_data = np.zeros(len(v))
        bools = (v==d[:,None]).all(2).any(0)
        #print(v)
        for i in range(0,len(bools)):
            if bools[i]:
                E_data[i]=1
            else:
                E_data[i]=0

        v_all_sample = []

        #print(v_all_sample)
        E_reco = np.asarray(self.energyr(v_all,h_all)[1])
        Z = self.True_Z(v_all,h_all,1)

        E_reco = np.exp(-E_reco)/Z

        E_reco = tf.math.reduce_sum(E_reco,1,keepdims=True).numpy()
        #print(E_reco)
        dkl = 0
        for j in range(0,len(E_data)):
            if E_data[j]==0:
                dkl+=0
            else:
                dkl+=(1/len(d))*np.log((1/len(d))/(E_reco[j]))   
        
        return dkl
##############################################################################################   
    
    def KL_divergence(self, data, n_points, k_neigh, MC_steps=1):
        #todo: I should try with reconstructing point starting from other points
        rnd_test_points_idx = np.random.randint(low=0, high=data['x_test'].shape[0], size=n_points)
        rnd_test_points_idx_2 = np.random.randint(low=0, high=data['x_test'].shape[0], size=n_points)
        test_points = data['x_test'][rnd_test_points_idx, :]
        test_points_2 = data['x_test'][rnd_test_points_idx_2, :]
        #reconstruction = np.empty(test_points_2.shape)
        reconstruction = self.parallel_sample(inpt=test_points_2, n_step_MC=MC_steps)[0]
        nbrs_data = NearestNeighbors(n_neighbors=k_neigh, algorithm='ball_tree', metric='jaccard', n_jobs =-1)
        nbrs_data.fit(test_points)
        nbrs_model = NearestNeighbors(n_neighbors=k_neigh, algorithm='ball_tree', metric='jaccard', n_jobs =-1)
        nbrs_model.fit(reconstruction)

        rho, _ = nbrs_data.kneighbors(test_points)
        nu, _ = nbrs_model.kneighbors(test_points)

        rho_inv, _ = nbrs_data.kneighbors(reconstruction)
        nu_inv, _ = nbrs_model.kneighbors(reconstruction)

        l = 0
        l_inv = 0
        # -2 is needed because in rho the first distance is always 0 and then with the point itself that we should not consider,
        #to effectively pick the k-th neigh w.r.t test points and reconstructions we have to take the k-th in rho and the k-th -1 in nu.
        for i in range(n_points):
            l += np.log(nu[i, k_neigh-2] / rho[i, k_neigh-1])
            l_inv += np.log(rho_inv[i, k_neigh-2] / nu_inv[i, k_neigh-1])
        DKL = self._v_dim / n_points * l + np.log(n_points / (n_points - 1))
        DKL_inv = self._v_dim / n_points * l_inv + np.log(n_points / (n_points - 1))
        return DKL, DKL_inv

    def AIS(machine,n_beta = 10000,n_conf = 20):
        #configurations = np.random.choice([0, 1], size=(100,784), p=[0.5, 0.5])
        n_step=1
        #standard versione without manipulation on expectation of ratio
        beta = np.linspace(0,1,n_beta)
        batch = np.random.choice([0, 1], size=(n_conf,machine._v_dim ), p=[0.5, 0.5]).astype(np.float64)
        #beta = np.concatenate([np.linspace(0,0.5,int(n_beta/4)),np.linspace(0.5,0.9,int(n_beta/4)),np.linspace(0.9,1,n_beta)])
        #rate = np.ones((beta.shape[0]-1, batch.shape[0], machine._h_dim))
        rate = np.ones((beta.shape[0]-1, batch.shape[0], machine._h_dim))

        for k,b in enumerate(beta[:-1]):
            hidden_probabilities_0_A = tf.sigmoid((1-b)*machine.hidden_biases)  # dimension W + 1 row for biases
            hidden_probabilities_0_B = tf.sigmoid(b*(tf.tensordot(batch, machine.weights, axes=[[1], [1]]) + machine.hidden_biases))  # dimension W + 1 row for biases
            hidden_states_0_A = machine.calculate_state(hidden_probabilities_0_A)
            hidden_states_0_B = machine.calculate_state(hidden_probabilities_0_B)

            for _ in range(n_step):  # gibbs update
                visible_probabilities_1_AB = tf.sigmoid((1-b)*machine.visible_biases + b*(tf.tensordot(hidden_states_0_B, machine.weights, axes=[[1], [0]]) + machine.visible_biases)) # dimension W + 1 row for biases
                visible_states_1_AB = machine.calculate_state(visible_probabilities_1_AB)

                hidden_probabilities_1_A = tf.sigmoid((1 - b) * machine.hidden_biases)  # dimension W + 1 row for biases
                hidden_probabilities_1_B = tf.sigmoid(b * (tf.tensordot(visible_states_1_AB, machine.weights, axes=[[1], [1]]) + machine.hidden_biases))  # dimension W + 1 row for biases
                hidden_states_1_A = machine.calculate_state(hidden_probabilities_1_A)
                hidden_states_1_B = machine.calculate_state(hidden_probabilities_1_B)

                hidden_states_0_A = hidden_states_1_A
                hidden_states_0_B = hidden_states_1_B
            batch = visible_states_1_AB
            if k == 0:
                p_k = 1 +tf.exp(machine.hidden_biases)
                p_k_1 =(1 + np.exp((1 - beta[k+1]) * machine.hidden_biases)) * (1 + np.exp(beta[k+1]*(tf.tensordot(visible_states_1_AB, machine.weights, axes=[[1], [1]]) + machine.hidden_biases))) #p_(k)
            elif k == n_beta - 2:
                p_k = (1 + np.exp((1 - b) * machine.hidden_biases)) * (1 + np.exp(b*(tf.tensordot(visible_states_1_AB, machine.weights, axes=[[1], [1]]) + machine.hidden_biases)))
                p_k_1 = 1 + np.exp(beta[k+1]*(tf.tensordot(visible_states_1_AB, machine.weights, axes=[[1], [1]]) + machine.hidden_biases))
            else:
                p_k = (1 + np.exp((1 - b) * machine.hidden_biases)) * (1 + np.exp(b*(tf.tensordot(visible_states_1_AB, machine.weights, axes=[[1], [1]]) + machine.hidden_biases)))
                p_k_1 = (1 + np.exp((1 - beta[k+1]) * machine.hidden_biases)) * (1 + np.exp(beta[k+1]*(tf.tensordot(visible_states_1_AB, machine.weights, axes=[[1], [1]]) + machine.hidden_biases)))#p_(k)
            rate[k,:,:] = (p_k_1/p_k)
        rate = np.product(rate,0)
        rate = np.mean(rate, 0)
        #w = np.product(rate)
        logr_ais = np.sum(np.log(rate))
        #logr_ais = np.log(np.mean(w,0))
        #logr_ais = np.log(w)
        #variance_w = 0 # np.std(w,0)
        logZ_A = np.sum(np.log(1+tf.exp(machine.visible_biases))) + np.sum(np.log(1+tf.exp(machine.hidden_biases))) #if needed add astype(np.float64)
        return logZ_A+logr_ais



    def log_likelihood(self,points):
        """

        :param points: data points to calculate likelihood
        :param test: some random points to start the gibbs sampling to estimate the partition function
        :return: float
        """
        # probably I should calculate the partition function just once
        log_partition_function = self.AIS(20000,20)
        #a lot of dubts wheter I should calculate the partition function on test or train and if i should calculate the likelihood for all the points or not
        log_L = np.inner(self.visible_biases.numpy(),points) + np.sum(np.log((1+np.exp(tf.tensordot(points, self.weights.numpy(), axes=[[1], [1]]) + self.hidden_biases.numpy()))),1)

        return  np.mean(log_L) - log_partition_function
    #fix log likelihood!!!!!!!
    def exact_log_likelihood(self,points, configurations):

        p_v = (np.inner(self.visible_biases.numpy(), configurations) + np.product((1 + np.exp(
            tf.tensordot(configurations.astype(np.float64), self.weights.numpy().astype(np.float64),
                         axes=[[1], [1]]) + self.hidden_biases.numpy().astype(np.float64))), 1))
        exact_logZ = np.log(np.sum(p_v))
        log_L = np.inner(self.visible_biases.numpy(),points) + np.sum(np.log((1+np.exp(tf.tensordot(points.astype(np.float64), self.weights.numpy().astype(np.float64), axes=[[1], [1]]) + self.hidden_biases.numpy().astype(np.float64)))),1)
        return +np.sum(log_L) - exact_logZ, exact_logZ

    def magnetization_reconstruction(self,test_points):
        magn_test = np.mean(test_points,1)
        reconstruction = self.parallel_sample(test_points)[0]
        magn_reconstruction = np.mean(reconstruction, 1)
        error = np.mean(np.abs((magn_test - magn_reconstruction)/magn_test))
        return error

    def variable_summaries(self,var, step):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean, step)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev, step)
            tf.summary.scalar('max', tf.reduce_max(var), step)
            tf.summary.scalar('min', tf.reduce_min(var), step)
            tf.summary.histogram('histogram', var, step = step)

    def train(self, data, optimizer):
        """
        This function shuffle the dataset and create #data_train/batch_size mini batches and perform contrastive divergence on each vector of the batch. 
        The upgrade of the parameters is performed only at the end of each batch by taking the average of the gradients on the batch. 
        In the last part a random datapoint is sampled from the test set to calculate the error reconstruction. The entire procedure is repeted 
         _n_epochs times.

        :param data: dict, dictionary of numpy arrays with labels ['x_train','y_train','x_test', 'y_test']
               optimizer: object optimizer

        :return: self
        """
        print('Start training...')
        self.temp_evo=[self.initial_temperature]
        if self.small_Big:
            # Lazy import: only needed for the BAS small/big experiment mode
            from datasets.bas_data import get_everything_2
            v_all = get_everything_2(self._v_dim)
            print(self._v_dim)
            h_all = get_everything_2(self._h_dim)
            print(self._h_dim)
            print(len(v_all),len(h_all))

        
        #x = [i for i in product(range(2), repeat=self._v_dim)]
        #conf = np.array(x)
        test_fixed = np.random.randint(low=0, high=data['x_test'].shape[0], size=self.n_test_samples)
        for epoch in range(1,self._n_epoch+1):
            
            self.epoch = epoch
            #sys.stdout.write('\r')
            print('Epoch:',epoch, '/', self._n_epoch)
            np.random.shuffle(data['x_train'])
            #with tf.name_scope('Learning rate'):
                #learning_rate = self.exp_decay_l_r(epoch)
            for i in tqdm(range(0, data['x_train'].shape[0], self._batch_size)):
                #print(data['x_train'].shape[0])
                if self.training_algorithm == 'cd':
                    x_train_mini = data['x_train'][i:i + self._batch_size]
                    
                    if self.non_parallel:
                        batch_dw, batch_dvb, batch_dhb= self.contr_divergence(x_train_mini)
                            
                    else: 
                        batch_dw, batch_dvb, batch_dhb= self.parallel_cd(x_train_mini)
                    
                elif self.training_algorithm == 'pcd':
                    if not i%50 or i == 1:
                        x_train_mini = data['x_train'][i:i + self._batch_size]
                    x_train_mini = self.parallel_sample(x_train_mini)[0]
                    batch_dw, batch_dvb, batch_dhb = self.parallel_cd(x_train_mini)

                self.grad_dict = {'weights': batch_dw,
                                  'visible_biases': batch_dvb,
                                  'hidden_biases': batch_dhb}
                optimizer.fit()
            #Save model every epoch
            self.save_model()

            #test every epoch
            np.random.shuffle(data['x_test'])
            rnd_test_points_idx = np.random.randint(low = 0,high = data['x_test'].shape[0], size=self.n_test_samples) #sample size random points indexes from test
            # with tf.name_scope('Performance Metrics'): #TODO: I should computer the reconstruction once and use it inside all these estimatiojs
            #     #DKL, DKL_inv = self.KL_divergence(data,10,7)
            #rec_error = self.reconstruction_cross_entropy(data['x_test'][rnd_test_points_idx,:]) #TODO: add random test datapoint
            #q_error = self.average_squared_error(data['x_test'][rnd_test_points_idx,:])
            #     free_energy = self.free_energy(data['x_test'][rnd_test_points_idx[0],:]) #
            #     pseudo_log = self.pseudo_log_likelihood(data['x_test'][rnd_test_points_idx[0],:])
            #     recon_c_e = self.recon_c_e(data['x_test'][rnd_test_points_idx,:])
            #     #DKL, DKL_inv = self.KL_divergence(data,1000,7)
            #     #log_L_AIS, logZ_AIS = self.log_likelihood(data['x_train'],data['x_train'][rnd_test_points_idx,:])
            #     #log_L, logZ = self.exact_log_likelihood(data['x_train'],conf)
            #     #magnetization_reco_error = self.magnetization_reconstruction(data['x_test'][rnd_test_points_idx,:])
            #     Energy = self.energy(data['x_test'])
            Temp = self.initial_temperature*np.exp(-self.annealing_decay*self.epoch)
            Beta = 1/(Temp*self.boltzmann)
            #     EE = 0
            #     if self.small_Big:
            #         tot_E, mat_E = self.energyr(v_all,h_all)
            #         x = [i for i in product(range(2), repeat=self._v_dim)]
            #         conf = np.array(x)
            #         log_L, logZ = self.exact_log_likelihood(data['x_train'],conf)
            #         #Z_true = tf.exp(logZ)
            #         Z_true = self.True_Z(v_all,h_all,Beta)
            #         #EE,EEM = self.expected_energy(data['x_train'])
            #         EE  = self.expected_energy_2(mat_E,Z_true,Beta)
            #         Entropy, Entropy_M = self.thermo_entropy(mat_E,Z_true,Beta)
            #         F_full = -(1/Beta)*np.log(Z_true)
            #         #Entropy2 = EE+logZ 
            #         Tot = EE-F_full-Temp*Entropy
            #         Entropy_Temp = Temp*Entropy
            #         #if Entropy1 < 0:
            #         #    print("Approximate partition is insufficient, attempting full partition calculation")
            #         #    EE  = self.expected_energy_2(data['x_train'],Z_true)
            #         #    Z_true = self.True_Z(v_all,h_all)
            #         #    Entropy1 = self.thermo_entropy(data['x_train'],Z_true)
            #         #    F_full = EE - Entropy1
            #         #CHANGE DKL ONCE IN A WHILE
            #         DKL = self.KL_div_BIG_SMALL(v_all,h_all,data['x_train'])[0]
            #         tf.summary.scalar('E-F-S', Tot, step = epoch)
            #         tf.summary.scalar('Energy_NOT_EXPECTED', Energy, step = epoch)
            #         tf.summary.scalar('Expected Energy',EE,step=epoch)
            #         tf.summary.scalar('Free Energy', F_full, step = epoch)
            #         tf.summary.scalar('Entropy', Entropy, step = epoch)
            #         tf.summary.scalar('Log Likelihood exact', log_L, step=epoch)
            #         tf.summary.scalar('Partition', Z_true, step=epoch)
            #         tf.summary.scalar('Temperature',Temp,step=epoch)
            #         tf.summary.scalar('Entropy_Temp_TS',Entropy_Temp,step=epoch)
                    
            #     else:
            #         Energy = 0
            #         DKL, DKL_inv = self.KL_divergence(data,1000,7)
            #         tf.summary.scalar('Temperature',Temp,step=epoch)
            #         tf.summary.scalar('Free Energy', free_energy, step = epoch)
            #     tf.summary.scalar('rec_error', rec_error, step = epoch)
            #     tf.summary.scalar('squared_error', q_error, step = epoch)
            #     tf.summary.scalar('Pseudo log likelihood', pseudo_log, step=epoch)
            #     tf.summary.scalar('Binary cross entropy', recon_c_e, step=epoch)
            #     #tf.summary.scalar('inverse KL divergence', DKL_inv, step=epoch)
            #     tf.summary.scalar('KL divergence', DKL, step=epoch)
            #     #tf.summary.scalar('Log Likelihood AIS', log_L_AIS, step=epoch)
            #     #tf.summary.scalar('PArtition AIS', logZ_AIS, step=epoch)
            #     #tf.summary.scalar('Log Likelihood exact', log_L, step=epoch)
            #     #tf.summary.scalar('Parition', logZ, step=epoch)
            #     #tf.summary.scalar('Magn reconstruction error', magnetization_reco_error, step=epoch)
            # with tf.name_scope('Weights'):
            #     self.variable_summaries(self.weights, step = epoch)
            # with tf.name_scope('hidden_biases'):
            #     self.variable_summaries(self.hidden_biases, step = epoch)
            # with tf.name_scope('visible_biases'):
            #     self.variable_summaries(self.visible_biases, step=epoch)
            # with tf.name_scope('Gradients'):
            #     self.variable_summaries(self.grad_dict['weights'], step=epoch)
            #     self.variable_summaries(self.grad_dict['visible_biases'], step=epoch)
            #     self.variable_summaries(self.grad_dict['hidden_biases'], step=epoch)
                
            self.weight_evo = np.concatenate((self.weight_evo,[self.weights]), axis = 0)
            self.visible_evo = np.concatenate((self.visible_evo,[self.visible_biases]), axis = 0) 
            self.hidden_evo = np.concatenate((self.hidden_evo,[self.hidden_biases]), axis = 0)
            self.temp_evo   = np.concatenate((self.temp_evo,[Temp]),axis=None)
            #print(self.weight_evo)
            #print(np.shape(self.weight_evo),np.shape(self.visible_evo),np.shape(self.hidden_evo),np.shape(self.temp_evo))
        

                        
            #reconstruction_plot,prob,inpt,_ = self.sample(inpt=data['x_test'][rnd_test_points_idx[0],:])
            #pic = tf.concat([tf.reshape(inpt,(1,self._v_dim)),prob,reconstruction_plot],0)
            #tf.summary.image('Reconstruction pictures ',tf.reshape(pic,(3,self._picture_shape[0],self._picture_shape[1],1)),max_outputs=100,step = epoch)
            # if self.small_Big:
                
            #     print(self._log_dir)
            #     print("epoch %d" % (epoch + 1),"Rec error: %s" % np.asarray(rec_error),"sq_error %s" % np.asarray(sq_error),"DKL: %.4f" % DKL,"Expected Energy %.5f" % EE,"Free Energy %.5f" % F_full, "TS %.5f" % Entropy_Temp, "Entropy %.5f" % Entropy, "Beta %.5f" % Beta, "E-F+S %.5f" % Tot )
            # else: 
            #     print(self._log_dir)
            #     print("epoch %d" % (epoch + 1),"Rec error: %s" % np.asarray(rec_error),"sq_error %s" % np.asarray(sq_error),"Beta: %.3f" % Beta, "DKL: %.3f" % DKL)
            print(self._log_dir)
            print("epoch %d" % (epoch + 1),"Beta: %.3f" % Beta)
            #print("Z %.5f " % Z_true,"Z_default %.5f" % tf.exp(logZ))
        desgranges_master_file = pd.DataFrame({'Weights':self.weight_evo.tolist(),'Visible Bias':self.visible_evo.tolist(),'Hidden Bias':self.hidden_evo.tolist(),'Temperature':self.temp_evo.tolist()})
        data_file = {self.NAME : [h5direc]}
        ddf = pd.DataFrame(data_file)
                    
        if self.annealing_decay>0:
            string1 = "annealed"
        else:
            string1 = "not_annealed"
        if self._v_dim==4:
            string2 = "_2x2_BAS_"
        else:
            string2 = "_3x3_BAS_"
        string3 = str(self.initial_temperature)
        string_for_file = "results/" + "temp_" + string3 + string2 + string1 + ".csv"
        print(string_for_file)
        
        # wrap the bottom of train() like this:
        try:
            desgranges_master_file.to_csv(string_for_file, mode='w', index=True)
            ddf.to_csv("results/past_machines.csv", mode='a', index=False, header=False)
            print(self.NAME + ' has been appended')
        except Exception as e:
            print(f"Warning: could not save CSV: {e}")
