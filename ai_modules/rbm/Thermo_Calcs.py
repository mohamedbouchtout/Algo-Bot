import numpy as np 
import tensorflow as tf
from datasets.bas_data import get_everything
from datasets.bas_data import get_data
class thermo_props:
    def __init__(self,v_bias,h_bias,weight):
        self.visible_biases = v_bias
        self.hidden_biases = h_bias
        self.weights = weight
        self.boltzmann = 1
    

    def True_Z(self,v_all,h_all,Beta):
            #Z1 = -np.inner(v_all, self.visible_biases)
            #Z2 = -np.inner(h_all,self.hidden_biases)
            #Z3 = -np.inner(tf.tensordot(h_all,self.weights,1),v_all)
            Z = 0
            for i in v_all:
                for j in h_all:
                    Z+= tf.exp((np.inner(i, self.visible_biases) +np.inner(j,self.hidden_biases) +np.inner(j, tf.tensordot(self.weights,i,1)))*Beta)
            return Z[0]
        
    def energy(self, visible_config):
        visible_layer_e = visible_config #new adds
        visible_layer_e1 = np.transpose(visible_layer_e) #new adds
        hidden_probabilities_1 = tf.sigmoid(tf.tensordot(visible_layer_e, self.weights, axes=[[1], [1]]) + self.hidden_biases) # dimension W + 1 row for biases
        hidden_state = self.calculate_state(hidden_probabilities_1)
        E = -np.inner(visible_layer_e, self.visible_biases) -np.inner(hidden_state,self.hidden_biases) -np.inner(hidden_state, np.transpose(tf.tensordot(self.weights,visible_layer_e1,1)))

        return E[0][0]
    
    def energyr(self, v_all,h_all):
        visible_layer_e = v_all #new adds
        visible_layer_e1 = np.transpose(visible_layer_e) #new adds
        hidden_state = h_all
        
        E_matrix = []
        for i in v_all:
            E_rows = []
            for j in h_all:
                E = -np.inner(i, self.visible_biases)-np.inner(j,self.hidden_biases)-np.inner(j, np.transpose(tf.tensordot(self.weights,i,1)))
                #print(E)
                E_rows = np.concatenate((E_rows,E))
            E_matrix.append(E_rows)
        
        #print(E_matrix)
        Et = -np.sum(E_matrix)

        return Et,E_matrix
    
    def expected_energy(self,visible_config):
        
        visible_layer_e = visible_config #new adds
        visible_layer_e1 = np.transpose(visible_layer_e) #new adds
        hidden_probabilities_1 = tf.sigmoid(tf.tensordot(visible_layer_e, self.weights, axes=[[1], [1]]) + self.hidden_biases) # dimension W + 1 row for biases
        hidden_state = self.calculate_state(hidden_probabilities_1)
        
        E = -np.inner(visible_layer_e, self.visible_biases) -np.inner(hidden_state,self.hidden_biases) -np.inner(hidden_state, tf.tensordot(self.weights,visible_layer_e, axes=[[1], [1]]))
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