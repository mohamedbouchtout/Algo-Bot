import pandas as pd 
import numpy as np
from numpy import random as rng
import tensorflow as tf
from datasets.bas_data import get_everything
from datasets.bas_data import get_everything_2
from datasets.bas_data import get_permute_everything
from datasets.bas_data import get_data
import ast
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import plot_image_grid, plot_single_image, plot_input_sample
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()



#Graphing
def plot_weights(df, name, num_visible=4, num_hidden=6):
    epochs = len(df)
    n = name+'Weight.pdf'
    # Convert 'Weights' column to a list of lists
    weight_history = df['Weights'].apply(ast.literal_eval).tolist()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    #depending on data structure, swap indices
    for i in range(num_visible):
        for j in range(num_hidden):
            weights = [w[i][j] for w in weight_history]
            ax.plot(range(epochs), weights, label=f"Weight ({i+1}, {j+1})")       
    
    ax.set_title("Weight Evolution")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weight Value")
    #ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    #plt.tight_layout()
    #plt.show()
    plt.savefig(n, bbox_inches='tight')
def plot_visible_biases(df, name, num_visible=9):
    epochs = len(df)
    n = name+'Visible_Bias.pdf'
    # Convert 'Visible Bias' column to a list of lists
    bias_history = df['Visible Bias'].apply(ast.literal_eval).tolist()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    #depending on data structure, swap indices or add layer
    for i in range(num_visible):
        biases = [b[i] for b in bias_history]
        ax.plot(range(epochs), biases, label=f"Visible Bias {i+1}")
    
    ax.set_title("Visible Bias Evolution")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Bias Value")
    #ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    #plt.tight_layout()
    #plt.show()
    plt.savefig(n, bbox_inches='tight')
def plot_hidden_biases(df, name,num_hidden=14):
    epochs = len(df)
    n = name+'Hidden_Bias.pdf'
    # Convert 'Hidden Bias' column to a list of lists
    bias_history = df['Hidden Bias'].apply(ast.literal_eval).tolist()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    #depending on data structure, swap indices or add layer
    for i in range(num_hidden):
        biases = [b[i] for b in bias_history]
        ax.plot(range(epochs), biases, label=f"Hidden Bias {i+1}")
    
    ax.set_title("Hidden Bias Evolution")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Bias Value")
    #ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    #plt.tight_layout()
    #plt.show()
    plt.savefig(n, bbox_inches='tight')
def plot_marginal_probabilities(vdim,hdim,initial_visible_marginal, initial_hidden_marginal,
                                    final_visible_marginal, final_hidden_marginal,preamble):
        fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(12, 5))
        name ="5k_"+'Marginals_hidden.pdf'
        #print('Plot', initial_visible_marginal)
        # Plot visible layer marginal probabilities
        number=2**vdim
        my_array=[i for i in range(number)]
        ax1.bar(my_array, initial_visible_marginal, label='Initial')
        ax1.bar(my_array, final_visible_marginal, label='Final')
        ax1.set_title('Visible Layer Marginal Probabilities')
        ax1.set_xlabel('v')
        ax1.set_yscale("linear")
        ax1.set_ylabel('p(v)')
        ax1.legend()
        
        #Plot hidden layer marginal probabilities
        number=2**hdim
        my_array=[i for i in range(number)]
        ax2.bar(my_array, initial_hidden_marginal, label='Initial')
        ax2.bar(my_array, final_hidden_marginal, label='Final')
        ax2.set_title('Hidden Layer Marginal Probabilities')
        ax2.set_yscale("linear")
        ax2.set_xlabel('h')
        ax2.set_ylabel('p(h)')
        ax2.legend()
        
        plt.tight_layout()
        #plt.show()
        plt.savefig(name)
def plot_energy_heatmap(energies,v_configs,h_configs,preamble, energy_range=(-100, 100)):
    # Plot a heat map of the RBM energy as a function of visible and hidden unit configurations
    name = "Heat_Maps\\"+preamble+'Energy_Heat_Map.pdf'
#     # Generate all possible binary configurations of visible and hidden units
#     v_configs = [np.array(v_config) for v_config in product([0, 1], repeat = num_visible)]
#     h_configs = [np.array(h_config) for h_config in product([0, 1], repeat = num_hidden)]
    
    # Compute the energues for all configurations
#     energies = np.zeros(shape = (2**(num_visible**2), 2**num_hidden),dtype=np.float64)
#     for i in range(2**(num_visible**2)):
#         for j in range(2**num_hidden):
#             energies[i][j] = rbm_energy(v_config, h_config, W, b, c)
            
    # Mask energies outside the specified range
    masked_energies = np.ma.masked_outside(energies, energy_range[0], energy_range[1])
    
    # Convert visible unit configurations to decimal values
    v_config_decimals = [int(''.join(map(str, v_config)),2) for v_config in v_configs]
    # Convert hidden unit configurations to decimal values
    h_config_decimals = [int(''.join(map(str, h_config)),2) for h_config in h_configs]
    
    # Plot the heat map
    fig, ax = plt.subplots(figsize=(10,10)) # Adjust the width and height as needed
#    cmap = plt.get_cmap('Greys')  # Use the Greys colormap
#    im = ax.imshow(masked_energies, cmap=cmap, vmin=energy_range[0], vmax=energy_range[1])
    im = ax.imshow(masked_energies, cmap='viridis', vmin=energy_range[0], vmax=energy_range[1])
    #ax.set_xticks(np.arange(len(h_configs)))
    #ax.set_yticks(np.arange(len(v_configs)))
#    ax.set_xticklabels([f"{h_config}" for h_config in h_configs], rotation=90) # if not in decimal
    #ax.set_xticklabels(h_config_decimals, rotation=90)
#    ax.set_yticklabels([f"{v_config}" for v_config in v_configs], rotation=90) # if not in decimal
    #ax.set_yticklabels(v_config_decimals)
    # Invert the y-axis
    plt.gca().set_ylim(plt.gca().get_ylim()[::-1])
    
    ax.set_xlabel('Hidden Unit Configurations')
    ax.set_ylabel('Visible Unit Configurations')
    ax.set_title('RBM Energy Heat Map')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,orientation='horizontal') #The fraction parameter sets the width of the colorbar relative to the plot width, and pad sets the padding between the plot and the colorbar.
    #cbar = fig.colorbar(im, ax=ax, orientation='horizontal')
    cbar.set_ticks([-100, -50, 0, 50, 100])
    cbar.set_ticklabels(['-100', '-50', '0', '50', '100'])
    #    fig.colorbar(im, ax=ax)
    #plt.show()
    plt.savefig(name)

#FUNCTIONS
#Entropy
def thermo_entropy(E_matrix,Z,Beta):
    S = 0
    E = np.asarray(E_matrix)
    p_v_h = np.array(tf.exp(-E*Beta)/Z)
    probsv = np.sum(p_v_h,axis=1)
    #probsh = np.transpose(p_v_h)
    #probsh = tf.reduce_sum(probsh,axis=1)
    log_p_v_h = np.log(p_v_h)
    #Sv_ind= -tf.multiply(probsv,np.log(probsv))
    #Sh_ind= -tf.multiply(probsh,np.log(probsh))
    S_ind = -tf.multiply(p_v_h,log_p_v_h)
    S  = tf.reduce_sum(S_ind)
    #Sv = tf.reduce_sum(Sv_ind)
    #Sh = tf.reduce_sum(Sh_ind)
    return S,probsv
#Expected Energy
def expected_energy(E_matrix,Z,Beta):
    E = np.asarray(E_matrix)
    expo_E = np.exp(-E*Beta)/Z
    E_ind  =  tf.multiply(E,expo_E)
    Et     =  tf.reduce_sum(E_ind)
    
    return Et
#KL divergence Calculation
def KL_divergence(v_all,h_all,data,weight,vbias,hbias):
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
    #print(E_data)
    #print(bools)
    #v_all_sample = self.parallel_sample(v_all)[0]
    v_all_sample = []
    # for i in v_all:
    #     if len(v_all_sample)==0:
    #         v_all_sample = self.sample(i)[0]
    #     else:
    #         v_all_sample = np.concatenate((v_all_sample,self.sample(i)[0]),axis=0)
    
    #E_data = np.asarray(self.energyr(v_all,h_all)[1])
    #print(v_all_sample)
    E_reco = np.asarray(energyr(v_all,h_all,weight,vbias,hbias)[1])
    Z = True_Z(v_all,h_all,1,weight,vbias,hbias)
    #/len(v) maybe?
    #print(len(v_all_sample))
    #E_data = np.exp(-E_data)/Z
    E_reco = np.exp(-E_reco)/Z
    
    #print("\n",np.sum(E_reco,axis=1),"\n",E_data)
    #E_data = tf.math.reduce_sum(v_all/len(v_all[0]),1,keepdims=True).numpy()
    E_reco = tf.math.reduce_sum(E_reco,1,keepdims=True).numpy()
    #print(E_reco)
    dkl = 0
    #FIX need to multiply by  probability in distribution (i.e. 1/6 rather than by 1)
    for j in range(0,len(E_data)):
        if E_data[j]==0:
            dkl+=0
        else:
            dkl+=(1/len(d))*np.log((1/len(d))/(E_reco[j])) 
        #print(dkl)    
    
    return dkl
#sample model distribution
def parallel_sample(weights,visible_biases,hidden_biases,Temp, inpt ,n_step_MC=1,p_0=0.5,p_1=0.5, n_chains = 1):
    if len(inpt.shape) != 2:
        inpt = inpt.reshape(1,inpt.shape[0])
    hidden_probabilities_0 = tf.sigmoid((tf.tensordot(inpt, weights, axes=[[1], [1]]) + hidden_biases)*(1/Temp))  # dimension W + 1 row for biases
    hidden_states_0 = np.random.binomial(1, hidden_probabilities_0).astype(np.float64)
    for i in range(n_step_MC):  # gibbs update
        visible_probabilities_1 = tf.sigmoid((tf.tensordot(hidden_states_0, weights, axes=[[1], [0]]) + visible_biases)*(1/Temp))  # dimension W + 1 row for biases
        visible_states_1 = np.random.binomial(1, visible_probabilities_1).astype(np.float64)
        hidden_probabilities_1 = tf.sigmoid((tf.tensordot(visible_states_1, weights, axes=[[1], [1]]) + hidden_biases)*(1/Temp))  # dimension W + 1 row for biases
        hidden_states_1 = np.random.binomial(1, hidden_probabilities_1).astype(np.float64)
        hidden_states_0 = hidden_states_1
        return visible_states_1, visible_probabilities_1,inpt
#Calculate Partition Function
def True_Z(v_all,h_all,Beta,weights,visible_biases,hidden_biases):
        #print(visible_biases)
        #print(hidden_biases)
        #print(weight)
        #print(np.shape(weights),np.shape(visible_biases),np.shape(hidden_biases))
        Z3 = np.transpose(tf.tensordot(h_all.astype(np.float64),np.transpose(tf.tensordot(v_all.astype(np.float64),np.transpose(weights),1)),1))
        Z1 = tf.tensordot(v_all.astype(np.float64),np.transpose(visible_biases),1)
        #Z1 = np.array(Z1)
        Z2 = tf.tensordot(h_all.astype(np.float64),np.transpose(hidden_biases),1) 
        Z23e = tf.exp((Z3+np.transpose(Z2))*Beta)
        #print(np.shape(Z1),np.shape(Z23e))
        Z123e = tf.exp(Z1*Beta)*Z23e

        Z = tf.reduce_sum(Z123e)
        
        return Z
#Calculate Internal Energy of Configuration wrt dataset  
def energy(visible_config,weights,visible_biases,hidden_biases):
    visible_layer_e = visible_config #new adds
    hidden_probabilities_1 = tf.sigmoid(tf.add(tf.tensordot(visible_config,weights, axes=[[1], [1]]), hidden_biases)) # dimension W + 1 row for biases
    hidden_state = np.random.binomial(1, hidden_probabilities_1).astype(np.float64)
    #hidden_state = hidden_probabilities_1
    #print(np.shape(hidden_state),np.shape(visible_config),np.shape(weights),np.shape(visible_biases),np.shape(hidden_biases))
    #E = -np.inner(visible_layer_e, visible_biases) -np.inner(hidden_state,hidden_biases) -np.transpose(np.inner(hidden_state, np.transpose(tf.tensordot(visible_config,np.transpose(weights),1))))
    E3 = -hidden_state.astype(np.float64)*tf.tensordot(visible_config.astype(np.float64),weights,axes=[[1], [1]])
    E1 = -tf.tensordot(visible_config.astype(np.float64),np.transpose(visible_biases),1)
    E2 = -tf.tensordot(hidden_state.astype(np.float64),np.transpose(hidden_biases),1)
    E3 = np.sum(E3,axis=1).reshape(np.shape(E1))
    #print(np.shape(E1),np.shape(E2),np.shape(E3))
    E23 = E3+E2
    #print(np.shape(E23))
    E = (E1+E23).reshape((len(E23)))
    
    #print(np.shape(E))
    return E
#Calculate Internal Energy of Every Configuration of nodes
def energyr(v_all,h_all,weights,visible_biases,hidden_biases):
    
    
    E3 = -np.transpose(tf.tensordot(h_all.astype(np.float64),np.transpose(tf.tensordot(v_all.astype(np.float64),np.transpose(weights),1)),1))
    E1 = -tf.tensordot(v_all.astype(np.float64),np.transpose(visible_biases),1)
    E2 = -tf.tensordot(h_all.astype(np.float64),np.transpose(hidden_biases),1)
    #print(np.shape(E1),np.shape(E2),np.shape(E3))
    E23 = E3+np.transpose(E2)
    #print(np.shape(E23))
    E_matrix = E1+E23

    Et = -np.sum(E_matrix)

    return Et,E_matrix
#Calculate Latent Probability
def prob(states,h_all,weights,visible_biases,hidden_biases,Beta):
    Z = True_Z(states,h_all,Beta,weights,visible_biases,hidden_biases)
    energy_state = np.asarray(energyr(states,h_all,weights,visible_biases,hidden_biases)[1])
    probs = np.exp(-energy_state*Beta)/Z
    #norm = np.sum(prob)
    #probs = prob/norm
    probsv = np.sum(probs,axis=1)
    probsh = np.transpose(probs)
    probsh = np.sum(probsh,axis=1)
    
    return probsv,probsh
def average_squared_error(test_points,weights,visible_biases,hidden_biases,Temp):
    """
    Compute the mean squared error between a test vector and its reconstruction performed by the RBM, ||x - z||^2.  

    :param test_point: array, shape(visible_dim)
                        data point to test the reconstruction
    :return: sqr: float
                    error
    """
    ase_list=[]
    reconstruction, prob, _ = parallel_sample(weights,visible_biases,hidden_biases,Temp,test_points)
    as_e = tf.pow(test_points - reconstruction, 2)
    sqr = tf.reduce_sum(as_e, 1) / len(test_points)
    return np.mean(sqr)
#Calculate RCE
def reconstruction_cross_entropy(test_points,weights,visible_biases,hidden_biases,Temp):
    
    
    _,v_s_bar,v_s = parallel_sample(weights,visible_biases,hidden_biases,Temp,test_points)
    
    r_c_e_1 = v_s*np.log(v_s_bar)
    r_c_e_2 = (1-v_s)*np.log(1-v_s_bar)
    r_c_e = r_c_e_1+r_c_e_2
    R_c_E = -np.sum(r_c_e)/len(test_points)
    
    return R_c_E

#unload model here!
file_list = ["Initial_Temp_1_constant","Initial_Temp_1.3_annealed","Initial_Temp_1.3_constant","Initial_Temp_1.5_annealed","Initial_Temp_1.5_constant","Initial_Temp_1.75_annealed","Initial_Temp_1.75_constant","Initial_Temp_2_annealed","Initial_Temp_2_constant","Initial_Temp_3_annealed","Initial_Temp_3_constant"]
file_list = ["Initial_Temp_1_constant","Initial_Temp_1.3_annealed","Initial_Temp_1.5_annealed","Initial_Temp_1.75_annealed","Initial_Temp_2_annealed","Initial_Temp_3_annealed"]
file_list = ["Initial_Temp_3_annealed"]
vdim = [2] 
hdim = [6]
#other_name = ["KL_min","Mix_min","W_min"]
other_name = ["P1"]
sizel = "3x3"
sizel1 = ["2x2"]
#params = [["_3x3_BAS__KLreg_val__1000000000.0_frob_val__1000000.0","_3x3_BAS__KLreg_val__10.0_frob_val__0.01","_3x3_BAS__KLreg_val__1e-09_frob_val__1.0000000000000002e-12"],["_2x2_BAS__KLreg_val__1000000000.0_frob_val__1000000.0","_2x2_BAS__KLreg_val__1.0_frob_val__0.0001","_2x2_BAS__KLreg_val__1e-09_frob_val__1.0000000000000002e-14"]]
file_list = ["_temp_1_2x2_BAS_not_annealed","_temp_1.3_2x2_BAS_not_annealed","_temp_1.5_2x2_BAS_not_annealed","_temp_1.75_2x2_BAS_not_annealed","_temp_2_2x2_BAS_not_annealed","_temp_3_2x2_BAS_not_annealed","_temp_1.3_2x2_BAS_annealed","_temp_1.5_2x2_BAS_annealed","_temp_1.75_2x2_BAS_annealed","_temp_2_2x2_BAS_annealed","_temp_3_2x2_BAS_annealed"]
#file_list = ["_temp_1_3x3_BAS_not_annealed","_temp_1.3_3x3_BAS_not_annealed","_temp_1.5_3x3_BAS_not_annealed","_temp_1.75_3x3_BAS_not_annealed","_temp_2_3x3_BAS_not_annealed","_temp_3_3x3_BAS_not_annealed","_temp_1.3_3x3_BAS_annealed","_temp_1.5_3x3_BAS_annealed","_temp_1.75_3x3_BAS_annealed","_temp_2_3x3_BAS_annealed","_temp_3_3x3_BAS_annealed"]
#other_name = "Mix_min"
#other_name = "W_min"
#other_name = "KL_min"
#params = "_3x3_BAS__KLreg_val__10.0_frob_val__0.01_"
#params = "_3x3_BAS__KLreg_val__1e-09_frob_val__1.0000000000000002e-12_"
#params = "_3x3_BAS__KLreg_val__1000000000.0_frob_val__1000000.0_"

#source = "D:\Desktop\Master_Files_Thesis\W_metric_Annealing\p05\__2x2_BAS__KLreg_val__0.01_frob_val__1.0000000000000002e-06_"+".csv"
#name = "2x2_BAS_T1_not_annealed_"

for v in range(len(sizel1)):  
    dataset = get_data(rng,s=vdim[v])
    v_all = get_everything_2(vdim[v]**2)
    #print(vdim)
    h_all = get_everything_2(hdim[v])
    #print(hdim)
    counter = 0


    ########################################
    ##########################################
    for index in range(0,len(other_name)):
        for namel in file_list:
            #Declare Path Name for file
            #tring_for_file   = "D:\\Desktop\\Master_Files_Thesis\\W_metric_Annealing\\"+sizel1[v]+"\\"+other_name[index]+"_metrics\\"+namel+"_Thermo_.csv"
            #string_for_file_2 = "D:\\Desktop\\Master_Files_Thesis\\W_metric_Annealing\\"+sizel1[v]+"\\"+other_name[index]+"_metrics\\"+namel+"_Eval_.csv"
            #string_for_file_2 = "D:\\Desktop\\shrinkage\\new_evals_3.csv"
            #string_for_file_3 = "D:\\Desktop\\Master_Files_Thesis\\W_metric_Annealing\\"+sizel1[v]+"\\"+other_name[index]+"_metrics\\"+namel+"_Maps_.csv"
            string_for_file = "D:\Desktop\Master_Files_Thesis\\"+sizel1[v]+"_BAS\\"+other_name[index]+"_metrics\\_"+sizel+"_BAS_"+namel+".csv"
            string_for_file_2 = "D:\Desktop\Master_Files_Thesis\\"+sizel1[v]+"_BAS\\"+other_name[index]+"_metrics\\Evals_"+sizel+"_BAS_"+namel+".csv"
            string_for_file_3 = "D:\Desktop\Master_Files_Thesis\\"+sizel1[v]+"_BAS\\"+other_name[index]+"_metrics\\Map_Probs_"+sizel+"_BAS_"+namel+".csv"
            #string_for_file = "D:\Desktop\Master_Files_Thesis\\"+sizel+"_BAS\\__metric_data_T1check__"+namel+".csv"
            #Load Data file Here
            #source = "D:\\Desktop\\Master_Files_Thesis\\W_metric_Annealing\\"+sizel1[v]+"\\"+params[v][index]+"_"+namel+".csv"
            #source = "D:\\Desktop\\shrinkage\\new_3x3.csv"
            source = "D:\Desktop\Master_Files_Thesis\\2x2_BAS\\"+"_temp_1_2x2_BAS_not_annealed.csv"
            data = pd.read_csv(source)
            
            data1 = data.to_numpy().T
            Ttemp = np.array(data['Temperature'])
            #Wass_Dist = np.array(data['Wasserstein Distanct'])
            Tweights = []
            TvBias = []
            ThBias = []
            #Ttemp = np.array(Ttemp)
            print(source)
            for i in range(len(data1[1])):
                Tweights.append(np.array(ast.literal_eval(data1[1][i])))#.T
                TvBias.append(np.array(ast.literal_eval(data1[2][i]))) #.T
                ThBias.append(np.array(ast.literal_eval(data1[3][i]))) #.T
            Tweights = np.array(Tweights)
            TvBias = np.array(TvBias)
            ThBias =np.array(ThBias)

            # # print(len(data1[1]))
            # print(np.shape(Tweights),np.shape(TvBias),np.shape(ThBias))
            # weight = tf.Variable(np.array(Tweights[9999]).T,tf.float64,name="weights")
            # vbias  = tf.Variable([TvBias[9999]],tf.float64,name="visible_bias")
            # hbias  = tf.Variable([ThBias[9999]],tf.float64,name="hidden_bias")

            # visible_states_1,visible_probabilities_1,inpt = parallel_sample(weight,vbias,hbias,1,dataset,n_step_MC=100)
            
            # if counter == 0:
            #     counter = 1
            #     #plot_image_grid(dataset, (vdim[v],vdim[v]), hdim[v],name='BASInput_'+sizel+'_.pdf')
            # plot_image_grid(visible_probabilities_1, (vdim[v],vdim[v]), hdim[v],name='100_MC_BAS_'+sizel1[v]+'_10k_'+namel+other_name[index]+'.pdf')

            # weight = tf.Variable(np.array(Tweights[3000]).T,tf.float64,name="weights")
            # vbias  = tf.Variable([TvBias[3000]],tf.float64,name="visible_bias")
            # hbias  = tf.Variable([ThBias[3000]],tf.float64,name="hidden_bias")
            
            # visible_states_1,visible_probabilities_1,inpt = parallel_sample(weight,vbias,hbias,1,dataset,n_step_MC=100)
            
            # plot_image_grid(visible_probabilities_1, (vdim[v],vdim[v]), hdim[v],name='100_MC_BAS_'+sizel1[v]+'_3k_'+namel+other_name[index]+'.pdf')

            Z_Tot = np.zeros(shape=(len(data1[1])),dtype=np.float64)
            Free_Energy_Tot = np.zeros(shape=(len(data1[1])),dtype=np.float64)
            Expected_Energy_Tot = np.zeros(shape=(len(data1[1])),dtype=np.float64)
            Entropy_Tot = np.zeros(shape=(len(data1[1])),dtype=np.float64)
            Pv_Tot = np.zeros(shape=(len(data1[1])),dtype=np.float64)
            Sv_Tot = np.zeros(shape=(len(data1[1])),dtype=np.float64)
            Sh_Tot = np.zeros(shape=(len(data1[1])),dtype=np.float64)
            Ssum   = np.zeros(shape=(len(data1[1])),dtype=np.float64)
            KLD_Tot = np.zeros(shape=(len(data1[1])),dtype=np.float64)
            Efs_Tot = np.zeros(shape=(len(data1[1])),dtype=np.float64)
            Steps = np.zeros(shape=(len(data1[1])),dtype=np.integer)
            
            Energy_Map_i = []
            Energy_Map_f = []
            Latent_Probs0 = []
            Latent_Probsf = []
            #T/To Analysis
            RCE_Tot = np.zeros(shape=(len(data1[1])),dtype=np.float64)
            SQE_Tot = np.zeros(shape=(len(data1[1])),dtype=np.float64)
            #
            a = 8300
            b = 8500
            # for i in tqdm(range(a,b)):
            #     weight = tf.Variable(np.transpose(np.array(Tweights[i])),tf.float64,name="weights")
            #     vbias  = tf.Variable([TvBias[i]],tf.float64,name="visible_bias")
            #     hbias  = tf.Variable([ThBias[i]],tf.float64,name="hidden_bias")
            #     #print(np.shape(weight),np.shape(vbias),np.shape(hbias))
            #     Ztemp = np.array(True_Z(v_all,h_all,(1/Ttemp[i]),weight,vbias,hbias))
            #     _,E_mat = energyr(v_all,h_all,weight,vbias,hbias) 
            #     # Free_Energy = -(Ttemp[i])*np.log(Ztemp)
            #     Entropy,Sv = thermo_entropy(E_mat,Ztemp,(1/Ttemp[i]))
            #     #print(np.shape(p_v_h))
            #     Pv_Tot[i] = np.max(Sv)
            #     #Sv_Tot[i] = Sv
            #     #Sh_Tot[i] = Sh
            #     #Smvh1 = Entropy - (Sv+Sh)
            #     #Ssum[i] = Smvh1
            #     #print("Epoch ",i," S:",Entropy," Sv:",Sv," Sh:",Sh," Svh:",Smvh1,"\n")
            #     Steps[i]=i
            #     #Pvh_Tot[i] = p_v_h
            #     #Pvh_TotT[i] = p_v_ht
            #     #KL_div     = KL_divergence(v_all,h_all,dataset,weight,vbias,hbias)
            #     # Expected_Energy = expected_energy(E_mat,Ztemp,(1/Ttemp[i]))
            #     # E_f_s = Expected_Energy-Free_Energy-(Ttemp[i])*Entropy
            #     # Z_Tot[i] = Ztemp
            #     # Free_Energy_Tot[i] = Free_Energy
            #     # Expected_Energy_Tot[i] = Expected_Energy
            #     Entropy_Tot[i] = Entropy
            #     # KLD_Tot[i] = KL_div
            #     # Efs_Tot[i] = E_f_s
                
            # #     #Desgrange Add ons######## T/To ######
            #     # sqe_temp = average_squared_error(dataset,weight,vbias,hbias,Ttemp[i])
            #     # rce_temp = reconstruction_cross_entropy(dataset,weight,vbias,hbias,Ttemp[i])
            #     # RCE_Tot[i] = rce_temp
            #     # SQE_Tot[i] = sqe_temp
            # #Desgrange Add ons
            weight0 = tf.Variable(Tweights[0],tf.float64,name="weights")
            vbias0  = tf.Variable(TvBias[0],tf.float64,name="visible_bias")
            hbias0  = tf.Variable(ThBias[0],tf.float64,name="hidden_bias")
            # weight1k = tf.Variable(np.transpose(np.array(Tweights[3000])),tf.float64,name="weights")
            # vbias1k  = tf.Variable([TvBias[3000]],tf.float64,name="visible_bias")
            # hbias1k = tf.Variable([ThBias[3000]],tf.float64,name="hidden_bias")
            weight = tf.Variable(Tweights[2000],tf.float64,name="weights")
            vbias  = tf.Variable(TvBias[2000],tf.float64,name="visible_bias")
            hbias  = tf.Variable(ThBias[2000],tf.float64,name="hidden_bias")
            _,E_mati = energyr(v_all,h_all,weight0,vbias0,hbias0) 
            _,E_matf = energyr(v_all,h_all,weight,vbias,hbias) 
            #Energy_Map_i.append(E_mati)
            #Energy_Map_f.append(E_matf)
            Energy_Map_i = np.array(E_mati)
            Energy_Map_f = np.array(E_matf)
            prob_vi,prob_hi = prob(v_all,h_all,weight0,vbias0,hbias0,(1/Ttemp[0]))
            #prob_vm,prob_hm = prob(v_all,h_all,weight1k,vbias1k,hbias1k,(1/Ttemp[500]))
            prob_vf,prob_hf = prob(v_all,h_all,weight,vbias,hbias,(1/Ttemp[5000]))
            #print(prob_hi,prob_hf)
            
    ########################################


    ##########################################

            # plt.figure(figsize=(8,6))
        
            # plt.grid(True, which="both", ls="-")
            # plt.minorticks_on()
                #Sv_ind= -tf.multiply(probsv,np.log(probsv))
                #Sh_ind= -tf.multiply(probsh,np.log(probsh))
                #S_ind = -tf.multiply(p_v_h,log_p_v_h)
                #S  = tf.reduce_sum(S_ind)
                #Sv = tf.reduce_sum(Sv_ind)
                #Sh = tf.reduce_sum(Sh_ind)
            #Steps[i]=i
            #Pvh_Tot[i] = p_v_h
            #Pvh_TotT[i] = p_v_ht
                #probsv = np.sum(p_v_h,axis=1)
            #Ep_v_ht = np.transpose(p_v_h)
            #probsh = tf.reduce_sum(probsh,axis=1)
            #pv = np.sum(Pvh_Tot,axis=2)
            
            #ph = np.sum(Pvh_TotT,axis=2)
            #Sv_ind = -tf.multiply(pv,np.log(pv))
            
            #Sh_ind = -tf.multiply(ph,np.log(ph))
            
            #Sv = tf.reduce_sum(Sv_ind,axis=1)
            #Sh = tf.reduce_sum(Sh_ind,axis=1)
            #Smvh1 = Entropy_Tot-Sv-Sh

            #plt.plot(Steps[a:b:1], Pv_Tot[a:b:1], label='Max p(v)',color = 'blue')
            #plt.plot(Steps[a:b:1], Sv_Tot[a:b:1], label='Visible Entropy (Sv)',color = 'red')
            #plt.plot(Steps[a:b:1], Sh_Tot[a:b:1], label='Hidden Entropy (Sh)',color = 'green')
            #plt.plot(Steps[a:b:1], Ssum[a:b:1], label='S - Sv - Sh',color = 'orange')
            
            # plt.yscale('linear')
            # plt.xlabel('Epoch')
            # plt.ylabel('Probability')
            # plt.legend(loc='center left', bbox_to_anchor=(0.1, 0.8))
        
            # pltTitle1 = "Maximum Probability"
            # plt.title(pltTitle1)
            # plt.show()
            # Thermo_Properties = pd.DataFrame({"Z":Z_Tot,
            #                                 "Entropy":Entropy_Tot,
            #                                 "Free Energy":Free_Energy_Tot,
            #                                 "Expected Energy":Expected_Energy_Tot,
            #                                 "KL Divergence":KLD_Tot,
                                            
            #                                 "Temperature": Ttemp,
            #                                 "EFS": E_f_s})
            # Other_Evals =  pd.DataFrame({"RCE":RCE_Tot,
            #                                 "SQE":SQE_Tot,
            #                                 "Temperature": Ttemp
            #                                 })
            # EntropyP =  pd.DataFrame({"S":Entropy_Tot,
            #                                 "Pmax":Pv_Tot,     
            #                                 })
            # string_for_file = "83ENTROPY_"+namel+"10k"+other_name[index]+"_"+sizel1[v]+".csv"
            # EntropyP.to_csv(string_for_file,index=True,index_label="Index")
            # Thermo_Properties.to_csv(string_for_file,mode='x',index=True,index_label="Index")
            # Other_Evals.to_csv(string_for_file_2,mode='x',index=True,index_label="Index")

            #Map_Probs.to_csv(string_for_file_3,mode='x',index=True,index_label="Index")
            #print(np.mean(Efs_Tot))

            #plot_weights(data,sizel1[v]+"_"+namel+other_name[index],num_visible = vdim[v]**2,num_hidden=hdim[v])
            #plot_visible_biases(data,sizel1[v]+"_"+namel+other_name[index],num_visible = vdim[v]**2)
            #plot_hidden_biases(data,sizel1[v]+"_"+namel+other_name[index],num_hidden=hdim[v])
            #n = namel+sizel
            plot_marginal_probabilities(vdim[v]**2,hdim[v],prob_vi,prob_hi,prob_vf,prob_hf,namel+"5k"+other_name[index]+"_"+sizel1[v]+"_")
            break
            
            #plot_marginal_probabilities(vdim[v]**2,hdim[v],prob_vi,prob_hi,prob_vm,prob_hm,namel+"3k_"+other_name[index]+"_"+sizel1[v]+"_")
            # #plot_energy_heatmap(Energy_Map_i,v_all,h_all,namel+"_Initial_")
            plot_energy_heatmap(Energy_Map_f,v_all,h_all,namel+"_Final_"+sizel1[v]+"_"+other_name[index])
            #plot_energy_heatmap(Energy_Map_m,v_all,h_all,namel+"_Final_3k")
print('DONE!')