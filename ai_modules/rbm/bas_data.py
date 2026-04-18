# Copyright 2017 Author @Patric Fulop.
import numpy as np
import itertools as it

def get_data(rng,s=4):
    '''
    :param rng: A random number generator
    :return: A dataset containing 4x4 matrices either with rows as ones, or columns as ones.
    '''
    all_data = np.zeros(shape=s**2)
    size = s
    big_enough = 0
    while big_enough < 499:
        data_i = np.zeros(shape=(size,size))
        if rng.uniform() < 0.5:
            # to see whether we fill horizontally
            # direction = horizontal
            for s in range(0, size):
                if rng.uniform() < 0.5:
                    data_i[s] = np.zeros(shape=size)
                else:
                    data_i[s] = np.ones(shape=size)
            all_data = np.vstack([all_data, data_i.reshape(-1)])
        else:
            # direction = vertical
            for s in range(0, size):
                if rng.uniform() < 0.5:
                    data_i[:, s] = np.zeros(shape=size)
                else:
                    data_i[:, s] = np.ones(shape=size)
            all_data = np.vstack([all_data, data_i.reshape(-1)])
        big_enough += 1
    # uniqueness
    
    y = np.unique(all_data,axis=0)
    
    return y
def get_everything(s=4):
    
    placeholder_a=[]

    for i in range(0, s+1):
        dum = []
        for j in range(1,s+1):
            if i == 0:
                dum = np.append(dum,0)            
            elif j<=i:
                dum = np.append(dum,1)
            else:
                dum = np.append(dum,0)
        placeholder_a.append(dum)
    placeholder_a = np.abs(np.asarray(placeholder_a)-1)
    for i in range(0,s+1):
        if i == 0:
            vf = np.array(list(it.permutations(placeholder_a[i])))
            vf = np.unique(vf,axis=0)
        else:
            b =  np.array(list(it.permutations(placeholder_a[i])))
            b =  np.unique(b,axis=0)
            vf = np.concatenate((vf,b),axis=0)
        vf = np.unique(vf,axis=0)
    return vf,placeholder_a

def get_everything_2(s=4):
    
    placeholder_a=[]
    
    for i in range(0, s+1):
        dum = np.ones(shape=s,dtype=object)
        for j in range(i):
            dum[j]=0
        
        
        placeholder_a.append(dum.tolist())
    
    #print(placeholder_a)
    vf =[]
   

    for i in range(s+1):
        if i == 0:
            vf = get_permute_everything(placeholder_a[i])
        else:
            vf = np.concatenate((vf,get_permute_everything(placeholder_a[i])),axis=0)
        vf = np.unique(vf,axis=0)
    
    return vf

def get_permute_everything(inpt):
    output =[]
    #alg_work = True
    max_val = 0
    sample = inpt
    output.append(inpt.copy())
    
    while  (max_val < 100000):
        index = -1
        index_max = -1
        temp = sample
        
        for k in range(len(sample)-1):

            if sample[k]<sample[k+1]:
                index = k

            if sample[k+1] == max(sample):
                index_max = k+1
            
        if index == -1:
            break
        
        sample[index],sample[index_max] = temp[index_max],temp[index]
        
        t = index+1
        sample[t:len(sample)] = sample[t:len(sample)][::-1]
        
        output.append(sample.copy())

        max_val +=1
    return output
    