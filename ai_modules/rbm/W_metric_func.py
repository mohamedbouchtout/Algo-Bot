import numpy as np 
import tensorflow as tf
    
class W_metric:
    def __init__(self,epsilon=.0001,max_iter=1000000,gamma = 0.1,eta = 0.0001,lamda = 0.1,eps=1e-30):
        self.epsilon = epsilon
        self.max = max_iter
        self.gamma = gamma
        self.lamda = lamda
        self.eta  = eta
        self.eps = eps
                
    def hamming_distance(self,i,j):
        #i,j two binary vectors
        H = np.add(i,-j)
        H = np.abs(H)
        Hs= np.sum(H)
        #returns XOR op or hamming distance
        return H,Hs

    def hamming_norm(self,v_data):
        H = 0 
        for i in v_data:
            for j in v_data:
                H+=self.hamming_distance(i,j)[1]
        H = H/(len(v_data)**2)
        
        #Below is a small optimization, let me know if this is correct. Calculates the diagonal part of the matrix as its symmetric which halves the needed iterations
        
        #for i in range(len(v_data)):
        #    for j in range(i,len(v_data)):
        #        H+=self.hamming_distance(v_data[i],v_data[j])[1]
        #H = 2*H/(len(v_data)**2)
        
        
        return H
    def K_matrix(self,v_data,v_model,norm):
        
        D = []
        DD = []
        for i in v_model:
            rD= []
            dD= []
            for j in v_data:
                rD.append(-1+self.hamming_distance(i,j)[1]/(norm*self.gamma))
                dD.append((self.hamming_distance(i,j)[1]/norm))
            D.append(rD)
            DD.append(dD)
            
        
        H = np.exp(D)
        
        return H,DD   
    def p_q_vector(self,data,reco,all):
        v = np.asarray(all)
        d = np.asarray(data)
        b = np.asarray(reco)
        p_test = np.ones(len(d))
        E_data = np.zeros(len(v))
        bools = (v==d[:,None]).all(2).any(0)
        bools1=( v==b[:,None]).all(2).any(0)
        #print(v)
        
        for i in range(0,len(bools)):
            if bools[i]:
                E_data[i]=1
            else:
                E_data[i]=0   
        p = E_data/len(d)
        p_test = p_test/len(d)
        #for testing purposes q will be a shuffled p
        return p_test, bools,bools1
    def check_nan(self,x, name):    
        if np.isnan(x).any():         
            print(f"NaN detected in {name}")       
            # Handle the NaN (e.g., by replacing with a small value)
            np.nan_to_num(x, nan=self.eps)     
            return x  
        return x
    def sinkhorn_algorithm(self,K, p, q):
            # Initialize u and v with ones
        #print(p)
        #print(q)   
        #print(np.shape(q)) 
        u = np.ones(len(p))/len(p)
        v = np.ones(len(q))/len(q)
        
          
        i = 0
        max_dif_u = 0
        max_dif_v = 0
        min_dif_u = 0
        min_dif_v = 0
        counteru = 0
        counterv = 0
        while (True and i<self.max):
            # Compute u and v updates
            #print(K)
            v_new = (q/((K.T)@u+self.eps))/len(q)
            v_new = self.check_nan(v_new,"v_new")
            u_new = (p/(K@v+self.eps))/len(p)
            u_new = self.check_nan(u_new,"u_new")
            
            #and np.linalg.norm(v - v_new) < self.epsilon:
            if i == 0:
                min_dif_u = np.linalg.norm(u - u_new)
                min_dif_v = np.linalg.norm(v - v_new)
            # Check for convergence
            if np.linalg.norm(u - u_new) < self.epsilon and np.linalg.norm(v - v_new) < self.epsilon:
                break
            
            if np.linalg.norm(u - u_new) > max_dif_u:
                max_dif_u = np.linalg.norm(u - u_new)
                
                
            if np.linalg.norm(v - v_new) > max_dif_v:
                max_dif_v = np.linalg.norm(v - v_new)
                
            if np.linalg.norm(u - u_new) < min_dif_u:
                max_min_u = np.linalg.norm(u - u_new)
                counteru = i
            if np.linalg.norm(v - v_new) < min_dif_v:
                min_dif_v = np.linalg.norm(v - v_new)
                counterv = i
            #print("NEW ITERATION")
            # Update u and v
            i+=1
            if (i==self.max):
                print("ERROR: SINKHORN Condition not met")
                print("diffs u:",np.linalg.norm(u - u_new),"\ndiffs v:",np.linalg.norm(v - v_new),"\nMax Diff U:",max_dif_u,"\nMax Diff V:",max_dif_v,"\nMin Diff U:",min_dif_u,counteru,"\nMin Diff V:",min_dif_v,counterv)
            u = u_new
            v = v_new
        # Compute the dual potential α*
        #print("iterations:",i,"\nu:",u,"\nv",v)
        alpha_star1 = np.log(u)*self.gamma
        beta_star1  = np.log(v)*self.gamma
        beta_star   = beta_star1 + (np.mean(alpha_star1))
        alpha_star = alpha_star1 - np.mean(alpha_star1)
        #print(alpha_star)
        return alpha_star,beta_star,u,v,alpha_star1,beta_star1
    def alpha_star_i(self,alpha,beta,a_bool,b_bool):
        a_s_i = []
        b_s_i = []
        ai = []
        bi = []
        for i in range(0,len(a_bool)):
            if a_bool[i]:
                a_s_i.append(1)
            else:
                a_s_i.append(0)   
        for i in range(0,len(b_bool)):
            if b_bool[i]:
                b_s_i.append(1)
            else:
                b_s_i.append(0)   
        for i in range(len(a_bool)):
            if a_bool[i]==1:
                ai.append(alpha[i])
            if b_bool[i]==1:
                bi.append(beta[i])

        #print(a_s_i)
        return ai,bi
    def Wasserstein_Distance(self,p,q,a,b):
        a1 = a*p
        b1 = b*q
        W = np.sum(a1)+np.sum(b1)
        return W
    def Wasserstein_Distance_2(self,u,v,D,H):
        W = np.dot(u,(D*H)@v)
        return W
    def Wasserstein_Distance_3(self,alpha,beta,q,p,D):
        D1 = np.asarray(D)
        
        ap = np.mean(alpha*p)
        bq = np.mean(beta*q)
        ge3 = np.exp(beta/self.gamma-1-D1/self.gamma)
        ge2 = np.sum(ge3,axis=1)
        ge1 = np.exp(alpha/self.gamma)
        ge = -self.gamma*np.sum(ge1*ge2)
        W  = ap+bq+ge
        return W

