import numpy as np
import random
import seaborn as sns
from scipy.stats import ortho_group

import tensorly as tl
from tensorly.decomposition import tucker
from tensorly.tucker_tensor import tucker_to_tensor
import pandas as pd

from matrix_completion import * 
from sklearn.linear_model import LinearRegression
from itertools import chain
from sklearn.model_selection import KFold
from tqdm import tqdm

# basic functions
def Fnorm(M): #operator norm
    return np.sqrt(np.sum(M**2))

def opnorm(M): #operator norm
    return np.max(np.linalg.svd(M)[1])

def Dmetric(PA1,PA2,p0): #Dmetric
    return np.sqrt(1-np.trace(np.dot(PA1,PA2))/p0)

def projection(R):
    return np.dot(np.dot(R,np.linalg.inv(np.dot(R.T,R))),R.T)
    
def EP(Pt_list,nk_list): # input a list of matrix (with same dimension) and output their weighted average matrix
    K = len(Pt_list)
    EP = nk_list[0]*Pt_list[0].copy()
    for k in range(1,K):
        EP += nk_list[k]*Pt_list[k].copy()
    return EP

def GB(Pt_list,nk_list,r): # Grassmannian Barycenter
    EPA = EP(Pt_list,np.array(nk_list))
    UG = np.linalg.svd(EPA/np.sum(nk_list))[0][:,:r]
    Ps = np.dot(UG,UG.T)
    return Ps 

# output the weighted average matrix for the left leading r0 dimension singular spaces of data
def EPAop(data,r0):
    nd = len(data)
    X0 = data[0]
    OEA = np.linalg.svd(np.dot(X0,X0.T))[0]
    Ak = OEA[:,:r0]
    PAk = np.dot(Ak,Ak.T)/nd
    for k in range(1,nd):
        Xk = data[k]
        OEA = np.linalg.svd(np.dot(Xk,Xk.T))[0]
        Ak = OEA[:,:r0]
        PAk += np.dot(Ak,Ak.T)/nd
    EPAk = PAk
    return EPAk

# same as EPAop but we consider the right singular spaces
def EPBop(data,r0):
    nd = len(data)
    X0 = data[0]
    OEB = np.linalg.svd(np.dot(X0.T,X0))[0]
    Bk = OEB[:,:r0]
    PBk = np.dot(Bk,Bk.T)/nd
    for k in range(1,nd):
        Xk = data[k]
        OEB = np.linalg.svd(np.dot(Xk.T,Xk))[0]
        Bk = OEB[:,:r0]
        PBk += np.dot(Bk,Bk.T)/nd
    EPBk = PBk
    return EPBk

def EPk_initial(data,rk): #rk is a list of individual compression dimensions
    nd = len(data)
    X0 = data[0]
    OEA,S,OEB = np.linalg.svd(X0)
    Ak = OEA[:,:rk[0]]
    Bk = OEB.T[:,:rk[0]]
    PAk = np.dot(Ak,Ak.T)/nd
    PBk = np.dot(Bk,Bk.T)/nd
    for k in range(1,nd):
        Xk = data[k]
        OEA,S,OEB = np.linalg.svd(Xk)
        Ak = OEA[:,:rk[k]]
        Bk = OEB.T[:,:rk[k]]
        PAk += np.dot(Ak,Ak.T)/nd
        PBk += np.dot(Bk,Bk.T)/nd
    EPAk = PAk
    EPBk = PBk
    return EPAk,EPBk

def GB_initial(data,p0,q0,rk):
    EPAk,EPBk = EPk_initial(data,rk)
    Q = np.linalg.svd(EPAk)
    W = np.linalg.svd(EPBk)
    A_initial = Q[0][:,:p0]
    B_initial = W[0][:,:q0]
    return A_initial,B_initial


#gradient matrix completion method
def RCGD(idx_list,yk_list,p,q,r,eta=0.5,T=20,epsilon=0.1):
    initial = np.zeros((p, q))
    n = len(yk_list)
    for i in range(n):
        initial.flat[idx_list[i]] += (p*q)*yk_list[i]/n
    u,s,vt = np.linalg.svd(initial)
    U = u[:,:r]
    V = vt.T[:,:r]
    for _ in range(T):
        G = compressed_regression(U,V,idx_list,yk_list)
        L,Lambda,Rt = np.linalg.svd(G)
        R = Rt.T
    
        M = U@G@V.T
        
        delta = np.zeros([p,q])
        for i in range(n):
            idx = idx_list[i]
            d = yk_list[i] - (M.copy()).flat[idx]
            delta.flat[idx] += (p*q)*d/n
        
        PU0 = U@U.T
        PV0 = V@V.T
        
        U_ = U@L + eta*delta@V@R@np.linalg.inv(np.diag(Lambda))
        V_ = V@R + eta*delta.T@U@L@np.linalg.inv(np.diag(Lambda))
        U = np.linalg.svd(U_)[0][:,:r]
        V = np.linalg.svd(V_)[0][:,:r]
        
        PU = U@U.T
        PV = V@V.T
        
        if (Fnorm(PU-PU0)<epsilon) and (Fnorm(PV-PV0)<epsilon):
            break
    
    return feature_finetune(U,V,idx_list,yk_list)
        
    

#debiasing dataset
def low_rank_debiasing(M,idx_list,y): 
    p,q = M.shape
    n = len(y)
    delta = np.zeros([p,q])
    for i in range(n):
        idx = idx_list[i]
        d = y[i] - (M.copy()).flat[idx]
        delta.flat[idx] += (p*q)*d/n
    return M + delta

def kfold_leave_out(idx_list, yk_list,k): #pool the datasets except the k-th dataset
    idx_list_without_k = list(chain.from_iterable(idx_list[i] for i in range(len(idx_list)) if i != k))
    y_list_without_k = list(chain.from_iterable(yk_list[i] for i in range(len(yk_list)) if i != k))
    return idx_list_without_k, y_list_without_k

def to_matrix_factor_model_mean(idx_list_k, yk_list_k,p,q,r):
    mk_debiased = np.zeros([p,q])
    for _ in range(len(yk_list_k)):
        idx_k_out, y_k_out = kfold_leave_out(idx_list_k, yk_list_k,_)
        mk_ = RCGD(idx_k_out,y_k_out,p,q,r)
        mk_debiased += low_rank_debiasing(mk_,idx_list_k[_],yk_list_k[_])
    return mk_debiased/len(yk_list_k)

def to_matrix_factor_model_list(idx_list_k, yk_list_k,p,q,r):
    M_list_k = []
    for _ in range(len(yk_list_k)):
        idx_k_out, y_k_out = kfold_leave_out(idx_list_k, yk_list_k,_)
        mk_ = RCGD(idx_k_out,y_k_out,p,q,r)
        mk_debiased = low_rank_debiasing(mk_,idx_list_k[_],yk_list_k[_])
        M_list_k.append(mk_debiased)
    return M_list_k

def compressed_regression(R,C,idx_list,y):
    n = len(y)
    p,p0 = R.shape
    q,q0 = C.shape
    f = []
    for i in range(n):
        X = idx_to_matrix(p,q,idx_list[i])
        f.append((R.T@X@C).reshape(p0*q0,))
    F = (LinearRegression(fit_intercept=False).fit(f,y).coef_).reshape(p0,q0)
    return F


def feature_finetune(R,C,idx_list,y):
    F = compressed_regression(R,C,idx_list,y)
    return R@F@C.T

def no_debiasing(idx_list,y,p,q):
    n = len(y)
    M = np.zeros([p,q])
    for i in range(n):
        idx = idx_list[i]
        d = y[i]
        M.flat[idx] += (p*q)*d/n
    return M



# functions for generating rotating matrices
def GOE(p):
    A = np.random.normal(size=[p,p])
    return (A+A.T)/(2*np.sqrt(p))

def generate_linear_regression_y(idx, Theta,sd_epsilon=1): 
    # Step 1: Calculate the linear combination of features and parameters (eta)
    eta = (Theta.copy()).flat[idx]
    # Step 2: Generate binary labels (yi) based on the logistic function
    y = eta + sd_epsilon*np.random.normal()
    return y

def generate_uniform_idx(p, q):
    return np.random.randint(0, p*q)

def idx_to_matrix(p,q,idx):
    # Create a matrix filled with zeros
    matrix = np.zeros((p, q))
    # Place the 1 at the idx position
    matrix.flat[idx] = 1 
    return matrix

def generate_kfold_observations(signal,n_fold,k_fold,K,p,q):
    idx_list = []
    yk_list = []

    for k in range(K):
        idx_list_k = []
        yk_list_k = []
        
        mk = signal[k]
        for _ in range(k_fold):
            idx_ = []
            for k in range(n_fold):
                idx = generate_uniform_idx(p, q)
                idx_.append(idx)
            idx_list_k.append(idx_)
            yk_ = [generate_linear_regression_y(idx_[i], mk,sd_epsilon=1/np.sqrt(p*q)) for i in range(len(idx_))]
            yk_list_k.append(yk_)

        idx_list.append(idx_list_k)
        yk_list.append(yk_list_k)
    
    return idx_list, yk_list

def r_generate(p,K_useless_n,r_list):
    v = 100
    for k in range(K_useless_n):
        rotate = np.random.normal(0,v,(p,p))
        P_rotate, R = np.linalg.qr(rotate)
        r_list.append(P_rotate)
    return r_list










# K-means functions
def PA_selection(Pks,Ps,tau):
    K = len(Pks) # source list length
    PA = []
    k_selected = []
    dis_sk = np.zeros(K)
    for k in range(K):
        dis_sk[k] = np.trace(Ps@Pks[k]) 
        if dis_sk[k] > tau:
            PA.append(Pks[k])
            k_selected.append(k+1)
        else:
            d = Ps.shape[0]
            PA.append(np.zeros((d,d)))
            k_selected.append(0)
    return PA,k_selected

# K-means initial estimator construction
# Pks and nk_list: the projection matrix and the corresponding sample lists for sources
def Kmeans_initial(Pks,nk_list,tau,r):
    K = len(Pks) 
    PA = []
    k_selected = []
    dis0k = np.zeros(K)
    Ps_all = GB(Pks,nk_list,r)
    for k in range(K):
        dis0k[k] = np.trace(Ps_all@Pks[k])
        if dis0k[k] > tau:
            PA.append(Pks[k])
            k_selected.append(k)
    Ps = GB(PA,np.array(nk_list)[k_selected],r)
    return Ps

# K-means function with given tau_R and tau_C
def Kmeans_with_initial(Theta_list, nk_list, p0, q0, ri, tau_R, tau_C, T,
                       PR_initial=None, PC_initial=None):
    K = len(Theta_list)
    PR_list = []
    PC_list = []
    
    for k in range(K):
        Thetak = Theta_list[k]
        Rk = np.linalg.svd(np.dot(Thetak, Thetak.T))[0][:, :ri]
        Ck = np.linalg.svd(np.dot(Thetak.T, Thetak))[0][:, :ri]
        PR_list.append(np.dot(Rk, Rk.T))
        PC_list.append(np.dot(Ck, Ck.T))
    
    if PR_initial is not None and PC_initial is not None:
        PR_iter = PR_initial
        PC_iter = PC_initial
    else:
        PR_iter = Kmeans_initial(PR_list, nk_list, tau_R, p0)
        PC_iter = Kmeans_initial(PC_list, nk_list, tau_C, q0)
    
    for t in range(T):
        PR_selected, k_selected_R = PA_selection(PR_list, PR_iter, tau_R)
        PC_selected, k_selected_C = PA_selection(PC_list, PC_iter, tau_C)
        
        selected_indices_R = [i for i in k_selected_R if i > 0]
        selected_indices_C = [i for i in k_selected_C if i > 0]
        
        if selected_indices_R:
            selected_Pk_R = [PR_selected[i-1] for i in selected_indices_R]
            selected_nk_R = [nk_list[i-1] for i in selected_indices_R]
            PR_iter = GB(selected_Pk_R, selected_nk_R, p0)
        
        if selected_indices_C:
            selected_Pk_C = [PC_selected[i-1] for i in selected_indices_C]
            selected_nk_C = [nk_list[i-1] for i in selected_indices_C]
            PC_iter = GB(selected_Pk_C, selected_nk_C, q0)
    
    return k_selected_R, k_selected_C

# K-means function with given tau_R
def Kmeans_with_initial_R(Theta_list, nk_list, p0, ri, tau_R, T,
                       PR_initial=None):
    K = len(Theta_list)
    PR_list = []
    
    for k in range(K):
        Thetak = Theta_list[k]
        Rk = np.linalg.svd(np.dot(Thetak, Thetak.T))[0][:, :ri]
        PR_list.append(np.dot(Rk, Rk.T))
    
    if PR_initial is not None:
        PR_iter = PR_initial
    else:
        PR_iter = Kmeans_initial(PR_list, nk_list, tau_R, p0)
    
    for t in range(T):
        PR_selected, k_selected_R = PA_selection(PR_list, PR_iter, tau_R)
        
        selected_indices_R = [i for i in k_selected_R if i > 0]
        selected_Pk_R = [PR_selected[i-1] for i in selected_indices_R]
        selected_nk_R = [nk_list[i-1] for i in selected_indices_R]
        PR_iter = GB(selected_Pk_R, selected_nk_R, p0)
    
    return k_selected_R

# K-means function with given tau_C
def Kmeans_with_initial_C(Theta_list, nk_list, q0, ri, tau_C, T,
                       PC_initial=None):
    K = len(Theta_list)
    PC_list = []
    
    for k in range(K):
        Thetak = Theta_list[k]
        Ck = np.linalg.svd(np.dot(Thetak.T, Thetak))[0][:, :ri]
        PC_list.append(np.dot(Ck, Ck.T))
    
    if PC_initial is not None:
        PC_iter = PC_initial
    else:
        PC_iter = Kmeans_initial(PC_list, nk_list, tau_C, q0)
    
    for t in range(T):
        PC_selected, k_selected_C = PA_selection(PC_list, PC_iter, tau_C)
        selected_indices_C = [i for i in k_selected_C if i > 0]
        selected_Pk_C = [PC_selected[i-1] for i in selected_indices_C]
        selected_nk_C = [nk_list[i-1] for i in selected_indices_C]
        PC_iter = GB(selected_Pk_C, selected_nk_C, q0)
    
    return k_selected_C

# split data functions 
def split_data(idx_list_k, yk_list_k, cv_folds):
    if isinstance(idx_list_k[0], (list, np.ndarray)):
        idx_list_k = [item for sublist in idx_list_k for item in (sublist if isinstance(sublist, (list, np.ndarray)) else [sublist])]
    if isinstance(yk_list_k[0], (list, np.ndarray)):
        yk_list_k = [item for sublist in yk_list_k for item in (sublist if isinstance(sublist, (list, np.ndarray)) else [sublist])]
    
    n = len(idx_list_k)
    fold_size = n // cv_folds
    folds = []
    
    for i in range(cv_folds):
        start = i * fold_size
        end = (i + 1) * fold_size if i < cv_folds - 1 else n
        
        val_idx = idx_list_k[start:end]
        val_y = yk_list_k[start:end]
        
        train_idx = idx_list_k[:start] + idx_list_k[end:]
        train_y = yk_list_k[:start] + yk_list_k[end:]

        folds.append({
            'train_idx': train_idx,
            'train_y': train_y,
            'val_idx': val_idx,
            'val_y': val_y
        })
    
    return folds

def split_into_folds(idx_list_k, yk_list_k, cv_folds):
    if isinstance(idx_list_k[0], (list, np.ndarray)):
        idx_list_k = [item for sublist in idx_list_k for item in (sublist if isinstance(sublist, (list, np.ndarray)) else [sublist])]
    if isinstance(yk_list_k[0], (list, np.ndarray)):
        yk_list_k = [item for sublist in yk_list_k for item in (sublist if isinstance(sublist, (list, np.ndarray)) else [sublist])]
    
    n = len(idx_list_k)
    fold_size = n // cv_folds
    folds_idx = []
    folds_y = []

    for i in range(cv_folds):
        start = i * fold_size
        end = (i + 1) * fold_size if i < cv_folds - 1 else n
        
        fold_idx = idx_list_k[start:end]
        fold_y = yk_list_k[start:end]
        
        folds_idx.append(fold_idx)
        folds_y.append(fold_y)
    
    return folds_idx, folds_y

def compute_source_folds(idx_list, yk_list, cv_folds):
    K = len(idx_list)
    source_folds = []
    
    for k in range(K):
        folds_k = split_data(idx_list[k], yk_list[k], cv_folds)
        source_folds.append(folds_k)
    
    return source_folds

def compute_single_source_matrix(indices, values, p, q, r):
    mk_initial = RCGD(indices, values, p, q, r)
    mk_debiased = low_rank_debiasing(mk_initial, indices, values)
    return mk_debiased












# data generation functions
def simulation_non_oracle_nocenter(scenario,size,n,ri,Sscale,h): 
    p,q,p0,q0 = size
    n1,n2,n3,n4 = n
    r_list_empty = []
    signal = []

    # generating rotation matrices
    r_list_R = r_generate(p,n2,r_list_empty)
    r_list_C = r_generate(q,n3,r_list_empty)
    r_list_RC_R = r_generate(p,n4,r_list_empty)
    r_list_RC_C = r_generate(q,n4,r_list_empty)
    
    R = ortho_group.rvs(dim=p)[:,:p0]
    C = ortho_group.rvs(dim=q)[:,:q0]
    PR = np.dot(R,R.T)
    PC = np.dot(C,C.T)
    R_target_useless = ortho_group.rvs(dim=p)[:,:p0]
    C_target_useless = ortho_group.rvs(dim=q)[:,:q0]
    PR_target_useless = np.dot(R_target_useless,R_target_useless.T)
    PC_target_useless = np.dot(C_target_useless,C_target_useless.T)

    if scenario == 0:
        PR0 = PR + h*GOE(p)
        PC0 = PC + h*GOE(q)
        U0 = np.linalg.svd(PR0)[0][:,:ri]
        V0 = np.linalg.svd(PC0)[0][:,:ri]
        Sigma0 = np.diag(np.random.uniform(1,2,size=ri))
        St = Sscale * U0@Sigma0@V0.T
        signal.append(St)
    if scenario == 1:
        PR0 = PR_target_useless + h*GOE(p)
        PC0 = PC + h*GOE(q)
        U0 = np.linalg.svd(PR0)[0][:,:ri]
        V0 = np.linalg.svd(PC0)[0][:,:ri]
        Sigma0 = np.diag(np.random.uniform(1,2,size=ri))
        St = Sscale * U0@Sigma0@V0.T
        signal.append(St)
    if scenario == 2:
        PR0 = PR + h*GOE(p)
        PC0 = PC_target_useless + h*GOE(q)
        U0 = np.linalg.svd(PR0)[0][:,:ri]
        V0 = np.linalg.svd(PC0)[0][:,:ri]
        Sigma0 = np.diag(np.random.uniform(1,2,size=ri))
        St = Sscale * U0@Sigma0@V0.T
        signal.append(St)
    if scenario == 3:
        PR0 = PR_target_useless + h*GOE(p)
        PC0 = PC_target_useless + h*GOE(q)
        U0 = np.linalg.svd(PR0)[0][:,:ri]
        V0 = np.linalg.svd(PC0)[0][:,:ri]
        Sigma0 = np.diag(np.random.uniform(1,2,size=ri))
        St = Sscale * U0@Sigma0@V0.T
        signal.append(St)

    # sources that owns the same column and row spaces with the target
    for i in range(1,n1):
        PRi = PR + h*GOE(p)
        PCi = PC + h*GOE(q)
        Ui = np.linalg.svd(PRi)[0][:,:ri]
        Vi = np.linalg.svd(PCi)[0][:,:ri]
        Sigmai = np.diag(np.random.uniform(1,2,size=ri))
        St = Sscale * Ui@Sigmai@Vi.T
        signal.append(St)
    
    # sources that owns the same column with the target
    for i in range(n1,n1+n2):
        r_i = r_list_R[i-n1]
        R_useless = r_i @ R
        PR_useless = np.dot(R_useless,R_useless.T)
        PRi = PR_useless + h*GOE(p)
        PCi = PC + h*GOE(q)
        Ui = np.linalg.svd(PRi)[0][:,:ri]
        Vi = np.linalg.svd(PCi)[0][:,:ri]
        Sigmai = np.diag(np.random.uniform(1,2,size=ri))
        St = Sscale * Ui@Sigmai@Vi.T
        signal.append(St)
        
    # sources that owns the same row spaces with the target
    for i in range(n1+n2,n1+n2+n3):
        r_i = r_list_C[i-n1-n2]
        C_useless = r_i @ C
        PC_useless = np.dot(C_useless,C_useless.T)
        PCi = PC_useless + h*GOE(p)
        PRi = PR + h*GOE(q)
        Ui = np.linalg.svd(PRi)[0][:,:ri]
        Vi = np.linalg.svd(PCi)[0][:,:ri]
        Sigmai = np.diag(np.random.uniform(1,2,size=ri))
        St = Sscale * Ui@Sigmai@Vi.T
        signal.append(St)
    
    # sources that owns the different column and row spaces with the target
    for i in range(n1+n2+n3,n1+n2+n3+n4):
        r_i_R = r_list_RC_R[i-n1-n2-n3]
        r_i_C = r_list_RC_C[i-n1-n2-n3]
        R_useless = r_i_R @ R
        C_useless = r_i_C @ C
        PR_useless = np.dot(R_useless,R_useless.T)
        PC_useless = np.dot(C_useless,C_useless.T)
        PRi = PR_useless + h*GOE(p)
        PCi = PC_useless + h*GOE(p)
        Ui = np.linalg.svd(PRi)[0][:,:ri]
        Vi = np.linalg.svd(PCi)[0][:,:ri]
        Sigmai = np.diag(np.random.uniform(1,2,size=ri))
        St = Sscale * Ui@Sigmai@Vi.T
        signal.append(St)        
        
    return signal,R,C


def simulation_non_oracle_center(scenario,size,n,ri,Sscale,h): 
    p,q,p0,q0 = size
    n1,n2,n3,n4 = n
    r_list_empty = []
    signal = []
    
    R = ortho_group.rvs(dim=p)[:,:p0]
    C = ortho_group.rvs(dim=q)[:,:q0]
    R_useless = ortho_group.rvs(dim=p)[:,:p0]
    C_useless = ortho_group.rvs(dim=q)[:,:q0]
    R_target_useless = ortho_group.rvs(dim=p)[:,:p0]
    C_target_useless = ortho_group.rvs(dim=q)[:,:q0]
    PR = np.dot(R,R.T)
    PC = np.dot(C,C.T)
    PR_useless = np.dot(R_useless,R_useless.T)
    PC_useless = np.dot(C_useless,C_useless.T)
    PR_target_useless = np.dot(R_target_useless,R_target_useless.T)
    PC_target_useless = np.dot(C_target_useless,C_target_useless.T)

    if scenario == 0:
        PR0 = PR + h*GOE(p)
        PC0 = PC + h*GOE(q)
        U0 = np.linalg.svd(PR0)[0][:,:ri]
        V0 = np.linalg.svd(PC0)[0][:,:ri]
        Sigma0 = np.diag(np.random.uniform(1,2,size=ri))
        St = Sscale * U0@Sigma0@V0.T
        signal.append(St)
    if scenario == 1:
        PR0 = PR_target_useless + h*GOE(p)
        PC0 = PC + h*GOE(q)
        U0 = np.linalg.svd(PR0)[0][:,:ri]
        V0 = np.linalg.svd(PC0)[0][:,:ri]
        Sigma0 = np.diag(np.random.uniform(1,2,size=ri))
        St = Sscale * U0@Sigma0@V0.T
        signal.append(St)
    if scenario == 2:
        PR0 = PR + h*GOE(p)
        PC0 = PC_target_useless + h*GOE(q)
        U0 = np.linalg.svd(PR0)[0][:,:ri]
        V0 = np.linalg.svd(PC0)[0][:,:ri]
        Sigma0 = np.diag(np.random.uniform(1,2,size=ri))
        St = Sscale * U0@Sigma0@V0.T
        signal.append(St)
    if scenario == 3:
        PR0 = PR_target_useless + h*GOE(p)
        PC0 = PC_target_useless + h*GOE(q)
        U0 = np.linalg.svd(PR0)[0][:,:ri]
        V0 = np.linalg.svd(PC0)[0][:,:ri]
        Sigma0 = np.diag(np.random.uniform(1,2,size=ri))
        St = Sscale * U0@Sigma0@V0.T
        signal.append(St)

    for i in range(1,n1):
        PRi = PR + h*GOE(p)
        PCi = PC + h*GOE(q)
        Ui = np.linalg.svd(PRi)[0][:,:ri]
        Vi = np.linalg.svd(PCi)[0][:,:ri]
        Sigmai = np.diag(np.random.uniform(1,2,size=ri))
        St = Sscale * Ui@Sigmai@Vi.T
        signal.append(St)
    
    for i in range(n1,n1+n2):
        PRi = PR_useless + h*GOE(p)
        PCi = PC + h*GOE(q)
        Ui = np.linalg.svd(PRi)[0][:,:ri]
        Vi = np.linalg.svd(PCi)[0][:,:ri]
        Sigmai = np.diag(np.random.uniform(1,2,size=ri))
        St = Sscale * Ui@Sigmai@Vi.T
        signal.append(St)
        
    for i in range(n1+n2,n1+n2+n3):
        PCi = PC_useless + h*GOE(q)
        PRi = PR + h*GOE(p)
        Ui = np.linalg.svd(PRi)[0][:,:ri]
        Vi = np.linalg.svd(PCi)[0][:,:ri]
        Sigmai = np.diag(np.random.uniform(1,2,size=ri))
        St = Sscale * Ui@Sigmai@Vi.T
        signal.append(St)
    
    for i in range(n1+n2+n3,n1+n2+n3+n4):
        PRi = PR_useless + h*GOE(p)
        PCi = PC_useless + h*GOE(q)
        Ui = np.linalg.svd(PRi)[0][:,:ri]
        Vi = np.linalg.svd(PCi)[0][:,:ri]
        Sigmai = np.diag(np.random.uniform(1,2,size=ri))
        St = Sscale * Ui@Sigmai@Vi.T
        signal.append(St)        
    
    return signal,R,C






############################ cv2DTransMC function ############################
def cv2DTransMC(idx_target, y_target, idx_list, yk_list, nk_list, p, q, p0, q0, ri, tau_grid, lambda_grid, cv_folds, T, n_iter, test_method='l_2_loss'):
    """Select tau_R, tau_C, lambda_R, lambda_C by CV procedure"""
    # step 0: obtain M_list (Theta_list in the main article)
    idx_target = list(chain.from_iterable(idx_target[i] for i in range(len(idx_target)))) 
    y_target = list(chain.from_iterable(y_target[i] for i in range(len(y_target)))) 
    M0 = RCGD(idx_target,y_target,p,q,ri) 
    M_list = []
    for k in range(len(idx_list)): 
        M_list_k = to_matrix_factor_model_mean(idx_list[k],yk_list[k],p,q,ri) 
        M_list.append(M_list_k) 

            
    # step 1: acquire training datasets and test datasets for the target samples
    train_M_list = []        # training matrices list
    val_M_list = []          # test matrices list
    target_folds = split_data(idx_target, y_target, cv_folds) # split target samples into cv_fold folds and compute training and test matrices
    for fold in range(cv_folds):
        if len(target_folds[fold]['train_idx'])/cv_folds >= 5: 
            target_fold_train_idx, target_fold_train_y = split_into_folds(target_folds[fold]['train_idx'], target_folds[fold]['train_y'],cv_folds)
            M_0_train = to_matrix_factor_model_mean(target_fold_train_idx, target_fold_train_y,p,q,ri)
        else:
            M_0_train = compute_single_source_matrix(target_folds[fold]['train_idx'], target_folds[fold]['train_y'], p, q, ri)
        if len(target_folds[fold]['val_idx'])/cv_folds >= 5:
            target_fold_val_idx, target_fold_val_y = split_into_folds(target_folds[fold]['val_idx'], target_folds[fold]['val_y'],cv_folds)
            M_0_val = to_matrix_factor_model_mean(target_fold_val_idx, target_fold_val_y,p,q,ri)
        else:
            M_0_val = compute_single_source_matrix(target_folds[fold]['val_idx'], target_folds[fold]['val_y'], p, q, ri)
        train_M_list.append(M_0_train)
        val_M_list.append(M_0_val)

    
    # step 2: use CV to tune tau_U and lambda_U
    # initializations
    best_tau_R = tau_grid[0]
    best_lambda_R = lambda_grid[0]
    best_metric_R_Fnorm = -np.inf
    best_metric_R_l2 = np.inf
    R_target_all_targetdata = np.linalg.svd(np.dot(M0, M0.T))[0][:, :ri]
    C_target_all_targetdata = np.linalg.svd(np.dot(M0.T, M0))[0][:, :ri]
    PR_target_all_targetdata = np.dot(R_target_all_targetdata, R_target_all_targetdata.T)
    PC_target_all_targetdata = np.dot(C_target_all_targetdata, C_target_all_targetdata.T)
    # run over all tau_grid and lambda_grid to select the optimal tau_R and lambda_R
    for tau_R in tau_grid:
        for lambda_R in lambda_grid:
            total_metric_R = 0.0
            total_folds_R = 0
            for fold in range(cv_folds):                
                try:
                    R_target = np.linalg.svd(np.dot(train_M_list[fold], train_M_list[fold].T))[0][:, :ri]
                    PR_target = np.dot(R_target, R_target.T)
                    k_selected_R_ini = Kmeans_with_initial_R(
                        M_list, nk_list,
                        p0, ri, tau_R, T=T,
                        PR_initial=None
                    )
                    k_selected_R_no_zero_ini = [i for i in k_selected_R_ini if i > 0]
                    if len(k_selected_R_no_zero_ini) == 0:
                        continue
                    selected_M_R_ini = [M_list[i-1] for i in k_selected_R_no_zero_ini if i <= len(M_list)]
                    PR_est = EPAop(selected_M_R_ini, ri)

                    # penalty function iteration
                    for iter_num in range(n_iter):  
                        # weighted average
                        PR_weighted = PR_est + lambda_R * PR_target
                        U_R, _, _ = np.linalg.svd(PR_weighted)
                        R_new = U_R[:, :p0]
                        # K-means
                        PR_new = np.dot(R_new, R_new.T)
                        k_selected_R = Kmeans_with_initial_R(
                            M_list, nk_list, p0, ri, 
                            tau_R, T=T,  
                            PR_initial=PR_new
                        )
                        k_selected_R_no_zero = [i for i in k_selected_R if i > 0]
                        selected_M_R = [M_list[i-1] for i in k_selected_R_no_zero if i <= len(M_list)]
                        # update R
                        if len(selected_M_R) > 0:
                            PR_est = EPAop(selected_M_R, ri)

                    
                    # calculate the metric values
                    if len(val_M_list) > 0:
                        # target的验证集矩阵
                        target_val_M = val_M_list[fold]  
                        if test_method == 'F_norm':      # F_norm
                            projected_R = PR_new @ target_val_M @ PC_target_all_targetdata
                            metric_val_R = np.linalg.norm(projected_R, 'fro')**2
                        elif test_method == 'l_2_loss':  # l_2_loss
                            final_matrix_R = feature_finetune(R_new, C_target_all_targetdata, target_folds[fold]['val_idx'], target_folds[fold]['val_y'])
                            metric_val_R = Fnorm(final_matrix_R-target_val_M)**2/Fnorm(target_val_M)**2
                        total_metric_R += metric_val_R
                        total_folds_R += 1
                
                except Exception as e:
                    print(f"[R-CV error] tau_R={tau_R}, lambda_R={lambda_R}, fold={fold}: {e}")
                    continue

            # calculate the average matric value across all folds
            if total_folds_R > 0:
                avg_metric_R = total_metric_R / total_folds_R
            else:
                print(f"[R-CV skipped] tau_R={tau_R}, lambda_R={lambda_R}: no valid folds")
                continue

            # update tau_R and lambda_R
            if test_method == 'F_norm':
                if avg_metric_R >= best_metric_R_Fnorm:
                    best_metric_R_Fnorm = avg_metric_R
                    best_tau_R = tau_R
                    best_lambda_R = lambda_R
            elif test_method == 'l_2_loss':
                if avg_metric_R <= best_metric_R_l2:
                    best_metric_R_l2 = avg_metric_R
                    best_tau_R = tau_R
                    best_lambda_R = lambda_R

    
    # step 3: use CV to tune tau_C and lambda_C
    best_tau_C = tau_grid[0]
    best_lambda_C = lambda_grid[0]
    best_metric_C_Fnorm = -np.inf
    best_metric_C_l2 = np.inf
    for tau_C in tau_grid:
        for lambda_C in lambda_grid:
            total_metric_C = 0.0
            total_folds_C = 0
            for fold in range(cv_folds):
                try:
                    C_target = np.linalg.svd(np.dot(train_M_list[fold].T, train_M_list[fold]))[0][:, :ri]
                    PC_target = np.dot(C_target, C_target.T)
                    k_selected_C_ini = Kmeans_with_initial_C( 
                        M_list, nk_list,
                        q0, ri, tau_C, T=T,
                        PC_initial=None
                    )
                    k_selected_C_no_zero_ini = [i for i in k_selected_C_ini if i > 0]
                    if len(k_selected_C_no_zero_ini) == 0:
                        continue
                    selected_M_C_ini = [M_list[i-1] for i in k_selected_C_no_zero_ini if i <= len(M_list)]
                    PC_est = EPBop(selected_M_C_ini, ri)

                    for iter_num in range(n_iter):  
                        PC_weighted = PC_est + lambda_C * PC_target
                        U_C, _, _ = np.linalg.svd(PC_weighted)
                        C_new = U_C[:, :q0]
                        PC_new = np.dot(C_new, C_new.T)
                        k_selected_C = Kmeans_with_initial_C(
                            M_list, nk_list, q0, ri, 
                            tau_C, T=T,  
                            PC_initial=PC_new
                        )
                        k_selected_C_no_zero = [i for i in k_selected_C if i > 0]
                        selected_M_C = [M_list[i-1] for i in k_selected_C_no_zero if i <= len(M_list)]
                        if len(selected_M_C) > 0:
                            PC_est = EPBop(selected_M_C, ri)

                    
                    if len(val_M_list) > 0:
                        target_val_M = val_M_list[fold]  
                        if test_method == 'F_norm':      # F_norm 
                            projected_C = PR_target_all_targetdata @ target_val_M @ PC_new
                            metric_val_C = np.linalg.norm(projected_C, 'fro')**2
                        elif test_method == 'l_2_loss':  # l_2_loss 
                            final_matrix_C = feature_finetune(R_target_all_targetdata, C_new, target_folds[fold]['val_idx'], target_folds[fold]['val_y'])
                            metric_val_C = Fnorm(final_matrix_C-target_val_M)**2/Fnorm(target_val_M)**2
                        total_metric_C += metric_val_C
                        total_folds_C += 1
                
                except Exception as e:
                    print(f"[C-CV error] tau_C={tau_C}, lambda_C={lambda_C}, fold={fold}: {e}")
                    continue

            if total_folds_C > 0:
                avg_metric_C = total_metric_C / total_folds_C
            else:
                print(f"[C-CV skipped] tau_C={tau_C}, lambda_C={lambda_C}: no valid folds")
                continue

            if test_method == 'F_norm':
                if avg_metric_C >= best_metric_C_Fnorm:
                    best_metric_C_Fnorm = avg_metric_C
                    best_tau_C = tau_C
                    best_lambda_C = lambda_C
            elif test_method == 'l_2_loss':
                if avg_metric_C <= best_metric_C_l2:
                    best_metric_C_l2 = avg_metric_C
                    best_tau_C = tau_C
                    best_lambda_C = lambda_C

    
    # step 4: use the optimal tau_R, lambda_R, tau_C and lambda_C to rerun the whole process
    R_target = np.linalg.svd(np.dot(M0, M0.T))[0][:, :ri]
    C_target = np.linalg.svd(np.dot(M0.T, M0))[0][:, :ri]
    PR_target = np.dot(R_target, R_target.T)
    PC_target = np.dot(C_target, C_target.T)
    
    PR_list = []
    PC_list = []
    K = len(idx_list)
    for k in range(K): 
        M_k = M_list[k]
        Rk = np.linalg.svd(np.dot(M_k, M_k.T))[0][:, :ri]
        Ck = np.linalg.svd(np.dot(M_k.T, M_k))[0][:, :ri]
        PR_list.append(np.dot(Rk, Rk.T))
        PC_list.append(np.dot(Ck, Ck.T))


    k_selected_R, k_selected_C = Kmeans_with_initial(
        M_list, nk_list, p0, q0, ri, 
        best_tau_R, best_tau_C, T=T
    )
    k_selected_R = [i for i in k_selected_R if i > 0]
    k_selected_C = [i for i in k_selected_C if i > 0]

    selected_M_R = [M_list[i-1] for i in k_selected_R if i <= len(M_list)]
    selected_M_C = [M_list[i-1] for i in k_selected_C if i <= len(M_list)]
    
    for iter_num in range(n_iter):  
        if len(selected_M_R) > 0 and len(selected_M_C) > 0:
            PR_est = EPAop(selected_M_R, ri)
            PC_est = EPBop(selected_M_C, ri)
        PR_weighted = PR_est + best_lambda_R * PR_target
        PC_weighted = PC_est + best_lambda_C * PC_target
        U_R, _, _ = np.linalg.svd(PR_weighted)
        U_C, _, _ = np.linalg.svd(PC_weighted)
        R_new = U_R[:, :p0]
        C_new = U_C[:, :q0]
        PR_new = np.dot(R_new, R_new.T)
        PC_new = np.dot(C_new, C_new.T)
        k_selected_R, k_selected_C = Kmeans_with_initial(
            M_list, nk_list, p0, q0, ri, 
            best_tau_R, best_tau_C, T=T,  
            PR_initial=PR_new, PC_initial=PC_new
        )
        k_selected_R_no_zero = [i for i in k_selected_R if i > 0]
        k_selected_C_no_zero = [i for i in k_selected_C if i > 0]
        selected_M_R = [M_list[i-1] for i in k_selected_R_no_zero if i <= len(M_list)]
        selected_M_C = [M_list[i-1] for i in k_selected_C_no_zero if i <= len(M_list)]

    
    return R_new, C_new , best_tau_R, best_tau_C, best_lambda_R, best_lambda_C





########################### main function #################################
def main(iterate, scenario, data_generate, p, q, p0, q0, ri, n_fold, k_fold, K, 
         K_useful, K_useless_R, K_useless_C, K_useless_RC, Sscale, h,
         tau_grid, lambda_grid, cv_folds, T_kmeans, n_iter, test_method):
    """
    Main function: run iterate experiments and compare the performance of five matrix completion methods
    
    Parameters:
    -----------
    iterate : int
        Number of experiment repetitions
    scenario : int
        Data generation scenario
    data_generate: int
        Used during data generation to determine whether useless sources have centers
    p, q : int
        Dimensions of the target matrix
    p0, q0 : int
        Dimensions of the row and column center subspaces
    ri : int
        Dimension of the local subspace
    n_fold : int
        Sample size for each dataset in each fold
    k_fold : int
        Number of splits for each dataset
    K : int
        Total number of datasets (one target + K-1 sources)
    K_useful, K_useless_R, K_useless_C, K_useless_RC : int
        Number of source datasets of each type
    Sscale : float
        Signal strength scaling factor
    h : float
        Subspace similarity
    tau_grid, lambda_grid : array
        Cross-validation parameter grids
    cv_folds : int
        Number of cross-validation folds used in the cv2DTransMC function
    T_kmeans : int
        Number of k-means iterations used in the cv2DTransMC function
    n_iter : int
        Number of penalty iterations used in the cv2DTransMC function
    test_method : str
        Evaluation metric method used in the cv2DTransMC function
    """
    
    # Store the results of each iteration
    results = {
        'target': [],
        'blindMC': [],
        'CVTransMC': [],
        'unbiasedMC': [],
        'Tensor Completion': []
    }
    
    print("=" * 80)
    print(f"Method Comparison for 5 methods")
    print("=" * 80)
    
    # Main loop
    for iteration in tqdm(range(iterate), desc="Simulation progress"):
        print(f"\n--- The {iteration + 1}/{iterate}-th iteration starts ---")
        
        # Step 1: Generate data
        ##Generate the target dataset and K-1 source datasets without centers
        if data_generate == 0:
            signal,R,C = simulation_non_oracle_nocenter(scenario,[p,q,p0,q0],
                                                        [K_useful,K_useless_R,K_useless_C,K_useless_RC]
                                                        ,ri,Sscale,h)

        ##Generate the target dataset and K-1 source datasets, where all useless sources have centers
        elif data_generate == 1:
            signal, R, C = simulation_non_oracle_center(
                scenario, [p, q, p0, q0],
                [K_useful, K_useless_R, K_useless_C, K_useless_RC],
                ri, Sscale, h
            )
        
        # Generate k-fold observed data for the target and sources
        idx_list, yk_list = generate_kfold_observations(
            signal, n_fold, k_fold, K, p, q
        )
        
        # Ground truth of the target matrix
        m0 = signal[0]  
        
        # Complete target data (merge all folds of the target)
        idx_target = list(chain.from_iterable(idx_list[0][i] for i in range(len(idx_list[0]))))
        y_target = list(chain.from_iterable(yk_list[0][i] for i in range(len(yk_list[0]))))
        
        iteration_losses = []
        
        # ============ Method 1: Target Only ============
        try:
            m0_tilde = RCGD(idx_target, y_target, p, q, ri, eta=0.5)
            loss_target = (Fnorm(m0_tilde - m0) ** 2) / (Fnorm(m0) ** 2)
        except Exception as e:
            print(f"  Target method fails: {e}")
            loss_target = np.nan
        results['target'].append(loss_target)
        iteration_losses.append(loss_target)
        
        # ============ Method 2: Blind Transfer MC ============
        try:
            debiased_matrices = []
            for k in range(K):
                M_list_k = to_matrix_factor_model_list(idx_list[k], yk_list[k], p, q, ri)
                debiased_matrices += M_list_k
            
            RGB, CGB = GB_initial(debiased_matrices, p0, q0, [ri] * len(debiased_matrices))
            m0_feature_finetune = feature_finetune(RGB, CGB, idx_target, y_target)
            loss_blind = (Fnorm(m0_feature_finetune - m0) ** 2) / (Fnorm(m0) ** 2)
        except Exception as e:
            print(f"  Blind Transfer MC method fails: {e}")
            loss_blind = np.nan
        results['blindMC'].append(loss_blind)
        iteration_losses.append(loss_blind)
        
        # ============ Method 3: Nora Transfer MC with CV ============
        try:
            R_est, C_est, tau_R, tau_C, lambda_R, lambda_C = cv2DTransMC(
                idx_list[0], yk_list[0],  # target
                idx_list[1:], yk_list[1:],  # sources
                [n_fold*k_fold]*(K-1), p, q, p0, q0, ri,
                tau_grid, lambda_grid, cv_folds, T_kmeans, n_iter, test_method
            )
            final_matrix = feature_finetune(R_est, C_est, idx_target, y_target)
            loss_cv = (Fnorm(final_matrix - m0) ** 2) / (Fnorm(m0) ** 2)
        except Exception as e:
            print(f"  CV Trans MC fails: {e}")
            loss_cv = np.nan
        results['CVTransMC'].append(loss_cv)
        iteration_losses.append(loss_cv)
        
        # ============ Method 4: Unbiased Transfer MC ============
        try:
            unbiased_matrices = []
            for k in range(1, K):
                idx_k_pool = list(chain.from_iterable(idx_list[k][i] for i in range(len(idx_list[k]))))
                y_k_pool = list(chain.from_iterable(yk_list[k][i] for i in range(len(yk_list[k]))))
                unbiased_matrices.append(no_debiasing(idx_k_pool, y_k_pool, p, q))
            
            RGB, CGB = GB_initial(unbiased_matrices, p0, q0, [ri] * len(unbiased_matrices))
            m0_feature_finetune = feature_finetune(RGB, CGB, idx_target, y_target)
            loss_unbiased = (Fnorm(m0_feature_finetune - m0) ** 2) / (Fnorm(m0) ** 2)
        except Exception as e:
            print(f"  Unbiased Transfer MC fails: {e}")
            loss_unbiased = np.nan
        results['unbiasedMC'].append(loss_unbiased)
        iteration_losses.append(loss_unbiased)
        
        # ============ Method 5: Tensor Completion ============
        try:
            tensor = np.zeros((K, p, q))
            alpha = p * q / (n_fold * k_fold)  # Correct the alpha calculation formula
            
            for k in range(K):
                idx_k_pool = list(chain.from_iterable(idx_list[k][i] for i in range(len(idx_list[k]))))
                y_k_pool = list(chain.from_iterable(yk_list[k][i] for i in range(len(yk_list[k]))))
                for i, idx in enumerate(idx_k_pool):
                    tensor[k].flat[idx] += y_k_pool[i] * alpha
            
            rank = [min(p0 * q0, K), p0, q0]
            core, factors = tucker(tensor, rank=rank)
            reconstructed_tensor = tucker_to_tensor((core, factors))
            m0_tensor = reconstructed_tensor[0, :, :]
            loss_tensor = (Fnorm(m0_tensor - m0) ** 2) / (Fnorm(m0) ** 2)
        except Exception as e:
            print(f"  Tensor Completion fails: {e}")
            loss_tensor = np.nan
        results['Tensor Completion'].append(loss_tensor)
        iteration_losses.append(loss_tensor)
        
        # Output the summary of this iteration
        print(f"  The summary of the {iteration + 1}-th iteration: target={loss_target:.5f}, blindMC={loss_blind:.5f}, "
              f"CV2DTransMC={loss_cv:.5f}, unbiasedMC={loss_unbiased:.5f}, Tensor={loss_tensor:.5f}")
    
    # ============ Result Summary and Saving ============
    # Create DataFrame
    df_results = pd.DataFrame(results)
    df_results.insert(0, 'iterate time', range(1, len(df_results) + 1))  # Insert a column directly at the beginning
    
    means = df_results.iloc[:, 1:].mean(axis=0)    # Calculate the mean of each column (skip the first column)
    
    # Add the mean row
    mean_row = pd.DataFrame([[ 'Mean' ] + means.values.tolist()], columns=df_results.columns)
    df_with_means = pd.concat([df_results, mean_row], ignore_index=True)
    
    # Save to CSV file
    filename = f"cvTransMC_results_iter{iterate}_p{p}_q{q}_K{K}.csv"
    df_with_means.to_csv(filename, index=False, float_format='%.6f')
    print(f"\nThe results have been saved in: {filename}")
    
    # Print the detailed results table
    print("\n" + "=" * 100)
    print("Average performance:")
    print("=" * 100)
    for method, mean_val in means.items():
        print(f"  {method:15s}: {mean_val:.6f}")
    
    # Return the result DataFrames for further analysis
    return df_results, df_with_means