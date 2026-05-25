import numpy as np
import random
import torch
import seaborn as sns
from scipy.stats import ortho_group

import tensorly as tl
from tensorly.decomposition import tucker
from tensorly.tucker_tensor import tucker_to_tensor

from matrix_completion import * 
from sklearn.linear_model import LinearRegression
from itertools import chain
from sklearn.model_selection import KFold

# basic functions
def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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
def low_rank_debiasing(M,idx_list,y): # X[i] = idx
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
    
    # sources that owns the same column spaces with the target
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



############################ penalized_Nora function ############################
def penalized_Nora(idx_target, y_target, M0, m0, idx_list, yk_list, nk_list, p, q, p0, q0, ri, tau_value, lambda_value, T, n_iter): 
    R0 = np.linalg.svd(np.dot(M0, M0.T))[0][:, :ri]
    C0 = np.linalg.svd(np.dot(M0.T, M0))[0][:, :ri]
    PR0 = np.dot(R0, R0.T)
    PC0 = np.dot(C0, C0.T)

    M_list = []
    for k in range(len(idx_list)): 
        M_list_k = to_matrix_factor_model_mean(idx_list[k], yk_list[k], p, q, ri)
        M_list.append(M_list_k)

    R_new = R0.copy()

    k_selected_R_ini = Kmeans_with_initial_R(
        M_list, nk_list,
        p0, ri, tau_value, T=T,
        PR_initial=None
    )

    k_selected_R_no_zero_ini = [i for i in k_selected_R_ini if i > 0]
    selected_M_R_ini = [M_list[i - 1] for i in k_selected_R_no_zero_ini if i <= len(M_list)]

    if len(selected_M_R_ini) == 0:
        final_matrix_R = feature_finetune(R_new, C0, idx_target, y_target)
        metric_val_R = Fnorm(final_matrix_R - m0) ** 2 / Fnorm(m0) ** 2
        return metric_val_R

    PR_est = EPAop(selected_M_R_ini, ri)

    for iter_num in range(n_iter):  
        PR_weighted = PR_est + lambda_value * PR0

        U_R, _, _ = np.linalg.svd(PR_weighted)
        R_new = U_R[:, :p0]

        PR_new = np.dot(R_new, R_new.T)

        k_selected_R = Kmeans_with_initial_R(
            M_list, nk_list, p0, ri, 
            tau_value, T=T,  
            PR_initial=PR_new
        )

        k_selected_R_no_zero = [i for i in k_selected_R if i > 0]
        selected_M_R = [M_list[i - 1] for i in k_selected_R_no_zero if i <= len(M_list)]

        if len(selected_M_R) == 0:
            R_new = R0.copy()
            break

        PR_est = EPAop(selected_M_R, ri)

    final_matrix_R = feature_finetune(R_new, C0, idx_target, y_target)
    metric_val_R = Fnorm(final_matrix_R - m0) ** 2 / Fnorm(m0) ** 2

    return metric_val_R









############################ run_simulation functions ############################
def run_simulation_nonoracle_nocenter(scenario,p,q,p0,q0,K,K_useful,K_useless_R,K_useless_C,K_useless_RC,ri,h,Sscale,n_fold,k_fold,tau_range, lambda_range,iterate,T,n_iter):
    """
    Run pernalized NoraTMC under non-oracle background with no center on generated data and evaluate its performance.

    Parameters:
    - scenario (int): scenario number (R useful or not)
    - p (int): the row number of matrices
    - q (int): the column number of matrices
    - p0 (int): the row central dimension of matrices
    - p0 (int): the row central dimension of matrices
    - K_useful (int): useful source for both R and C
    - K_useless_R (int): useful source for both C
    - K_useless_C (int): useful source for both R
    - K_useless_RC (int): sources that are both useless for R and C
    (in this file we treat all four part as useful sources to correspond to the oracle case)
    - ri (int): the local dimension of matrices
    - h (float): similarity parameter
    - Sscale (float): scale parameter for signal part.
    - n_fold (int): number of the sample size for each fold of each source
    - k_fold (int): fold number for each source
    - iterate (int): iteration time for each set of tau and lambda.
    - T (int): iteration time for kmeans iteration to choose useful dataset
    - n_iter (int): iteration time for penalized nora optimization
    """

    set_random_seed(0)
    
    # Generate data
    Error_average = np.zeros((len(tau_range),len(lambda_range))) # penalized_Nora with varying tau and delta under oracle and useful R scenarios 
    for i in range(len(tau_range)):
        for j in range(len(lambda_range)):
            tau_value = tau_range[i]
            lambda_value = lambda_range[j]
            print("=" * 80)
            print(f" (tau, lambda) = ({float(tau_value):.6f}, {float(lambda_value):.6f}) start")
            print("=" * 80)
            for t in range(iterate):
                signal,R,C = simulation_non_oracle_nocenter(scenario,[p,q,p0,q0],
                                            [K_useful,K_useless_R,K_useless_C,K_useless_RC]
                                            ,ri,Sscale,h)
                idx_list, yk_list = generate_kfold_observations(signal,n_fold,k_fold,K,p,q)
                # target MC
                idx_target = list(chain.from_iterable(idx_list[0][i] for i in range(len(idx_list[0]))))
                y_target = list(chain.from_iterable(yk_list[0][i] for i in range(len(yk_list[0]))))
                m0_tilde = RCGD(idx_target,y_target,p,q,ri,eta=0.5)
                m0 = signal[0] # true target
                metric_target = Fnorm(m0_tilde-m0)**2/Fnorm(m0)**2
                
                # penalized NoraTMC
                metric_Nora = penalized_Nora(idx_target, y_target, m0_tilde, m0, idx_list[1:], yk_list[1:], [n_fold * k_fold] * (K-1), 
                                             p, q, p0, q0, ri, tau_value, lambda_value, T, n_iter)

                # Compute Frobenius norm errors
                Error_average[i,j] += (metric_Nora - metric_target)/iterate

    return Error_average