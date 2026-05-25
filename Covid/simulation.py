import numpy as np
from scipy.stats import ortho_group

def GOE(p):
    A = np.random.normal(size=[p,p])
    return (A+A.T)/(2*np.sqrt(p))

def simulation(size,K,ri,Sscale,h=0.01): #size = [p,q,p0,q0]
    p,q,p0,q0 = size
    R = ortho_group.rvs(dim=p)[:,:p0]
    C = ortho_group.rvs(dim=q)[:,:q0]

    PR = np.dot(R,R.T)
    PC = np.dot(C,C.T)
    
    signal = []
    for k in range(K):
        PRi = PR + h*GOE(p)
        PCi = PC + h*GOE(q)
        Ui = np.linalg.svd(PRi)[0][:,:ri]
        Vi = np.linalg.svd(PCi)[0][:,:ri]
        Sigmai = np.diag(np.random.uniform(1,2,size=ri))
        St = Sscale * Ui@Sigmai@Vi.T
        signal.append(St)
    return signal,R,C



def r_generate(p,K_useless_n,r_list):
    v = 100
    for k in range(K_useless_n):
        rotate = np.random.normal(0,v,(p,p))
        P_rotate, R = np.linalg.qr(rotate)
        r_list.append(P_rotate)
    return r_list

def simulation_non_oracle_nocenter(scenario,size,n,ri,Sscale,h=0.01): 
    p,q,p0,q0 = size
    n1,n2,n3,n4 = n
    r_list_empty = []
    signal = []
    
    r_list_R = r_generate(p,n2,r_list_empty)#生成针对R空间的旋转矩阵
    r_list_C = r_generate(q,n3,r_list_empty)#生成针对C空间的旋转矩阵
    r_list_RC_R = r_generate(p,n4,r_list_empty)#生成针对R、C空间都做旋转的旋转矩阵
    r_list_RC_C = r_generate(q,n4,r_list_empty)
    R = ortho_group.rvs(dim=p)[:,:p0]
    C = ortho_group.rvs(dim=q)[:,:q0]
    PR = np.dot(R,R.T)
    PC = np.dot(C,C.T)
    R_target_useless = ortho_group.rvs(dim=p)[:,:p0]
    C_target_useless = ortho_group.rvs(dim=q)[:,:q0]
    PR_target_useless = np.dot(R_target_useless,R_target_useless.T)
    PC_target_useless = np.dot(C_target_useless,C_target_useless.T)

    if scenario == 0:#target的左右奇异空间都能用source的信息
        PR0 = PR + h*GOE(p)
        PC0 = PC + h*GOE(q)
        U0 = np.linalg.svd(PR0)[0][:,:ri]
        V0 = np.linalg.svd(PC0)[0][:,:ri]
        Sigma0 = np.diag(np.random.uniform(1,2,size=ri))
        St = Sscale * U0@Sigma0@V0.T
        signal.append(St)
    if scenario == 1:#target的C奇异空间能用source的信息
        PR0 = PR_target_useless + h*GOE(p)
        PC0 = PC + h*GOE(q)
        U0 = np.linalg.svd(PR0)[0][:,:ri]
        V0 = np.linalg.svd(PC0)[0][:,:ri]
        Sigma0 = np.diag(np.random.uniform(1,2,size=ri))
        St = Sscale * U0@Sigma0@V0.T
        signal.append(St)
    
    if scenario == 2:#target的R奇异空间能用source的信息
        PR0 = PR + h*GOE(p)
        PC0 = PC_target_useless + h*GOE(q)
        U0 = np.linalg.svd(PR0)[0][:,:ri]
        V0 = np.linalg.svd(PC0)[0][:,:ri]
        Sigma0 = np.diag(np.random.uniform(1,2,size=ri))
        St = Sscale * U0@Sigma0@V0.T
        signal.append(St)
    if scenario == 3:#target的R、C奇异空间都不能用source的信息
        PR0 = PR_target_useless + h*GOE(p)
        PC0 = PC_target_useless + h*GOE(q)
        U0 = np.linalg.svd(PR0)[0][:,:ri]
        V0 = np.linalg.svd(PC0)[0][:,:ri]
        Sigma0 = np.diag(np.random.uniform(1,2,size=ri))
        St = Sscale * U0@Sigma0@V0.T
        signal.append(St)
    
        #行列空间均相同
    for i in range(1,n1):
        PRi = PR + h*GOE(p)
        PCi = PC + h*GOE(q)
        Ui = np.linalg.svd(PRi)[0][:,:ri]
        Vi = np.linalg.svd(PCi)[0][:,:ri]
        Sigmai = np.diag(np.random.uniform(1,2,size=ri))
        St = Sscale * Ui@Sigmai@Vi.T
        signal.append(St)
    
    #R的空间相同但是和前n/2个不同，C的空间且和前n/2个C相同
    for i in range(n1,n1+n2):
        r_i = r_list_R[i-n1]
        R_useless = r_i @ R#i从n/4到4*n/6，对其中每个数据集的R所对应的空间进行旋转
        PR_useless = np.dot(R_useless,R_useless.T)
        PRi = PR_useless + h*GOE(p)
        PCi = PC + h*GOE(q)#i从n/4到n/2，C所对应的空间不发生变化
        Ui = np.linalg.svd(PRi)[0][:,:ri]
        Vi = np.linalg.svd(PCi)[0][:,:ri]
        Sigmai = np.diag(np.random.uniform(1,2,size=ri))
        St = Sscale * Ui@Sigmai@Vi.T
        signal.append(St)
        
    #C的空间相同但是和前n/2个不同，R的空间相同且和前n/2个R相同
    for i in range(n1+n2,n1+n2+n3):
        r_i = r_list_C[i-n1-n2]
        C_useless = r_i @ C#i从4*n/6到5*n/6，对其中每个数据集的C所对应的空间进行旋转
        PC_useless = np.dot(C_useless,C_useless.T)
        PCi = PC_useless + h*GOE(p)
        PRi = PR + h*GOE(q)#i从n/2到3n/4，R所对应的空间不发生变化
        Ui = np.linalg.svd(PRi)[0][:,:ri]
        Vi = np.linalg.svd(PCi)[0][:,:ri]
        Sigmai = np.diag(np.random.uniform(1,2,size=ri))
        St = Sscale * Ui@Sigmai@Vi.T
        signal.append(St)
    
    #R和C的空间相同和前n/2个都不同
    for i in range(n1+n2+n3,n1+n2+n3+n4):
        r_i_R = r_list_RC_R[i-n1-n2-n3]
        r_i_C = r_list_RC_C[i-n1-n2-n3]
        R_useless = r_i_R @ R#i从5*n/6到n，对其中每个数据集的R所对应的空间进行旋转
        C_useless = r_i_C @ C#i从5*n/6到n，对其中每个数据集的C所对应的空间进行旋转
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


def simulation_non_oracle_center(scenario,size,n,ri,Sscale,h=0.01): 
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

    if scenario == 0:#target的左右奇异空间都能用source的信息
        PR0 = PR + h*GOE(p)
        PC0 = PC + h*GOE(q)
        U0 = np.linalg.svd(PR0)[0][:,:ri]
        V0 = np.linalg.svd(PC0)[0][:,:ri]
        Sigma0 = np.diag(np.random.uniform(1,2,size=ri))
        St = Sscale * U0@Sigma0@V0.T
        signal.append(St)

    if scenario == 1:#target的C奇异空间能用source的信息
        PR0 = PR_target_useless + h*GOE(p)
        PC0 = PC + h*GOE(q)
        U0 = np.linalg.svd(PR0)[0][:,:ri]
        V0 = np.linalg.svd(PC0)[0][:,:ri]
        Sigma0 = np.diag(np.random.uniform(1,2,size=ri))
        St = Sscale * U0@Sigma0@V0.T
        signal.append(St)
    if scenario == 2:#target的R奇异空间能用source的信息
        PR0 = PR + h*GOE(p)
        PC0 = PC_target_useless + h*GOE(q)
        U0 = np.linalg.svd(PR0)[0][:,:ri]
        V0 = np.linalg.svd(PC0)[0][:,:ri]
        Sigma0 = np.diag(np.random.uniform(1,2,size=ri))
        St = Sscale * U0@Sigma0@V0.T
        signal.append(St)
    if scenario == 3:#target的R、C奇异空间都不能用source的信息
        PR0 = PR_target_useless + h*GOE(p)
        PC0 = PC_target_useless + h*GOE(q)
        U0 = np.linalg.svd(PR0)[0][:,:ri]
        V0 = np.linalg.svd(PC0)[0][:,:ri]
        Sigma0 = np.diag(np.random.uniform(1,2,size=ri))
        St = Sscale * U0@Sigma0@V0.T
        signal.append(St)

        #行列空间均相同
    for i in range(1,n1):
        PRi = PR + h*GOE(p)
        PCi = PC + h*GOE(q)
        Ui = np.linalg.svd(PRi)[0][:,:ri]
        Vi = np.linalg.svd(PCi)[0][:,:ri]
        Sigmai = np.diag(np.random.uniform(1,2,size=ri))
        St = Sscale * Ui@Sigmai@Vi.T
        signal.append(St)
    
    #R的空间相同但是和前n/2个不同，C的空间且和前n/2个C相同
    for i in range(n1,n1+n2):
        PRi = PR_useless + h*GOE(p)
        PCi = PC + h*GOE(q)#i从n/4到n/2，C所对应的空间不发生变化
        Ui = np.linalg.svd(PRi)[0][:,:ri]
        Vi = np.linalg.svd(PCi)[0][:,:ri]
        Sigmai = np.diag(np.random.uniform(1,2,size=ri))
        St = Sscale * Ui@Sigmai@Vi.T
        signal.append(St)

    #C的空间相同但是和前n/2个不同，R的空间相同且和前n/2个R相同
    for i in range(n1+n2,n1+n2+n3):
        PCi = PC_useless + h*GOE(q)
        PRi = PR + h*GOE(p)
        Ui = np.linalg.svd(PRi)[0][:,:ri]
        Vi = np.linalg.svd(PCi)[0][:,:ri]
        Sigmai = np.diag(np.random.uniform(1,2,size=ri))
        St = Sscale * Ui@Sigmai@Vi.T
        signal.append(St)
    
    #R和C的空间相同和前n/2个都不同
    for i in range(n1+n2+n3,n1+n2+n3+n4):
        PRi = PR_useless + h*GOE(p)
        PCi = PC_useless + h*GOE(q)
        Ui = np.linalg.svd(PRi)[0][:,:ri]
        Vi = np.linalg.svd(PCi)[0][:,:ri]
        Sigmai = np.diag(np.random.uniform(1,2,size=ri))
        St = Sscale * Ui@Sigmai@Vi.T
        signal.append(St)        
    return signal,R,C


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
