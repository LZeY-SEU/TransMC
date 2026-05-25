import numpy as np
from sklearn.linear_model import LinearRegression
from simulation import *
from itertools import chain


def Fnorm(M): #operator norm
    return np.sqrt(np.sum(M**2))

def opnorm(M): #operator norm
    return np.max(np.linalg.svd(M)[1])

def Dmetric(PA1,PA2,p0): #Dmetric
    return np.sqrt(1-np.trace(np.dot(PA1,PA2))/p0)

def projection(R):
    return np.dot(np.dot(R,np.linalg.inv(np.dot(R.T,R))),R.T)



#2D-PCA methods

def PCA2D2(data,p0,q0): #2D2PCA algorithm
    nd = len(data)
    X0 = data[0]
    EXXT = np.dot(X0,X0.T)/nd
    EXTX = np.dot(X0.T,X0)/nd
    for k in range(1,nd):
        Xk = data[k]
        EXXT += np.dot(Xk,Xk.T)/nd
        EXTX += np.dot(Xk.T,Xk)/nd
    OEA = np.linalg.svd(EXXT)[0] #2D2PCA
    A2D2 = OEA[:,:p0]
    OEB = np.linalg.svd(EXTX)[0]
    B2D2 = OEB[:,:q0]
    return A2D2,B2D2

def PE(data,p0,q0,initial,epsilon=0.01,T=10): #PE algorithm initial = [A2D2,B2D2]
    nd = len(data)
    for t in range(T): #MPCA iteration
        AM0 = initial[0]
        BM0 = initial[1]
        PAM0 = np.dot(AM0,AM0.T)
        PBM0 = np.dot(BM0,BM0.T)
        X0 = data[0]
        EXTPAX = np.dot(X0.T,np.dot(PAM0,X0))/nd
        for k in range(1,nd):
            Xk = data[k]
            EXTPAX += np.dot(Xk.T,np.dot(PAM0,Xk))/nd
        X0 = data[0]
        EXPBXT = np.dot(X0,np.dot(PBM0,X0.T))/nd
        for k in range(1,nd):
            Xk = data[k]
            EXPBXT += np.dot(Xk,np.dot(PBM0,Xk.T))/nd
        OEA = np.linalg.svd(EXPBXT)[0]
        OEB = np.linalg.svd(EXTPAX)[0]
        AM = OEA[:,:p0]
        BM = OEB[:,:q0]
        PAM = np.dot(AM,AM.T)
        PBM = np.dot(BM,BM.T)
        if (Fnorm(PAM-PAM0)<epsilon) and (Fnorm(PBM-PBM0)<epsilon):
            return AM,BM
        else:
            initial = [AM,BM]
    return AM,BM

def Huberweight(data,p0,q0,initial,tau='default'): 
    W = []
    AH0 = initial[0]
    BH0 = initial[1]
    PAH0 = np.dot(AH0,AH0.T)
    PBH0 = np.dot(BH0,BH0.T)
    k = [Fnorm(xs-np.dot(PAH0,np.dot(xs,PBH0))) for xs in data]
    if tau == 'default':
        tau = np.median(k)
    for xs in data:
        xc = np.dot(PAH0,np.dot(xs,PBH0))
        if Fnorm(xs-xc) <= tau:
            W.append(1/(p0*q0))
        else:
            us = np.dot(AH0.T,np.dot(xs,BH0))
            w = tau/((p0*q0)*np.sqrt(Fnorm(xs)**2-Fnorm(us)**2/(p0*q0)))
            W.append(w)
    return np.array(W)

def HuberPCA(data,p0,q0,initial,epsilon=0.01,T=10): #Huber algorithm initial = [A2D2,B2D2]
    nd = len(data)
    for t in range(T):
        AH0 = initial[0]
        BH0 = initial[1]
        W = Huberweight(data,p0,q0,initial)
        PAH0 = np.dot(AH0,AH0.T)
        PBH0 = np.dot(BH0,BH0.T)
        X0 = data[0]
        EXTPAX = W[0]*np.dot(X0.T,np.dot(PAH0,X0))/nd
        for k in range(1,nd):
            Xk = data[k]
            EXTPAX += W[k]*np.dot(Xk.T,np.dot(PAH0,Xk))/nd
        X0 = data[0]
        EXPBXT = W[0]*np.dot(X0,np.dot(PBH0,X0.T))/nd
        for k in range(1,nd):
            Xk = data[k]
            EXPBXT += W[k]*np.dot(Xk,np.dot(PBH0,Xk.T))/nd
        OEA = np.linalg.svd(EXPBXT)[0]
        OEB = np.linalg.svd(EXTPAX)[0]
        AH = OEA[:,:p0]
        BH = OEB[:,:q0]
        PAH = np.dot(AH,AH.T)
        PBH = np.dot(BH,BH.T)
        if (Fnorm(PAH-PAH0)<epsilon) and (Fnorm(PBH-PBH0)<epsilon):
            return AH,BH,W
        else:
            initial = [AH,BH]
        return AH,BH,W
    
#Grassmannian barycenter methods
    
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

def EPAkt(data,rk,B0):
    nd = len(data)
    X0 = data[0]
    OEA = np.linalg.svd(np.dot(X0,B0))[0]
    Ak = OEA[:,:rk[0]]
    EPAk = np.dot(Ak,Ak.T)/nd
    for k in range(1,nd):
        Xk = data[k]
        OEA = np.linalg.svd(np.dot(Xk,B0))[0]
        Ak = OEA[:,:rk[k]]
        EPAk += np.dot(Ak,Ak.T)/nd
    return EPAk

def EPBkt(data,rk,A0):
    nd = len(data)
    X0 = data[0]
    OEB = np.linalg.svd(np.dot(X0.T,A0))[0]
    Bk = OEB[:,:rk[0]]
    EPBk = np.dot(Bk,Bk.T)/nd
    for k in range(1,nd):
        Xk = data[k]
        OEB = np.linalg.svd(np.dot(Xk.T,A0))[0]
        Bk = OEB[:,:rk[k]]
        EPBk += np.dot(Bk,Bk.T)/nd
    return EPBk

def GB_iterate(data,p0,q0,rk,initial,epsilon=0.01,T=10): #initial = [Aop,Bop]
    for t in range(1,T+1):
        A0 = initial[0]
        B0 = initial[1]
        PA0 = np.dot(A0,A0.T)
        PB0 = np.dot(B0,B0.T)
        EPAk = EPAkt(data,rk,B0)
        EPBk = EPBkt(data,rk,A0)
        A = np.linalg.svd(EPAk)[0][:,:p0]
        B = np.linalg.svd(EPBk)[0][:,:q0]
        PA = np.dot(A,A.T)
        PB = np.dot(B,B.T)
        if (Fnorm(PA-PA0)<epsilon) and (Fnorm(PB-PB0)<epsilon):
            return A,B
        else:
            initial = [A,B]
    return A,B



def EPk_APVD(data,rk): #rk is a list of individual compression dimensions
    nd = len(data)
    X0 = data[0]
    OEA,S,OEB = np.linalg.svd(X0)
    Ak = OEA[:,:rk[0]]@np.diag(S[:rk[0]])
    Bk = OEB.T[:,:rk[0]]@np.diag(S[:rk[0]])
    PAk = np.dot(Ak,Ak.T)/nd
    PBk = np.dot(Bk,Bk.T)/nd
    for k in range(1,nd):
        Xk = data[k]
        OEA,S,OEB = np.linalg.svd(Xk)
        Ak = OEA[:,:rk[k]]@np.diag(S[:rk[k]])
        Bk = OEB.T[:,:rk[k]]@np.diag(S[:rk[k]])
        PAk += np.dot(Ak,Ak.T)/nd
        PBk += np.dot(Bk,Bk.T)/nd
    EPAk = PAk
    EPBk = PBk
    return EPAk,EPBk

def APVD(data,p0,q0,rk):
    EPAk,EPBk = EPk_APVD(data,rk)
    Q = np.linalg.svd(EPAk)
    W = np.linalg.svd(EPBk)
    A_initial = Q[0][:,:p0]
    B_initial = W[0][:,:q0]
    return A_initial,B_initial

def EPAkt_APVD(data,rk,B0):
    nd = len(data)
    X0 = data[0]
    OEA,S,OEB = np.linalg.svd(np.dot(X0,B0))
    Ak = OEA[:,:rk[0]]@np.diag(S[:rk[0]])
    EPAk = np.dot(Ak,Ak.T)/nd
    for k in range(1,nd):
        Xk = data[k]
        OEA,S,OEB = np.linalg.svd(np.dot(Xk,B0))
        Ak = OEA[:,:rk[k]]@np.diag(S[:rk[0]])
        EPAk += np.dot(Ak,Ak.T)/nd
    return EPAk

def EPBkt_APVD(data,rk,A0):
    nd = len(data)
    X0 = data[0]
    OEB,S,OEA = np.linalg.svd(np.dot(X0.T,A0))
    Bk = OEB[:,:rk[0]]@np.diag(S[:rk[0]])
    EPBk = np.dot(Bk,Bk.T)/nd
    for k in range(1,nd):
        Xk = data[k]
        OEB,S,OEA = np.linalg.svd(np.dot(Xk.T,A0))
        Bk = OEB[:,:rk[k]]@np.diag(S[:rk[0]])
        EPBk += np.dot(Bk,Bk.T)/nd
    return EPBk

def APVD_iterate(data,p0,q0,rk,initial,epsilon=0.01,T=10): #initial = [Aop,Bop]
    for t in range(1,T+1):
        A0 = initial[0]
        B0 = initial[1]
        PA0 = np.dot(A0,A0.T)
        PB0 = np.dot(B0,B0.T)
        EPAk = EPAkt_APVD(data,rk,B0)
        EPBk = EPBkt_APVD(data,rk,A0)
        A = np.linalg.svd(EPAk)[0][:,:p0]
        B = np.linalg.svd(EPBk)[0][:,:q0]
        PA = np.dot(A,A.T)
        PB = np.dot(B,B.T)
        if (Fnorm(PA-PA0)<epsilon) and (Fnorm(PB-PB0)<epsilon):
            return A,B
        else:
            initial = [A,B]
    return A,B


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


def to_matrix_factor_model_list(idx_list_k, yk_list_k,p,q,r):
    M_list_k = []
    for _ in range(len(yk_list_k)):
        idx_k_out, y_k_out = kfold_leave_out(idx_list_k, yk_list_k,_)
        mk_ = RCGD(idx_k_out,y_k_out,p,q,r)
        mk_debiased = low_rank_debiasing(mk_,idx_list_k[_],yk_list_k[_])
        M_list_k.append(mk_debiased)
    return M_list_k

def to_matrix_factor_model_mean(idx_list_k, yk_list_k,p,q,r):
    mk_debiased = np.zeros([p,q])
    for _ in range(len(yk_list_k)):
        idx_k_out, y_k_out = kfold_leave_out(idx_list_k, yk_list_k,_)
        mk_ = RCGD(idx_k_out,y_k_out,p,q,r)
        mk_debiased += low_rank_debiasing(mk_,idx_list_k[_],yk_list_k[_])
    return mk_debiased/len(yk_list_k)

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


def EP(Pt_list,nk_list): #输入一个矩阵列表 并将所有列表中的矩阵加权求和输出一个矩阵
    K = len(Pt_list)
    EP = nk_list[0]*Pt_list[0].copy()
    for k in range(1,K):
        EP += nk_list[k]*Pt_list[k].copy()
    return EP

#这里的r在R和C下分别是p0和q0
def GB(Pt_list,nk_list,r): 
    EPA = EP(Pt_list,np.array(nk_list))
    UG = np.linalg.svd(EPA/np.sum(nk_list))[0][:,:r]
    Ps = np.dot(UG,UG.T)
    return Ps 

#将每个数据集的X求其对应的前r0维左特征空间并进行加权求和
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

#将每个数据集的X求其对应的前r0维右特征空间并进行加权求和
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


def MCPCAop_nora(data1,data2,p0,q0,ri):
    EPAk = EPAop(data1,ri)
    EPBk = EPBop(data2,ri)
    Q = np.linalg.svd(EPAk)
    W = np.linalg.svd(EPBk)
    Aop = Q[0][:,:p0]
    Bop = W[0][:,:q0]
    return Aop,Bop

#在新的non-oracle情况下，我们需要把target数据集（也就是0指标）排除在数据集选择之外，所以算法中对应的
#“把0指标一定包含在有用数据集之内”的步骤要进行修改

#Pks是对应的输入进来的去除掉target之后的剩余的K-1个数据集中的debiased_matrices
#Ps是我们的中心
#tau是选择数据集的参数
def PA_selection(Pks,Ps,tau):
    K = len(Pks)#这里的"K"实际上是K-1个source
    PA = []
    k_selected = []
    dis_sk = np.zeros(K)
    for k in range(K):
        dis_sk[k] = np.trace(Ps@Pks[k])#将第k个source中的P与Ps求tr然后放入dis_sk之中
        if dis_sk[k] > tau:
            PA.append(Pks[k])
            k_selected.append(k+1)
        else:
            d = Ps.shape[0]
            PA.append(np.zeros((d,d)))
            k_selected.append(0)
    return PA,k_selected

#另一种更有优势的初始估计量（最后的算法里用的也是这种估计量作为迭代初始值）
def Kmeans_initial(Pks,nk_list,tau,r):
    K = len(Pks)#这里的"K"实际上是K-1个source
    PA = []
    k_selected = []
    dis0k = np.zeros(K)
    Ps_all = GB(Pks,nk_list,r)
    for k in range(K):
        dis0k[k] = np.trace(Ps_all@Pks[k])#将第k个source中的P与target的P求tr然后放入dis0k之中
        if dis0k[k] > tau:
            PA.append(Pks[k])
            k_selected.append(k+1)
    Ps = GB(PA,np.array(nk_list)[k_selected],r)
    return Ps

#分别针对R和C的K-mean算法：
#Theta_list实际就是debiased_matrices,ri实际上就是EPkop中的r0
def Kmeans(Theta_list,nk_list,p0,q0,ri,tau_R,tau_C,T=10): #nora means Non-oracle
    K = len(Theta_list)
    PR_list = []
    PC_list = []
    
    
    for k in range(K):
        Thetak = Theta_list[k]
        Rk = np.linalg.svd(np.dot(Thetak,Thetak.T))[0][:,:ri]
        Ck = np.linalg.svd(np.dot(Thetak.T,Thetak))[0][:,:ri]
        PR_list.append(np.dot(Rk,Rk.T))
        PC_list.append(np.dot(Ck,Ck.T))

    
    #这里给出kmeans的initial value
    PR_ini = Kmeans_initial(PR_list,nk_list,tau_R,p0)
    PC_ini = Kmeans_initial(PC_list,nk_list,tau_C,q0)
    
    
    #开始进行K_means迭代
    PR_iter = PR_ini
    PC_iter = PC_ini
    for t in range(T):
        PR_selected,k_selected_R = PA_selection(PR_list,PR_iter,tau_R)
        PC_selected,k_selected_C = PA_selection(PC_list,PC_iter,tau_C)
        PR_iter = GB(PR_selected,np.array(nk_list)[k_selected_R],p0)
        PC_iter = GB(PC_selected,np.array(nk_list)[k_selected_C],q0)
        PR_list = PR_selected
        PC_list = PC_selected
    
    
    #输出k_means下的对于R和C的估计有用的数据集指标
    return k_selected_R,k_selected_C

def optional_finetune(idx_target,y_target,hatU,hatV,p,q,ri,delta_U,delta_V):
    m0_tilde = RCGD(idx_target,y_target,p,q,ri,eta=0.5)#使用RCGD函数针对target数据集跑出一个hatU_tar( hatP_tar=hatU_tar hatU_tar' )和hatV_tar
    hatP_U = np.dot(hatU,hatU.T)
    hatP_V = np.dot(hatV,hatV.T)
    #target的m0_tilde的左右奇异空间
    U_full,Lambda,V_full = np.linalg.svd(m0_tilde)
    U_target = U_full[:,:ri]
    V_target = V_full.T[:,:ri]
    PU_target = np.dot(U_target,U_target.T)
    PV_target = np.dot(V_target,V_target.T)
    #m0_tilde的左右奇异空间距离和hatP_U、hatP_V空间距离差异
    dis_U = np.trace(PU_target@hatP_U)
    dis_V = np.trace(PV_target@hatP_V)
    if dis_U > delta_U and dis_V > delta_V:
        #print("此时target的左右奇异空间和R和C估计量相似，能由source提供更多有用信息")
        m0_feature_finetune = feature_finetune(hatU,hatV,idx_target,y_target)
        return m0_feature_finetune
    elif dis_U > delta_U or dis_V > delta_V:
        if dis_U > delta_U:
            #print("此时target的左奇异空间和R的估计量相似，能由source提供更多有用信息")
            m0_feature_finetune = feature_finetune(hatU,V_target,idx_target,y_target)
            return m0_feature_finetune
        else:
            #print("此时target的右奇异空间和C的估计量相似，能由source提供更多有用信息")
            m0_feature_finetune = feature_finetune(U_target,hatV,idx_target,y_target)
            return m0_feature_finetune
    else:
        #print("此时target的左右奇异空间和R、C的估计量均不相似，target stduy不能由source提供更多有用信息")
        m0_feature_finetune = feature_finetune(U_target,V_target,idx_target,y_target)
        return m0_feature_finetune

