import numpy as np
import warnings
warnings.filterwarnings('ignore')

def _ternary_entropyf(p_p1, p_m1):
    Ht = -(p_p1 * np.log2(p_p1)) \
         -(p_m1 * np.log2(p_m1)) \
         -((1 - p_p1 - p_m1) * np.log2(1 - p_p1 - p_m1))
    Ht[np.isnan(Ht)] = 0
    Ht = np.sum(Ht)
    return Ht

def _calc_lambda(rho_p1, rho_m1, message_length, n):
    l3 = 1e+3
    m3 = float(message_length + 1)
    iterations = 0
    while m3 > message_length:
        l3 = l3 * 2
        p_p1 = (np.exp(-l3 * rho_p1))/(1 + np.exp(-l3 * rho_p1) + np.exp(-l3 * rho_m1))
        p_m1 = (np.exp(-l3 * rho_m1))/(1 + np.exp(-l3 * rho_p1) + np.exp(-l3 * rho_m1))
        m3 = _ternary_entropyf(p_p1, p_m1)
        iterations = iterations + 1
        if (iterations > 10):
            lamb = l3
            return lamb
    l1 = 0
    m1 = float(n)      
    lamb = 0
    alpha = float(message_length)/n    
    while  (float(m1-m3)/n > alpha/1000.0 ) and (iterations<30):
        lamb = l1+(l3-l1)/2
        p_p1 = (np.exp(-lamb * rho_p1))/(1 + np.exp(-lamb * rho_p1) + np.exp(-lamb * rho_m1))
        p_m1 = (np.exp(-lamb * rho_m1))/(1 + np.exp(-lamb * rho_p1) + np.exp(-lamb * rho_m1))
        m2 = _ternary_entropyf(p_p1, p_m1)
        if m2 < message_length:
            l3 = lamb
            m3 = m2
        else:
            l1 = lamb
            m1 = m2
        iterations = iterations + 1
    return lamb

def _embed(x, rho_p1, rho_m1, m, seed):
    #print('in')
    n = np.size(x)
    lamb = _calc_lambda(rho_p1, rho_m1, m, n)
    pChangeP1 = (np.exp(-lamb * rho_p1))/(1 + np.exp(-lamb * rho_p1) + np.exp(-lamb * rho_m1))
    pChangeM1 = (np.exp(-lamb * rho_m1))/(1 + np.exp(-lamb * rho_p1) + np.exp(-lamb * rho_m1))
    np.random.seed(seed)
    randChange = np.random.rand(x.size)
    y = x.copy()
    pos_p1 = randChange < pChangeP1
    y[pos_p1] = y[pos_p1] + 1
    pos_m1 = (randChange >= pChangeP1) & (randChange < pChangeP1+pChangeM1)
    y[pos_m1] = y[pos_m1] - 1
    return y

def pos_nzac(x):
    pos = np.ones(x.shape, x.dtype)
    pos[::8, ::8] = 0
    pos[x == 0] = 0
    return pos

def embed_part1(cov, rho_p1, rho_m1, pos, prate, krate, seed):
    ori_shape = cov.shape
    n = int(np.size(cov))
    ste = cov.copy()
    cov = cov.reshape((n,))
    ste = ste.reshape((n,))
    pos = pos.reshape((n,))
    rho_m1 = rho_m1.reshape((n,))
    rho_p1 = rho_p1.reshape((n,))
    
    np.random.seed(seed)
    perm_order = np.random.permutation(n)

    op_n = int(np.round(krate * float(n)))
    op_ord = perm_order[0:op_n]
    op_cov = cov[op_ord]
    op_pos = pos[op_ord]
    op_rho_m1 = rho_m1[op_ord]
    op_rho_p1 = rho_p1[op_ord]
    #print('in op')
    op_nzac = len(op_pos.nonzero()[0])
    op_m = int(np.round(prate * float(op_nzac)))
    op_ste = _embed(op_cov, op_rho_p1, op_rho_m1, op_m, seed + 1)

    ste[op_ord] = op_ste
    #cov = cov.reshape(ori_shape)
    ste = ste.reshape(ori_shape)
    #pos = pos.reshape(ori_shape)
    #rho_m1 = rho_m1.reshape(ori_shape)
    #rho_p1 = rho_p1.reshape(ori_shape)

    return ste

def embed_part2(cov, rho_p1, rho_m1, pos, prate, krate, seed):
    ori_shape = cov.shape
    n = int(np.size(cov))
    ste = cov.copy()
    cov = cov.reshape((n,))
    ste = ste.reshape((n,))
    pos = pos.reshape((n,))
    rho_m1 = rho_m1.reshape((n,))
    rho_p1 = rho_p1.reshape((n,))
    
    np.random.seed(seed)
    perm_order = np.random.permutation(n)

    op_n1 = int(np.round(krate * float(n)))
    op_ord = perm_order[op_n1:n]
    op_cov = cov[op_ord]
    op_pos = pos[op_ord]
    op_rho_m1 = rho_m1[op_ord]
    op_rho_p1 = rho_p1[op_ord]
    op_nzac = len(op_pos.nonzero()[0])
    op_m = int(np.round(prate * float(op_nzac)))
    op_ste = _embed(op_cov, op_rho_p1, op_rho_m1, op_m, seed + 1)

    ste[op_ord] = op_ste
    #cov = cov.reshape(ori_shape)
    ste = ste.reshape(ori_shape)
    #pos = pos.reshape(ori_shape)
    #rho_m1 = rho_m1.reshape(ori_shape)
    #rho_p1 = rho_p1.reshape(ori_shape)

    return ste

def embed_all(cov, rho_p1, rho_m1, pos, prate, seed):
    ori_shape = cov.shape
    n = int(np.size(cov))
    ste = cov.copy()
    cov = cov.reshape((n,))
    ste = ste.reshape((n,))
    pos = pos.reshape((n,))
    rho_m1 = rho_m1.reshape((n,))
    rho_p1 = rho_p1.reshape((n,))

    nzac = len(pos.nonzero()[0])
    m = int(np.round(prate * float(nzac)))
    ste = _embed(cov, rho_p1, rho_m1, m, seed)

    #cov = cov.reshape(ori_shape)
    ste = ste.reshape(ori_shape)
    #pos = pos.reshape(ori_shape)
    #rho_m1 = rho_m1.reshape(ori_shape)
    #rho_p1 = rho_p1.reshape(ori_shape)

    return ste