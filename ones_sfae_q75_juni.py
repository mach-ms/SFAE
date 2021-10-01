import tensorflow as tf
import scipy.io as sio
import numpy as np

import time
import sys
import os

import pm1_simulator as sim
import tf_dct_module as tdm
import def_extractor

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
tf_config = tf.ConfigProto()  

ITVL_NUM = 10
MU = 1.5
QF = 75
WET_COST = float(1e13)

#这里需要指定| here should set the parameter to run the program
PAY_RATES = [0.05, 0.1, 0.2, 0.3, 0.4]#相对负载率| payload rate
QTBL_MAT75_NAME = '' #存放jpg量化表的mat文件名| the file name of .mat to save the jpeg quant table
COV_ROOT = '' #存放cover 的jpeg dct系数的mat文件夹路径| the name of dir to save the cover's jpeg dcts 
JUNI_RHO_ROOT = '' #存放cover 的juniward 代价的mat文件夹路径| the name of dir to save the cover's juniward distortions
JUNI_STE_ROOTS = [
    '',
    '',
    '',
    '',
    '']#输出stego 的jpeg dct系数的mat文件夹路径| the names of dir to save the stego's jpeg dcts 
#上面PAY_RATES共有5个，即5个负载率，即输出五个文件夹| 5 payload rates, so there are 5 dirs 

#==============================================================================#
# 计算参数| compute parameter w and b
def get_w_b(c_fea, s_fea):
    w_tmp = c_fea - s_fea
    mid = (c_fea + s_fea) / 2.0
    b_tmp = -1 * mid @ w_tmp.transpose()
    w_tmp = np.reshape(w_tmp, (8000, ))
    w = np.zeros((2, 8000), dtype = np.float32)
    w[0, :] = w_tmp #.transpose()
    w[1, :] = -1 * w_tmp #.transpose()
    b = np.zeros((2, 1), dtype = np.float32)
    b[0, 0] = b_tmp
    b[1, 0] = -1 * b_tmp
    return w, b
#==============================================================================#

#==============================================================================#
# 整理原始代价 由于juni的数值过大可能会造成exp溢出| scale the rho, to avoid excceed
def deal_rho(x, rho_t):
    rho_p1 = rho_t.copy() / 24.0 / 24.0
    rho_p1[rho_t >= WET_COST] = WET_COST
    rho_p1[x > 1024] = WET_COST
    rho_m1 = rho_t.copy() / 24.0 / 24.0
    rho_m1[rho_t >= WET_COST] = WET_COST
    rho_m1[x < -1023] = WET_COST
    return rho_p1, rho_m1
#==============================================================================#

#==============================================================================#
# 根据梯度调整代价| adjust rho according to gradient
def partial_adjust_rhos(rho_p1, rho_m1, adv_grad, krate, seed):
    n = int(np.size(rho_p1))
    ori_shape = rho_p1.shape
    np.random.seed(seed)
    perm_order = np.random.permutation(n)
    op_ord = perm_order[0:int(np.round(krate * float(n)))]

    adv_grad = adv_grad.reshape((n,))
    op_grd = adv_grad[op_ord]

    adj_rho_p1 = rho_p1.copy()
    adj_rho_p1 = adj_rho_p1.reshape((n,))
    op_arho_p1 = adj_rho_p1[op_ord]
    op_arho_p1[op_grd > 0] = op_arho_p1[op_grd > 0] * MU
    op_arho_p1[op_grd < 0] = op_arho_p1[op_grd < 0] / MU
    adj_rho_p1[op_ord] = op_arho_p1
    adj_rho_p1 = adj_rho_p1.reshape(ori_shape)

    adj_rho_m1 = rho_m1.copy()
    adj_rho_m1 = adj_rho_m1.reshape((n,))
    op_arho_m1 = adj_rho_m1[op_ord]
    op_arho_m1[op_grd > 0] = op_arho_m1[op_grd > 0] / MU
    op_arho_m1[op_grd < 0] = op_arho_m1[op_grd < 0] * MU
    adj_rho_m1[op_ord] = op_arho_m1
    adj_rho_m1 = adj_rho_m1.reshape(ori_shape)

    return adj_rho_p1, adj_rho_m1
#==============================================================================#

def batch_gen(
        cov_root = '',
        ste_root = '',
        rho_root = '',
        pay_rate = 0.0,
        itvl_num = ITVL_NUM,
        config = tf_config):

    beg_t = time.time()
    ################################################################################
    # 得到图像批次的文件名，以及jpeg量化表| get the name list of covers and jpeg quant table
    name_list = os.listdir(cov_root)
    q_tbl = sio.loadmat(QTBL_MAT75_NAME)['qtbl']
    q_tbl = np.asarray(q_tbl, dtype = np.float32)
    ################################################################################

    ################################################################################
    # 计算图定义| definition of tf graph
    #==========================================
    # 定义特征提取器
    ph_dct = tf.placeholder(
        shape = (1, 256, 256, 1), 
        dtype = tf.float32, 
        name = 'ph_dct')
    fea_nndctr = def_extractor.xxxx_extractor(tdm.d_blk_dct(ph_dct, q_tbl), QF)
    #==========================================
    #==========================================
    # 定义整体分类器(线性映射器)| definition of linear mapper
    ph_w = tf.placeholder(
        shape = (2, 8000), 
        dtype = tf.float32, 
        name = 'ph_w')
    ph_b = tf.placeholder(
        shape = (2, 1), 
        dtype = tf.float32, 
        name = 'ph_b')
    logits = tf.reduce_mean(tf.transpose(ph_w @ tf.transpose(fea_nndctr) + ph_b), 0)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels = 0, 
        logits = logits))
    grad = tf.gradients(loss, ph_dct)
    #==========================================
    ################################################################################

    ################################################################################
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        dct_buf = np.empty(shape = (1, 256, 256, 1), dtype = np.float32)

        img_num = len(name_list)
        for img_idx in range(img_num):
            mat_name = name_list[img_idx]
            npy_name = mat_name.replace('mat', '') + 'npy'
            cov_name = cov_root + os.sep + mat_name
            rho_name = rho_root + os.sep + mat_name
            ste_name = ste_root + os.sep + mat_name
            dis_name = ste_root + os.sep + npy_name

           # 加载cover| load cover
            cov = sio.loadmat(cov_name)['coef']
            cov = np.asarray(cov, dtype = np.int32)

            # 计算cover的feature| compute cover feature
            dct_buf[0, :, :, 0] = cov.astype(np.float32)
            result = sess.run(
                fetches = fea_nndctr, 
                feed_dict = {ph_dct: dct_buf})
            c_fea = np.reshape(result[0], (1, 8000))
            
            # 加载原始代价| load juniward rho
            rho_t = sio.loadmat(rho_name)['rho']
            rho_p1, rho_m1 = deal_rho(cov, rho_t)
            
            # 计算位置| compute the position to change
            pos_nzac = sim.pos_nzac(cov)

            # 生成原始自适应隐写图像| generate the common stego
            ste = sim.embed_all(cov, rho_p1, rho_m1, pos_nzac, pay_rate, img_idx)

            # 计算stego的feature| compute the feature of stego
            dct_buf[0, :, :, 0] = ste.astype(np.float32)
            result = sess.run(
                fetches = fea_nndctr, 
                feed_dict = {ph_dct: dct_buf})
            s_fea = np.reshape(result[0], (1, 8000))

            # 计算linear mapper及对抗梯度| compute the w and b in linear mapper, and the adversarial gradient
            w, b = get_w_b(c_fea, s_fea)
            result = sess.run(
                fetches = grad, 
                feed_dict = {ph_dct: dct_buf, ph_w: w, ph_b: b})
            adv_grad = np.reshape(result[0], (256, 256))

            # 生成不同的stego| generate different stegos
            ste_img_candi = np.empty(
                shape = (itvl_num - 1, cov.shape[0], cov.shape[1]), 
                dtype = cov.dtype)
            dists = np.empty(shape = (itvl_num - 1, ), dtype = c_fea.dtype)
            
            kstep = 1.0 / float(itvl_num)
            for k_idx in range(itvl_num - 1):
                krate = kstep * (k_idx + 1)
                
                adj_rho_p1, adj_rho_m1 = partial_adjust_rhos(
                    rho_p1      = rho_p1, 
                    rho_m1      = rho_m1, 
                    adv_grad    = adv_grad, 
                    krate       = krate, 
                    seed        = img_idx + 1)
                ste = sim.embed_all(
                    cov         = cov, 
                    rho_p1      = adj_rho_p1, 
                    rho_m1      = adj_rho_m1, 
                    pos         = pos_nzac, 
                    prate       = pay_rate, 
                    seed        = img_idx)
                ste_img_candi[k_idx, :, :] = ste

                # 计算stego的feature| compute the feature of stego
                dct_buf[0, :, :, 0] = ste.astype(np.float32)
                result = sess.run(
                    fetches = fea_nndctr, 
                    feed_dict = {ph_dct: dct_buf})
                s_fea = np.reshape(result[0], (1, 8000))
                dists[k_idx] = np.linalg.norm(c_fea - s_fea)

            kmin = dists.argmin(axis = 0)
            ste = ste_img_candi[kmin, :, :]
            sio.savemat(ste_name, {'coef', ste})
            np.save(dis_name, dists)
    end_t = time.time()
    print(ste_root + ' :' + str(end_t - beg_t))
    ################################################################################

if __name__ == '__main__':
    juni_num = len(JUNI_STE_ROOTS)
    for juni_idx in range(juni_num):
        batch_gen(
            cov_root = COV_ROOT,
            ste_root = JUNI_STE_ROOTS[juni_idx],
            rho_root = JUNI_RHO_ROOT,
            pay_rate = PAY_RATES[juni_idx])