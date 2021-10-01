#import os
import numpy as np
import tensorflow as tf
#import keras


#XXXX_DIM = 8000

################################################################################
def _fill_filters():
    a = np.asarray(range(8), dtype = np.float32)
    [k, l] = np.meshgrid(a, a)
    A = 0.5 * np.cos((np.multiply((2*k + 1), l*np.pi))/16)
    A[0, :] = np.divide(A[0, :], np.sqrt(2))
    A = np.transpose(A)
    np_kernel = np.zeros((8, 8, 1, 64), dtype = np.float32)
    for mode_r in range(8):
        for mode_c in range(8):
            mode_idx = mode_r*8 + mode_c
            A_r = np.reshape(A[:, mode_r], (8, 1))
            A_c = np.reshape(A[:, mode_c], (8, 1))
            np_kernel[:, :, 0, mode_idx] = \
                np.matmul(A_r, np.transpose(A_c))
    kernel_filler = tf.convert_to_tensor(np_kernel, tf.float32)
    return kernel_filler
################################################################################

################################################################################
def _get_q(qf):
    if qf < 50.0:
        q = min(8.0*(50.0 / qf), 100.0)
    else:
        q = max(8.0*(2.0 - (qf/50.0)), 0.2)
    return q
################################################################################

################################################################################
def _split_phase(input):
    # please ensure that height and width can be divided by 8
    phase00 = input[:, 0::8, 0::8, :]
    phase01 = input[:, 0::8, 1::8, :]
    phase02 = input[:, 0::8, 2::8, :]
    phase03 = input[:, 0::8, 3::8, :]
    phase04 = input[:, 0::8, 4::8, :]
    phase05 = input[:, 0::8, 5::8, :]
    phase06 = input[:, 0::8, 6::8, :]
    phase07 = input[:, 0::8, 7::8, :]
    phase10 = input[:, 1::8, 0::8, :]
    phase11 = input[:, 1::8, 1::8, :]
    phase12 = input[:, 1::8, 2::8, :]
    phase13 = input[:, 1::8, 3::8, :]
    phase14 = input[:, 1::8, 4::8, :]
    phase15 = input[:, 1::8, 5::8, :]
    phase16 = input[:, 1::8, 6::8, :]
    phase17 = input[:, 1::8, 7::8, :]
    phase20 = input[:, 2::8, 0::8, :]
    phase21 = input[:, 2::8, 1::8, :]
    phase22 = input[:, 2::8, 2::8, :]
    phase23 = input[:, 2::8, 3::8, :]
    phase24 = input[:, 2::8, 4::8, :]
    phase25 = input[:, 2::8, 5::8, :]
    phase26 = input[:, 2::8, 6::8, :]
    phase27 = input[:, 2::8, 7::8, :]
    phase30 = input[:, 3::8, 0::8, :]
    phase31 = input[:, 3::8, 1::8, :]
    phase32 = input[:, 3::8, 2::8, :]
    phase33 = input[:, 3::8, 3::8, :]
    phase34 = input[:, 3::8, 4::8, :]
    phase35 = input[:, 3::8, 5::8, :]
    phase36 = input[:, 3::8, 6::8, :]
    phase37 = input[:, 3::8, 7::8, :]
    phase40 = input[:, 4::8, 0::8, :]
    phase41 = input[:, 4::8, 1::8, :]
    phase42 = input[:, 4::8, 2::8, :]
    phase43 = input[:, 4::8, 3::8, :]
    phase44 = input[:, 4::8, 4::8, :]
    phase45 = input[:, 4::8, 5::8, :]
    phase46 = input[:, 4::8, 6::8, :]
    phase47 = input[:, 4::8, 7::8, :]
    phase50 = input[:, 5::8, 0::8, :]
    phase51 = input[:, 5::8, 1::8, :]
    phase52 = input[:, 5::8, 2::8, :]
    phase53 = input[:, 5::8, 3::8, :]
    phase54 = input[:, 5::8, 4::8, :]
    phase55 = input[:, 5::8, 5::8, :]
    phase56 = input[:, 5::8, 6::8, :]
    phase57 = input[:, 5::8, 7::8, :]
    phase60 = input[:, 6::8, 0::8, :]
    phase61 = input[:, 6::8, 1::8, :]
    phase62 = input[:, 6::8, 2::8, :]
    phase63 = input[:, 6::8, 3::8, :]
    phase64 = input[:, 6::8, 4::8, :]
    phase65 = input[:, 6::8, 5::8, :]
    phase66 = input[:, 6::8, 6::8, :]
    phase67 = input[:, 6::8, 7::8, :]
    phase70 = input[:, 7::8, 0::8, :]
    phase71 = input[:, 7::8, 1::8, :]
    phase72 = input[:, 7::8, 2::8, :]
    phase73 = input[:, 7::8, 3::8, :]
    phase74 = input[:, 7::8, 4::8, :]
    phase75 = input[:, 7::8, 5::8, :]
    phase76 = input[:, 7::8, 6::8, :]
    phase77 = input[:, 7::8, 7::8, :]

    output = tf.concat(
        values = [
            phase00,
            phase01, phase07,
            phase02, phase06,
            phase03, phase05,
            phase04,
            phase10, phase70,
            phase11, phase17, phase71, phase77, 
            phase12, phase16, phase72, phase76,
            phase13, phase15, phase73, phase75,
            phase14, phase74,
            phase20, phase60,
            phase21, phase27, phase61, phase67, 
            phase22, phase26, phase62, phase66,
            phase23, phase25, phase63, phase65,
            phase24, phase64,
            phase30, phase50,
            phase31, phase37, phase51, phase57, 
            phase32, phase36, phase52, phase56,
            phase33, phase35, phase53, phase55,
            phase34, phase54,
            phase40,
            phase41, phase47, 
            phase42, phase46,
            phase43, phase45,
            phase44
        ],
        axis = 2
    )

    return output
################################################################################

################################################################################
def _gauss_hist(input): # input is a 4-D tensor
    input_shape = input.shape.as_list()
    elem_num = tf.fill(
        [input_shape[0], 1, 1, input_shape[3]], 
        float(input_shape[1] * input_shape[2])
    )
    dev = tf.fill(input_shape, 0.6)
    minus  = tf.fill(input_shape, -1.0)

    std1 = tf.divide(tf.subtract(input, tf.fill(input_shape, 0.0)), dev) 
    gauss1 = tf.exp(tf.multiply(tf.square(std1), minus))
    bin1 = tf.divide(tf.reduce_sum(gauss1, [1, 2], keepdims = True), elem_num)
    
    std2 = tf.divide(tf.subtract(input, tf.fill(input_shape, 1.0)), dev)
    gauss2 = tf.exp(tf.multiply(tf.square(std2), minus))
    bin2 = tf.divide(tf.reduce_sum(gauss2, [1, 2], keepdims = True), elem_num)
    
    std3 = tf.divide(tf.subtract(input, tf.fill(input_shape, 2.0)), dev)
    gauss3 = tf.exp(tf.multiply(tf.square(std3), minus))
    bin3 = tf.divide(tf.reduce_sum(gauss3, [1, 2], keepdims = True), elem_num)
    
    std4 = tf.divide(tf.subtract(input, tf.fill(input_shape, 3.0)), dev)
    gauss4 = tf.exp(tf.multiply(tf.square(std4), minus))
    bin4 = tf.divide(tf.reduce_sum(gauss4, [1, 2], keepdims = True), elem_num)
    
    std5 = tf.divide(tf.subtract(input, tf.fill(input_shape, 4.0)), dev)
    gauss5 = tf.exp(tf.multiply(tf.square(std5), minus))
    bin5 = tf.divide(tf.reduce_sum(gauss5, [1, 2], keepdims = True), elem_num)

    output = tf.concat(
        values = [bin1, bin2, bin3, bin4, bin5],
        axis = 2
    )

    return output
################################################################################

################################################################################
def _merge_phase_2_hist(input): # input is residual
    input_shape = input.shape.as_list()
    w_phase = int(input_shape[2] / 8)

    merged_phase = _split_phase(input)
    
    mphist1  = _gauss_hist(merged_phase[:, :,  0*w_phase: 1*w_phase, :])
    mphist2  = _gauss_hist(merged_phase[:, :,  1*w_phase: 3*w_phase, :])
    mphist3  = _gauss_hist(merged_phase[:, :,  3*w_phase: 5*w_phase, :])
    mphist4  = _gauss_hist(merged_phase[:, :,  5*w_phase: 7*w_phase, :])
    mphist5  = _gauss_hist(merged_phase[:, :,  7*w_phase: 8*w_phase, :])

    mphist6  = _gauss_hist(merged_phase[:, :,  8*w_phase:10*w_phase, :])
    mphist7  = _gauss_hist(merged_phase[:, :, 10*w_phase:12*w_phase, :])
    mphist8  = _gauss_hist(merged_phase[:, :, 12*w_phase:16*w_phase, :])
    mphist9  = _gauss_hist(merged_phase[:, :, 16*w_phase:20*w_phase, :])
    mphist10 = _gauss_hist(merged_phase[:, :, 20*w_phase:22*w_phase, :])

    mphist11 = _gauss_hist(merged_phase[:, :, 22*w_phase:24*w_phase, :])
    mphist12 = _gauss_hist(merged_phase[:, :, 24*w_phase:28*w_phase, :])
    mphist13 = _gauss_hist(merged_phase[:, :, 28*w_phase:32*w_phase, :])
    mphist14 = _gauss_hist(merged_phase[:, :, 32*w_phase:36*w_phase, :])
    mphist15 = _gauss_hist(merged_phase[:, :, 36*w_phase:38*w_phase, :])

    mphist16 = _gauss_hist(merged_phase[:, :, 38*w_phase:40*w_phase, :])
    mphist17 = _gauss_hist(merged_phase[:, :, 40*w_phase:44*w_phase, :])
    mphist18 = _gauss_hist(merged_phase[:, :, 44*w_phase:48*w_phase, :])
    mphist19 = _gauss_hist(merged_phase[:, :, 48*w_phase:52*w_phase, :])
    mphist20 = _gauss_hist(merged_phase[:, :, 54*w_phase:56*w_phase, :])

    mphist21 = _gauss_hist(merged_phase[:, :, 56*w_phase:57*w_phase, :])
    mphist22 = _gauss_hist(merged_phase[:, :, 57*w_phase:59*w_phase, :])
    mphist23 = _gauss_hist(merged_phase[:, :, 59*w_phase:61*w_phase, :])
    mphist24 = _gauss_hist(merged_phase[:, :, 61*w_phase:62*w_phase, :])
    mphist25 = _gauss_hist(merged_phase[:, :, 62*w_phase:63*w_phase, :])

    output = output = tf.concat(
        values = [
            mphist1,  mphist2,  mphist3,  mphist4,  mphist5,
            mphist6,  mphist7,  mphist8,  mphist9,  mphist10,
            mphist11, mphist12, mphist13, mphist14, mphist15,
            mphist16, mphist17, mphist18, mphist19, mphist20,
            mphist21, mphist22, mphist23, mphist24, mphist25
        ],
        axis = 2
    )

    return output
################################################################################

################################################################################
def xxxx_extractor(input, qf):
    q = _get_q(qf)
    filters = _fill_filters()
    img_ac = tf.subtract(input, tf.fill(input.shape, 128.0))
    residual = tf.nn.conv2d(
        input = img_ac,
        filter = filters,
        strides=[1, 1, 1, 1],
        padding="SAME"
    )
    quant_factors = tf.fill(residual.shape, q)
    abs_residual = tf.abs(residual)
    quant = tf.divide(abs_residual, quant_factors)
    trunc_quant = tf.clip_by_value(quant, 0, 4)
    fea_out = tf.layers.flatten(_merge_phase_2_hist(trunc_quant))

    return fea_out
################################################################################
