import numpy as np
import tensorflow as tf

def d_blk_dct(input, Q):

    a = np.cos(np.pi / 4)   /2
    b = np.cos(np.pi / 16)  /2
    c = np.cos(np.pi / 8)   /2
    d = np.cos(np.pi * 3/16)/2
    e = np.cos(np.pi * 5/16)/2
    f = np.cos(np.pi * 3/8) /2
    g = np.cos(np.pi * 7/16)/2

    A_np = np.array(
        [[  a,  a,  a,  a,  a,  a,  a,  a],
         [  b,  d,  e,  g, -g, -e, -d, -b],
         [  c,  f, -f, -c, -c, -f,  f,  c],
         [  d, -g, -b, -e,  e,  b,  g, -d],
         [  a, -a, -a,  a,  a, -a, -a,  a],
         [  e, -b,  g,  d, -d, -g,  b, -e],
         [  f, -c,  c, -f, -f,  c, -c,  f],
         [  g, -e,  d, -b,  b, -d,  e, -g]],
        dtype = np.float32)
    A_np_t = A_np.transpose()
    A_np = tf.convert_to_tensor(value = A_np, dtype = tf.float32)
    A_np_t = tf.convert_to_tensor(value = A_np_t, dtype = tf.float32)

    input_shape_list = input.shape.as_list()
    BLK_ROW = int(input_shape_list[1] / 8)
    BLK_COL = int(input_shape_list[2] / 8)

    total_q = np.zeros(
        shape = (1, input_shape_list[1], input_shape_list[2], 1),
        dtype = np.float32)#tf.zeros_like(input)
    for blk_row_idx in range(BLK_ROW):
        for blk_col_idx in range(BLK_COL):
            row_beg = blk_row_idx * 8
            col_beg = blk_col_idx * 8
            row_end = row_beg + 8
            col_end = col_beg + 8
            total_q[0, row_beg:row_end, col_beg:col_end, 0] = Q
    total_q = tf.convert_to_tensor(
        value = total_q,
        dtype = tf.float32)
    inputa = tf.multiply(
        x = input,
        y = total_q)

    output = tf.zeros([])
    for blk_row_idx in range(BLK_ROW):
        tmpr = tf.zeros([])
        for blk_col_idx in range(BLK_COL):
            row_beg = blk_row_idx * 8
            col_beg = blk_col_idx * 8
            row_end = row_beg + 8
            col_end = col_beg + 8
            s_blk = inputa[0, row_beg:row_end, col_beg:col_end, 0]
            tmp = A_np_t @ s_blk @ A_np
            if blk_col_idx == 0:
                tmpr = tmp
            else:
                tmpr = tf.concat([tmpr, tmp], axis = 1)
        if blk_row_idx == 0:
            output = tmpr
        else:
            output = tf.concat([output, tmpr], axis = 0)

    outa = tf.reshape(output, (1, input_shape_list[1], input_shape_list[2], 1))
    dc = tf.fill((1, input_shape_list[1], input_shape_list[2], 1), value = 128.0)
    outa = outa + dc

    return outa
################################################################################