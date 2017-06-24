import tensorflow as tf

# 足し算
with tf.name_scope('add_scope'):
    x = tf.constant(1, name='x')
    y = tf.constant(2, name='y')
    z = x + y

# 上の結果に掛け算
with tf.name_scope('multiply_scope'):
    zz = y * z

with tf.Session() as sess:
    with tf.name_scope('init_scope'):
        sess.run(tf.global_variables_initializer())
    sess.run(zz)
    # グラフを書こう
    summary_writer = tf.summary.FileWriter('data', graph=sess.graph)
    tf.summary.scalar('calc_sample3', zz)
