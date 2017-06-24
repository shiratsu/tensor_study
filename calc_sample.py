import tensorflow as tf

# 定数で1 + 2
x = tf.constant(1, name='x')
y = tf.constant(2, name='y')
z = x * y + y

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(z)
    # SummaryWriterでグラフを書く
    summary_writer = tf.summary.FileWriter('data', graph=sess.graph)
    tf.summary.scalar('calc_sample', z)
