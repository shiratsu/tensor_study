import tensorflow as tf

for i in range(4):
    with tf.variable_scope('scope-{}'.format(i)):
        for j in range(25):
             v = tf.Variable(1, name=str(j))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(v)
    # SummaryWriterでグラフを書く
    summary_writer = tf.summary.FileWriter('data', graph=sess.graph)
    tf.summary.scalar('one_plus_one_summary', v)
