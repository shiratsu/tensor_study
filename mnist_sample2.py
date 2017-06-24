import tensorflow as tf

# mnist手書きデータの用意
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# softmaxでgradient discentする簡易な記述
images = tf.placeholder(tf.float32, shape=[None, 784])
labels = tf.placeholder(tf.float32, shape=[None, 10])
weights = tf.Variable(tf.zeros([784,10]))
softmax = tf.nn.softmax(tf.matmul(images, weights))
cross_entropy = -tf.reduce_sum(labels * tf.log(softmax))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# sessionの用意
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# summaryの設定
tf.summary.scalar('cross_entropy', cross_entropy)
summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('data', graph=sess.graph)

# 100回実行してcross_entropyのsummaryを記録
for step in range(100):
    batch_images, batch_labels = mnist.train.next_batch(100)
    feed_dict = {images:batch_images, labels:batch_labels}
    sess.run([train_step, cross_entropy], feed_dict=feed_dict)
    summary_str = sess.run(summary_op, feed_dict=feed_dict)
    summary_writer.add_summary(summary_str, step)
