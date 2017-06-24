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

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    batch_images, batch_labels = mnist.train.next_batch(1000)
    sess.run(train_step, feed_dict={images:batch_images, labels:batch_labels})
    summary_writer = tf.summary.FileWriter('data', graph=sess.graph)
    tf.summary.scalar('cross_entropy', cross_entropy)
