import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import model
import time

model_dir = './lenet-5/'
mnist = input_data.read_data_sets("data/fashion", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
image = tf.reshape(x, [-1, 28, 28, 1])
keep_prob = tf.placeholder(tf.float32)
y, intermediate, w_fc2, w_fc3 = model.lenet_5(image, keep_prob)
predicted = tf.argmax(y, 1)
label = tf.argmax(y_, 1)
correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

saver = tf.train.Saver()
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))
    print('restore succeed.')

    samples = []
    neg_samples = []
    # pred = []
    ground = []
    neg_predict = []
    neg_ground = []
    # fc2 = []
    # fc3 = []
    equal = 0
    flag = True

    for i in range(55000):
        images = mnist.train.images[i:i+1, :]
        labels = mnist.train.labels[i:i+1, :]
        feed_dict = {x: images, y_: labels, keep_prob: 1.0}
        intermediateValues, predictedNp, labelNp, _, _ = sess.run([intermediate, predicted, label, w_fc2, w_fc3], feed_dict=feed_dict)
        if predictedNp == labelNp:
            equal += 1
            samples.extend(intermediateValues)
            ground.extend(labelNp)
        # if flag:
        #     fc2.extend(weight_fc2)
            # fc3.extend(weight_fc3)
            # flag = False
        else:
            neg_samples.extend(intermediateValues)
            neg_predict.extend(predictedNp)
            neg_ground.extend(labelNp)

    for i in range(5000):
        images = mnist.validation.images[i:i+1, :]
        labels = mnist.validation.labels[i:i+1, :]
        feed_dict = {x: images, y_: labels, keep_prob: 1.0}
        intermediateValues, predictedNp, labelNp, _, _ = sess.run([intermediate, predicted, label, w_fc2, w_fc3], feed_dict=feed_dict)
        if predictedNp == labelNp:
            equal += 1
            samples.extend(intermediateValues)
            ground.extend(labelNp)
        # if flag:
        #     fc2.extend(weight_fc2)
            # fc3.extend(weight_fc3)
            # flag = False
        else:
            neg_samples.extend(intermediateValues)
            neg_predict.extend(predictedNp)
            neg_ground.extend(labelNp)

    samples = np.array(samples)
    ground = np.array(ground)
    print(samples.shape, ground.shape)
    print(equal)
    np.save('training_set_neuron_outputs', samples)
    np.save('training_set_labels', ground)
    # np.save('weights_fc2', fc2)
    # np.save('weights_fc3', fc3)
    np.save('training_set_error_outputs', neg_samples)
    np.save('training_set_error_predicts', neg_predict)
    np.save('training_set_error_labels', neg_ground)
    samples_test = []
    pred_test = []
    label_test = []
    equal = 0
    none = 0
    for i in range(10000):
        images = mnist.test.images[i:i+1, :]
        labels = mnist.test.labels[i:i+1, :]
        feed_dict = {x: images, y_: labels, keep_prob: 1.0}
        intermediateValues, predictedNp, labelNp, = sess.run([intermediate, predicted, label], feed_dict=feed_dict)
        if predictedNp == labelNp:
            equal += 1
        else:
            none += 1
        samples_test.extend(intermediateValues)
        pred_test.extend(predictedNp)
        label_test.extend(labelNp)
    samples_test = np.array(samples_test)
    pred_test = np.array(pred_test)
    label_test = np.array(label_test)

    print(equal, none)
    print(samples_test.shape, pred_test.shape, label_test.shape)
    np.save('test_set_neuron_outputs', samples_test)
    np.save('test_set_predictions', pred_test)
    np.save('test_set_labels', label_test)

    start_time = time.time()
    color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
             '#17becf']
    pca = PCA(n_components=2)
    pca.fit(samples)
    samples_reduction = pca.transform(samples)
    plt.figure(1)
    for i in range(10):
        print(samples_reduction[ground == i, 0].shape)
        plt.scatter(samples_reduction[ground == i, 0], samples_reduction[ground == i, 1], c=color[i], marker='o', s=2, linewidths=0, alpha=0.8, label='%s' % i)

    tsne = TSNE(n_components=2)
    tsne_reduction = tsne.fit_transform(samples)
    plt.figure(2)
    for i in range(10):
        print(tsne_reduction[ground == i, 0].shape)
        plt.scatter(tsne_reduction[ground == i, 0], tsne_reduction[ground == i, 1], c=color[i], marker='o', s=2, linewidths=0, alpha=0.8, label='%s' % i)
    plt.show()
    print(time.time()-start_time)
