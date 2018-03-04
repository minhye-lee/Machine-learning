import tensorflow as tf
import numpy as np

def load_file():
    mnist_test = np.loadtxt('fashion-mnist_test.csv', delimiter=',', dtype=np.float32)
    mnist_test_x = mnist_test[:, 1:]
    mnist_test_y = mnist_test[:, [0]]

    # In[26]:

    mnist_train = np.loadtxt('fashion-mnist_train.csv', delimiter=',', dtype=np.float32)
    mnist_train_x = mnist_train[:, 1:]
    mnist_train_y = mnist_train[:, [0]]

    # In[28]:

    print(mnist_test_x.shape, mnist_test_y.shape)
    print(mnist_test_x.shape[0])
    print(mnist_train_x.shape, mnist_train_y.shape)

    return [mnist_test_x, mnist_test_y, mnist_train_x, mnist_train_y]


def learning(training_epochs, batch_size, mnist_train_x, mnist_train_y, cost, optimizer, X, Y):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist_train_x.shape[0] / batch_size)

        for i in range(total_batch):
            batch_xs = mnist_train_x[i * 100: i * 100 + (batch_size)]
            batch_ys = mnist_train_y[i * 100: i * 100 + (batch_size)]
            feed_dict = {X: batch_xs, Y: batch_ys}
            c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost = ', '{:.9f}'.format(avg_cost))
    print('Learning Finished!')
    return (sess)

def accuracy(hypothesis, Y_one_hot, mnist_test_x, mnist_test_y, sess, X, Y):
    prediction = tf.argmax(hypothesis, 1)
    correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accuracy : ', sess.run(accuracy, feed_dict={X: mnist_test_x, Y: mnist_test_y}))
    return (prediction)

def details(sess, prediction, X, mnist_test_x, mnist_test_y):
    pred = sess.run(prediction, feed_dict={X: mnist_test_x})
    # y_data: (N,1) = flatten => (N, ) matches pred.shape
    for p, y in zip(pred, mnist_test_y.flatten()):
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))

