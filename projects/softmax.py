
import tensorflow as tf

tf.set_random_seed(777)

def softmax(learning_rate): #softmax model

    nb_classes = 10
    X = tf.placeholder(tf.float32, [None, 784])
    Y = tf.placeholder(tf.int32, [None, 1])
    Y_one_hot = tf.one_hot(Y, nb_classes)

    print("one_hot", Y_one_hot)
    Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
    print("reshape", Y_one_hot)

    W = tf.Variable(tf.random_normal([784, nb_classes]), name='weight')
    b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

    logits = tf.matmul(X, W) + b
    hypothesis = tf.nn.softmax(logits)
    cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)

    cost = tf.reduce_mean(cost_i)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    return (X, Y, Y_one_hot, hypothesis, cost, optimizer)

def nn(learning_rate): #Neural net model
    nb_classes = 10

    X = tf.placeholder(tf.float32, [None, 784])
    Y = tf.placeholder(tf.int32, [None, 1])

    Y_one_hot = tf.one_hot(Y, nb_classes)
    print("one_hot", Y_one_hot)

    Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
    print("reshape", Y_one_hot)

    W1 = tf.Variable(tf.random_normal([784, 256]))
    b1 = tf.Variable(tf.random_normal([256]))
    L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

    W2 = tf.Variable(tf.random_normal([256, 256]))
    b2 = tf.Variable(tf.random_normal([256]))
    L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

    W3 = tf.Variable(tf.random_normal([256, 10]))
    b3 = tf.Variable(tf.random_normal([10]))

    hypothesis = tf.matmul(L2, W3) + b3

    # define cost/loss & optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=hypothesis, labels=Y_one_hot))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    return (X, Y, Y_one_hot, hypothesis, cost, optimizer)

def xavier(learning_rate): #Neural net Xavier model
    nb_classes = 10

    X = tf.placeholder(tf.float32, [None, 784])
    Y = tf.placeholder(tf.int32, [None, 1])

    Y_one_hot = tf.one_hot(Y, nb_classes)
    print("one_hot", Y_one_hot)

    Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
    print("reshape", Y_one_hot)

    W1 = tf.get_variable("W1", shape=[784, 256],
                         initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.Variable(tf.random_normal([256]))
    L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

    W2 = tf.get_variable("W2", shape=[256, 256],
                         initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.Variable(tf.random_normal([256]))
    L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

    W3 = tf.get_variable("W3", shape=[256, 10],
                         initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.Variable(tf.random_normal([10]))
    hypothesis = tf.matmul(L2, W3) + b3

    # In[15]:

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=hypothesis, labels=Y_one_hot))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    return (X, Y, Y_one_hot, hypothesis, cost, optimizer)

def deep_dropout(learning_rate): #Deep neural net model
    nb_classes = 10
    X = tf.placeholder(tf.float32, [None, 784])
    Y = tf.placeholder(tf.int32, [None, 1])

    Y_one_hot = tf.one_hot(Y, nb_classes)
    print("one_hot", Y_one_hot)
    Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
    print("reshape", Y_one_hot)
    keep_prob = tf.placeholder(tf.float32)
    W1 = tf.get_variable("W1", shape=[784, 512],
                         initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.Variable(tf.random_normal([512]))
    L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
    L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

    W2 = tf.get_variable("W2", shape=[512, 512],
                         initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.Variable(tf.random_normal([512]))
    L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
    L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

    W3 = tf.get_variable("W3", shape=[512, 512],
                         initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.Variable(tf.random_normal([512]))
    L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
    L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

    W4 = tf.get_variable("W4", shape=[512, 512],
                         initializer=tf.contrib.layers.xavier_initializer())
    b4 = tf.Variable(tf.random_normal([512]))
    L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
    L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

    W5 = tf.get_variable("W5", shape=[512, 10],
                         initializer=tf.contrib.layers.xavier_initializer())
    b5 = tf.Variable(tf.random_normal([10]))
    hypothesis = tf.matmul(L4, W5) + b5
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=hypothesis, labels=Y_one_hot))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    return (X, Y, Y_one_hot, hypothesis, cost, optimizer, keep_prob)