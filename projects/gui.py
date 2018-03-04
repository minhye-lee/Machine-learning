from tkinter import *
import tensorflow as tf
import numpy as np
import softmax as sf

def load_file(): #파일로드하는 함수
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


def get_paramiters(): #파라미터의 변수를 얻는 함수
    learning_rate = my_learning.get()
    batch_size = my_batch.get()
    epoch = my_epoch.get()

    learning_rate_int= float(learning_rate)
    batch_size_int = int(batch_size)
    epoch_int = int(epoch)
    print (learning_rate_int, batch_size_int, epoch_int)
    return (learning_rate_int, batch_size_int, epoch_int)

def click_load(): #로드 버튼 클릭시 이벤트
    load_label = Label(myContainer1, text="start loading")
    load_label.grid(row=4, column = 1)
    global mnist_test_x, mnist_test_y, mnist_train_x, mnist_train_y
    mnist_test_x, mnist_test_y, mnist_train_x, mnist_train_y = load_file()
    load_label.config(text = "finish loading")


def click_softmax(): #softmax 버튼 클릭시 이벤트
    print("softmax")
    learn, bat, epo = get_paramiters()
    X, Y, Y_one_hot, hypothesis, cost, optimizer = sf.softmax(learn)
    printbar.delete(1.0, END)
    sess = learning(epo, bat, mnist_train_x, mnist_train_y, cost, optimizer, X, Y, printbar)
    prediction = accuracy(hypothesis, Y_one_hot, mnist_test_x, mnist_test_y, sess, X, Y, printbar)
    #if click_details():
    details(sess, prediction, X, mnist_test_x, mnist_test_y,printbar)

def click_nn(): # nn버튼 클릭시 이벤트
    print("nn")
    learn, bat, epo = get_paramiters()
    X, Y, Y_one_hot, hypothesis, cost, optimizer = sf.nn(learn)
    printbar.delete(1.0, END)
    sess = learning(epo, bat, mnist_train_x, mnist_train_y, cost, optimizer, X, Y, printbar)
    prediction = accuracy(hypothesis, Y_one_hot, mnist_test_x, mnist_test_y, sess, X, Y, printbar)

    details(sess, prediction, X, mnist_test_x, mnist_test_y, printbar)

def click_xavier(): #xavier버튼 클릭시 이벤트
    print("xavier")
    learn, bat, epo = get_paramiters()
    X, Y, Y_one_hot, hypothesis, cost, optimizer = sf.xavier(learn)
    printbar.delete(1.0, END)
    sess = learning(epo, bat, mnist_train_x, mnist_train_y, cost, optimizer, X, Y, printbar)
    prediction = accuracy(hypothesis, Y_one_hot, mnist_test_x, mnist_test_y, sess, X, Y, printbar)

    details(sess, prediction, X, mnist_test_x, mnist_test_y, printbar)

def click_deep(): #deep nn 버튼 클릭시 이벤트
    print("deep")
    learn, bat, epo = get_paramiters()
    X, Y, Y_one_hot, hypothesis, cost, optimizer, keep = sf.deep_dropout(learn)
    printbar.delete(1.0, END)
    sess = learning_dropout(epo, bat, mnist_train_x, mnist_train_y, cost, optimizer, X, Y, printbar, keep)
    prediction = accuracy_dropout(hypothesis, Y_one_hot, mnist_test_x, mnist_test_y, sess, X, Y, keep, printbar)
    details_dropout(sess, prediction, X, mnist_test_x, mnist_test_y, keep, printbar)

def learning(training_epochs, batch_size, mnist_train_x, mnist_train_y, cost, optimizer, X, Y, printT): #학습하는 함수

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

        temp = 'Epoch : ' + '%04d' % (epoch + 1) + 'cost = ' + '{:.9f}'.format(avg_cost) + '\n'
        print(temp)
        printT.insert(END, temp)
    print('Learning Finished!')
    return (sess)

def accuracy(hypothesis, Y_one_hot, mnist_test_x, mnist_test_y, sess, X, Y, printT): #정확도 도출 함수
    printT.insert(END, '\n')
    prediction = tf.argmax(hypothesis, 1)
    correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    temp2 = str(sess.run(accuracy, feed_dict={X: mnist_test_x, Y: mnist_test_y}))
    temp = 'Accuracy : ' +  temp2 + '\n'
    print(temp)
    printT.insert(END, temp)

    return (prediction)

def details(sess, prediction, X, mnist_test_x, mnist_test_y, printT): #세부사항 확인 함수
    printT.insert(END, '\n')
    pred = sess.run(prediction, feed_dict={X: mnist_test_x})
    # y_data: (N,1) = flatten => (N, ) matches pred.shape
    for p, y in zip(pred, mnist_test_y.flatten()):
        temp = "[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)) + '\n'
        print(temp)
        printT.insert(END, temp)


def learning_dropout(training_epochs, batch_size, mnist_train_x, mnist_train_y, cost, optimizer, X, Y, printT, keep_prob): #dropout 포함된 학습함수
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist_train_x.shape[0] / batch_size)

        for i in range(total_batch):
            batch_xs = mnist_train_x[i * 100: i * 100 + (batch_size)]
            batch_ys = mnist_train_y[i * 100: i * 100 + (batch_size)]
            feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 1}
            c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
            avg_cost += c / total_batch

        temp = 'Epoch : '+ '%04d' % (epoch + 1) + 'cost = ' + '{:.9f}'.format(avg_cost) +'\n'
        print(temp)
        printT.insert(END, temp)
    print('Learning Finished!')
    return (sess)

def accuracy_dropout(hypothesis, Y_one_hot, mnist_test_x, mnist_test_y, sess, X, Y, keep_prob, printT): #dropout 포함된 정확도 도출 함수
    printT.insert(END, '\n')
    prediction = tf.argmax(hypothesis, 1)
    correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    temp2 = str(sess.run(accuracy, feed_dict={X: mnist_test_x, Y: mnist_test_y, keep_prob : 1}))
    temp = 'Accuracy : ' +  temp2 + '\n'
    print(temp)
    printT.insert(END, temp)
    return (prediction)

def details_dropout(sess, prediction, X, mnist_test_x, mnist_test_y, keep_prob, printT): #dropout포함된 세부사항 확인 함수
    printT.insert(END, '\n')
    pred = sess.run(prediction, feed_dict={X: mnist_test_x, keep_prob: 1})
    for p, y in zip(pred, mnist_test_y.flatten()):
        temp = "[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)) + '\n'
        print(temp)
        printT.insert(END, temp)



root = Tk()

myContainer1 = Frame(root, height = 50, width = 100)
myContainer1.pack()

my_learning = StringVar()
my_batch = StringVar()
my_epoch = StringVar()

l_learning = Label(myContainer1, text="learning rate")
l_learning.grid(row=0, column=0)
t_learning = Entry(myContainer1, textvariable = my_learning)
t_learning.grid(row=0, column=1)

l_batch = Label(myContainer1, text="batch size")
l_batch.grid(row=1, column=0)
t_batch = Entry(myContainer1, textvariable = my_batch)
t_batch.grid(row=1, column=1)

l_epoch = Label(myContainer1, text="training epoch")
l_epoch.grid(row=2, column=0)
t_epoch = Entry(myContainer1, textvariable = my_epoch)
t_epoch.grid(row=2, column=1)

b_load = Button(myContainer1, command = click_load)
b_load["text"] = "Load files"
b_load.grid(row=0, column=2)

button1 = Button(myContainer1, command = click_softmax)
button1["text"] = "Softmax"
button1.grid(row=1 ,column=2)

button2 = Button(myContainer1, command = click_nn)
button2["text"] = "nn"
button2.grid(row=2 ,column=2)

button3 = Button(myContainer1, command = click_xavier)
button3["text"] = "xavier nn"
button3.grid(row=3 ,column=2)

button4 = Button(myContainer1, command = click_deep)
button4["text"] = "deep nn"
button4.grid(row=4 ,column=2)

printbar = Text(myContainer1, height = 30, width = 40)
printbar.config(undo=False, wrap = 'word')
printbar.grid(row=5, column=1, sticky='nsew', padx=2, pady=2)

scrollb = Scrollbar(myContainer1, command=printbar.yview, width = 2)
scrollb.grid(row=5, column=2, sticky='nsew')
printbar['yscrollcommand'] = scrollb.set


root.mainloop()