import numpy as np
import pandas as pd
from scipy.misc import imread
from sklearn.metrics import accuracy_score
import tensorflow as tf
import matplotlib.pyplot as plt

from Helper_Func import load_dataset
X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
print(X_train.shape, y_train.shape)
%matplotlib inline
plt.imshow(X_train[0], cmap="Greys");

from Helper_Func import one_hot_encode_labels
from Helper_Func import create_batch
inputUnits = 28*28
outputUnits = 10
epochs = 70
batchSize = 256
learningRate = 0.02

#Placeholders for input output
X = tf.placeholder(tf.float32,[None,inputUnits])
Y = tf.placeholder(tf.float32,[None,outputUnits])

#Variables for wights and biases
weights = {
    'layer1': tf.Variable(tf.random_normal([inputUnits, 150])),
    'layer2': tf.Variable(tf.random_normal([150, 50])),
    #'layer3': tf.Variable(tf.random_normal([100, 100])),
    'output': tf.Variable(tf.random_normal([50, outputUnits]))
}
biases = {
    'layer1': tf.Variable(tf.random_normal([150])),
    'layer2': tf.Variable(tf.random_normal([50])),
    #'layer3': tf.Variable(tf.random_normal([100])),
    'output': tf.Variable(tf.random_normal([outputUnits]))
}
#Build graph
#Layer1
hiddenLayer1 = tf.add(tf.matmul(X,weights['layer1']),biases['layer1'])
hiddenLayer1 = tf.layers.batch_normalization(hiddenLayer1)
hiddenLayer1 = tf.nn.relu(hiddenLayer1)
#Dropout
#hiddenLayer1 = tf.nn.dropout(hiddenLayer1,keep_prob=0.7)
#Layer2
hiddenLayer2 = tf.add(tf.matmul(hiddenLayer1,weights['layer2']),biases['layer2'])
hiddenLayer2 = tf.layers.batch_normalization(hiddenLayer2)
hiddenLayer2 = tf.nn.relu(hiddenLayer2)
#Dropout
#hiddenLayer2 = tf.nn.dropout(hiddenLayer2,keep_prob=0.8)
#Layer3
#hiddenLayer3 = tf.add(tf.matmul(hiddenLayer2,weights['layer3']),biases['layer3'])
#hiddenLayer3 = tf.layers.batch_normalization(hiddenLayer3)
#hiddenLayer3 = tf.nn.relu(hiddenLayer3)
#output
output = tf.add(tf.matmul(hiddenLayer2,weights['output']),biases['output'])

#cost 
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=Y))
regularizer = tf.nn.l2_loss(weights['output'])
cost = tf.reduce_mean(cost + 0.1 * regularizer)
#optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(cost)
correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
init = tf.global_variables_initializer()

def train(sess):
    totalTrainBatch_X,totalTrainBatch_Y = create_batch(X_train,y_train,X_train.shape[0],0)
    for epoch in range(epochs):
        iterationPerEpoch = int(X_train.shape[0]/batchSize)
        avgCost = 0
        for iteration in range(iterationPerEpoch):
            batch_X,batch_Y = create_batch(X_train,y_train,batchSize,batchSize*iteration)
            _,c = sess.run([optimizer,cost],feed_dict={X:batch_X , Y:batch_Y})
            avgCost += c /iterationPerEpoch
        print("Epoch #"+str(epoch)+" cost = " +str(avgCost))
    print("finished !!!")
    print("train Accuracy = ",sess.run(accuracy, feed_dict={X: totalTrainBatch_X, Y: totalTrainBatch_Y}))
    X_val_batch,y_val_batch = create_batch(X_val,y_val,X_val.shape[0],0)
    print("validate Accuracy = ",sess.run(accuracy, feed_dict={X: X_val_batch, Y: y_val_batch}))
def evaluate(sess):
    X_test_batch,y_test_batch =create_batch(X_test,y_test,X_test.shape[0],0) 
    print("test Accuracy = ",sess.run(accuracy, feed_dict={X: X_test_batch, Y: y_test_batch}))


with tf.Session() as sess:
    sess.run(init)
    train(sess)
    evaluate(sess)
