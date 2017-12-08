import matplotlib
matplotlib.use('Agg')
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import sys
import matplotlib.pyplot as plt
import numpy as np

def activation_func (x, W, b, func_type):
    """
        Activation function with different types: sigmoid, ReLu, tanh
    """
    a = tf.matmul (x, W) + b
    if func_type == "sigmoid":
        return tf.nn.sigmoid (a)
    elif func_type == "ReLu":
        return tf.nn.relu (a)
    elif func_type == "tanh":
        return tf.nn.tanh (a)
    else:
        raise ValueError (func_type + " is not supported!")

def main():
    # Parameters
    hidden_layer = [512]#*4#[512, 256, 128, 64]
    num_class = 10
    display_step = 100
    max_iter = 3000
    batch_size = 32
    alpha = 0.5

    # Import data
    mnist = input_data.read_data_sets('./', one_hot=True)

    # Placeholder
    x = tf.placeholder(tf.float32, [None, 784])
    y_label = tf.placeholder(tf.float32, [None, num_class])

    # Variables & hidden layer
    W = []
    b = []
    act_layer = []
    ### Your code here ###

    # Build network
    act_function = "ReLu" #, ReLu, tanh, sigmoid
    
    
    # Input layer
    W.append (tf.Variable (tf.random_normal ([784, hidden_layer[0]], stddev=0.03), name='W0'))
    b.append (tf.Variable (tf.random_normal ([hidden_layer[0]]), name='b0'))
    act_layer.append (activation_func (x, W[-1], b[-1], act_function))
    
    # Hidden layers
    num_hidden_layer = len (hidden_layer)
    
    for i in range (num_hidden_layer):
        if i < num_hidden_layer - 1:
            #W.append (tf.Variable (tf.zeros ([hidden_layer[i], hidden_layer[i + 1]]))) 
            #b.append (tf.Variable (tf.zeros ([hidden_layer[i+1]])))

            # Store weights (matrix) and biases (vector) into 2 lists W and b, initialize with random variable follow normal distribution with standard deviation = 0.03, and name of weights and biases corespond to each hidden layer
            W.append (tf.Variable (tf.random_normal ([hidden_layer[i], hidden_layer[i + 1]], stddev=0.03), name='W'+str(i+1))) 
            b.append (tf.Variable (tf.random_normal ([hidden_layer[i+1]]), name='b'+str(i+1)))

            # at each hidden layer, apply activation function
            act_layer.append (activation_func (act_layer[-1], W[-1], b[-1], act_function))
        else:
            # at the last layer, the output will be pass to softmax function 
            W.append (tf.Variable (tf.random_normal ([hidden_layer[i], num_class], stddev=0.03), name='W'+str(i+1))) 
            b.append (tf.Variable (tf.random_normal ([num_class]), name='b'+str(i+1)))
            o = tf.matmul (act_layer[-1], W[-1]) + b[-1]
     
    # output layer
    epsilon = tf.constant(value=0.001, shape=[32,10])
    o = o + epsilon
    y_label = tf.to_float(tf.reshape(y_label, (-1, 10)))

    #epsilon = tf.constant(value=0.00001, shape=[32,10])
    #o = o + epsilon
    out_layer = tf.nn.softmax(o) 
    ######################

    
    # Define loss function(Cross entropy)
    ### Comment this line in problem 2 ###
    #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_label, logits=out_layer))
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(out_layer), reduction_indices=[1]))
    #cross_entropy = -tf.reduce_mean(y_label * tf.log(out_layer))
    ######################################

    # Define optimizer
    optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(cross_entropy)

    # Define accuracy
    correct_prediction = tf.equal(tf.argmax(out_layer, 1), tf.argmax(y_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Define Session & initializer
    sess = tf.Session()
    init = tf.global_variables_initializer()

    # Train
    sess.run(init)
    for step in range(max_iter):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={x: batch_xs, y_label: batch_ys})
        if step%display_step == 0:
            acc, loss = sess.run([accuracy, cross_entropy], feed_dict={x: batch_xs, y_label: batch_ys})
            #print('Iteration: {}/{} Accuracy = {:.2f}% Loss = {:.2f}'.format(step, max_iter, 100*acc, loss))
            print ("%d\t%.2f\t%.2f" % (step, 100*acc, loss))
    print('Optimization finished')

    # Test network
    actu_y, pred_y, acc, loss = sess.run([y_label, out_layer, accuracy, cross_entropy], feed_dict={x: mnist.test.images, y_label: mnist.test.labels})
    print('Test accuracy = {:.2f}%, Test loss = {:.2f}'.format(100*acc, loss))
    print (actu_y, pred_y)

    # THIS WILL LOAD ONE TRAINING EXAMPLE
    """for num in range (10):
        x_test = mnist.test.images[num,:].reshape(1,784)
        y_test = mnist.test.labels[num,:]
        # THIS GETS OUR LABEL AS A INTEGER
        label = y_test.argmax()
        # THIS GETS OUR PREDICTION AS A INTEGER
        prediction = sess.run(out_layer, feed_dict={x: x_test}).argmax()

        plt.title('Prediction: %d Label: %d' % (prediction, label))
        plt.imshow(x_test.reshape([28,28]), cmap=plt.get_cmap('gray_r'))
        plt.savefig('myfig' + str (num))"""

if __name__ == '__main__':
    main()
