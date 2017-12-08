import numpy as np
import tensorflow as tf
#import sys
#import random
#random.seed (2)

def main():
    # Load data and split it to training and test data
    data = np.loadtxt('stock_data.txt')
    data_train_X = data[:450,:-1]
    data_train_Y = data[:450,-1]
    data_test_X  = data[450:-1,:-1]
    data_test_Y  = data[450:-1,-1]

    mean_test_Y = np.mean (data_test_Y)
    std_test_Y = np.std (data_test_Y)

    print ("mean_test_Y:", mean_test_Y, "std_test_Y:", std_test_Y)

    # Parameters
    INPUT_SIZE    = 8
    OUTPUT_SIZE   = 1
    HIDDEN_SIZE   = 50
    LEARNING_RATE = 0.001
    nEpoch 	      = 100
    NET_TYPE = "rnn" # "lstm", "rnn", "gru" 

    # Placehoders for inputs and outputs
    inputs   = tf.placeholder(tf.float32, (None, INPUT_SIZE))
    outputs  = tf.placeholder(tf.float32, (None))
    inputs_  = tf.transpose(tf.expand_dims(inputs,0),[2,1,0])
    outputs_ = tf.expand_dims(outputs,0)
    ### Your code here ###
    # RNN/LSTM/GRU layer
    layer = {'weights':tf.Variable(tf.random_normal([HIDDEN_SIZE,1])),
                 'biases':tf.Variable(tf.random_normal([1]))}

    if NET_TYPE == "lstm":
        cell = tf.nn.rnn_cell.BasicLSTMCell (HIDDEN_SIZE, state_is_tuple=True) #, activation=tf.nn.relu)
        #cell = tf.nn.rnn_cell.BasicLSTMCell (HIDDEN_SIZE) #, activation=tf.nn.relu)

    elif NET_TYPE == "rnn":
        cell = tf.nn.rnn_cell.BasicRNNCell (HIDDEN_SIZE)

    elif NET_TYPE == "gru":
        cell = tf.nn.rnn_cell.GRUCell (HIDDEN_SIZE)

    else:
        raise ValueError ("This network was not supported!")

    #net_outputs, states = tf.nn.static_rnn (cell, inputs_, dtype=tf.float32) #, states
    net_outputs, states = tf.nn.dynamic_rnn (cell, inputs_, dtype=tf.float32) #, states
    #net_outputs = tf.nn.dynamic_rnn (cell, inputs_, dtype=tf.float32) #, states

    output = tf.matmul(net_outputs[-1],layer['weights']) + layer['biases']

    # Define loss function(MSE)
    loss = tf.reduce_mean (tf.squared_difference (output, outputs_))
    ######################

    # Define optimizer
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    # Train
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    train_mse_list = []
    test_mse_list = []
    epoch_list = []
    for i in range(nEpoch):
        _, train_mse = sess.run([optimizer, loss], {inputs: data_train_X, outputs: data_train_Y})
        pred_y, test_mse = sess.run([output, loss], {inputs: data_test_X, outputs: data_test_Y})
        print("Epoch: %d Train loss: %.5f Test loss: %.5f " % (i+1, train_mse, test_mse))
        epoch_list.append (i)
        train_mse_list.append (train_mse)
        test_mse_list.append (test_mse)

    #print (pred_y.shape, data_test_Y.shape)
    #sys.exit (-1)
    a = np.concatenate ((pred_y, data_test_Y.reshape (data_test_Y.shape[0],1)), axis=1)
    #print (a)
    line = np.zeros(len (epoch_list), dtype=[('epoch', int), ('train_mse', float), ('test_mse', float)])
    line['epoch'] = epoch_list
    line['train_mse'] = train_mse_list
    line['test_mse'] = test_mse_list
    np.savetxt( NET_TYPE+"_loss.txt", line, fmt="%d\t%f\t%f")

if __name__=='__main__':
    main()
