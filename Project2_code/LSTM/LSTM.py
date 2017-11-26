import numpy as np
import tensorflow as tf

def main():
    # Load data and split it to training and test data
    data = np.loadtxt('stock_data.txt')
    data_train_X = data[:450,:-1]
    data_train_Y = data[:450,-1]
    data_test_X  = data[450:-1,:-1]
    data_test_Y  = data[450:-1,-1]

    # Parameters
    INPUT_SIZE    = 8
    OUTPUT_SIZE   = 1
    HIDDEN_SIZE   = 50
    LEARNING_RATE = 0.001
    nEpoch 	      = 100

    # Placehoders for inputs and outputs
    inputs   = tf.placeholder(tf.float32, (None, INPUT_SIZE))
    outputs  = tf.placeholder(tf.float32, (None))
    inputs_  = tf.transpose(tf.expand_dims(inputs,0),[2,1,0])
    outputs_ = tf.expand_dims(outputs,0)
    ### Your code here ###
    # RNN/LSTM/GRU layer

    # Define loss function(MSE)
    
    ######################

    # Define optimizer
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    # Train
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(nEpoch):
        _, train_mse = sess.run([optimizer, loss], {inputs: data_train_X, outputs: data_train_Y})
        test_mse = sess.run(loss, {inputs: data_test_X, outputs: data_test_Y})
        print("Epoch: %d Train loss: %.5f Test loss: %.5f " % (i+1, train_mse, test_mse))

if __name__=='__main__':
    main()