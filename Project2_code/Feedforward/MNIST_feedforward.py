from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

def main():
    # Parameters
    hidden_layer = [512, 256, 128, 64]
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
    W.append (tf.Variable (tf.zeros ([784, hidden_layer[0]])))
    num_hidden_layer = len (hidden_layer)
    for i in range (num_hidden_layer):
        if i < num_hidden_layer - 2:
            W.append (tf.Variable (tf.zeros ([hidden_layer[i], hidden_layer[i + 1]])) ) 
        else:
            W.append (tf.Variable (tf.zeros ([hidden_layer[i], num_class)) ) 







    ######################

    
    # Define loss function(Cross entropy)
    ### Comment this line in problem 2 ###
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_label, logits=out_layer))
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
            print('Iteration: {}/{} Accuracy = {:.2f}% Loss = {:.2f}'.format(step, max_iter, 100*acc, loss))
    print('Optimization finished')

    # Test network
    acc, loss = sess.run([accuracy, cross_entropy], feed_dict={x: mnist.test.images, y_label: mnist.test.labels})
    print('Test accuracy = {:.2f}%, Test loss = {:.2f}'.format(100*acc, loss))

if __name__ == '__main__':
    main()