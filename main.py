import tensorflow as tf
from os import makedirs
from os.path import exists, join
import utils
from model import my_net
import numpy as np

if __name__== '__main__':

    input_h, input_w = 128, 128    # Shape to which input is resized
    
    dataset_file = 'traffic_light_dataset_npy/traffic_light_dataset_mixed_resize_{}.npy'.format(input_h)
    dataset_npy = utils.init_from_npy(dataset_file)

    dataset_test_file = 'traffic_light_dataset_npy/traffic_light_dataset_simulator_resize_{}.npy'.format(input_h)
    dataset_test_npy = utils.init_from_npy(dataset_test_file)

    # Tensor placeholders
    x = tf.placeholder(dtype=tf.float32,shape=(None,input_h,input_w,3))
    y = tf.placeholder(dtype=tf.int32, shape=None)
    keep_prob = tf.placeholder(dtype=tf.float32)

    # training pipeline
    EPS  = np.finfo('float32').eps
    learning_rate = 1e-4
    n_classes = 4              # {void, red, yellow, green}
    predictions = my_net(x,n_classes=n_classes,keep_prob=keep_prob)
    y_onehot = tf.one_hot(y, depth=n_classes)
    loss = tf.reduce_mean(-tf.reduce_sum(y_onehot * tf.log(predictions + EPS), reduction_indices=1))
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    correct_predictions = tf.equal(tf.argmax(predictions, axis=1), tf.argmax(tf.one_hot(y, depth=n_classes), axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


    # create a checkpointer to log the weights during training
    saver = tf.train.Saver()
    
    checkpoint_dir = './checkpoint_mixed_{}'.format(input_h) 
    if not exists(checkpoint_dir):
        makedirs(checkpoint_dir)
    
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
        
        BATCHSIZE = 32

        BATCHES_PER_EPOCH = 1000

        epoch = 0

        while epoch < 30:

            loss_cur_epoch = 0
            batch_counter = 0
            for _ in range(BATCHES_PER_EPOCH):
                
                # Load a batch of training data
                x_batch, y_batch = utils.load_batch(dataset_npy,batch_size=BATCHSIZE, augmentation=True)

                # feed batches and calculate loss
                _, loss_this_batch = sess.run(fetches=[train_step, loss], feed_dict={x: x_batch, y: y_batch, keep_prob: 0.5})
                loss_cur_epoch += loss_this_batch
                batch_counter += 1
                if batch_counter % 200 == 0:
                    print("BATCH {} / 1000".format(batch_counter), end='\r')

            loss_cur_epoch /= BATCHES_PER_EPOCH
            print('Loss cur epoch: {:.04f}'.format(loss_cur_epoch))

            # Evalute training and validation
            average_train_accuracy = 0.0
            average_val_accuracy = 0.0
            num_train_batches = 500
            num_val_batches = 16
            for _ in range(num_train_batches):
                x_batch, y_batch = utils.load_batch(dataset_npy,batch_size=BATCHSIZE)
                average_train_accuracy += sess.run(accuracy, feed_dict={x: x_batch,
                                                             y: y_batch,
                                                             keep_prob: 1.0})
            average_train_accuracy /= num_train_batches

            for _ in range(num_val_batches):
                x_batch, y_batch = utils.load_batch(dataset_test_npy,batch_size=num_val_batches)
                average_val_accuracy += sess.run(accuracy, feed_dict={x: x_batch,
                                                             y: y_batch,
                                                             keep_prob: 1.0})
            average_val_accuracy /= num_val_batches

            print('Training accuracy: {:.03f} - Validation accuracy: {:.03f}'.format(average_train_accuracy,average_val_accuracy))
            print('*' * 50)

            # Save the variables to log file.
            save_path = saver.save(sess, join(checkpoint_dir, 'TLC_epoch_{}.ckpt'.format(epoch)))

            epoch += 1