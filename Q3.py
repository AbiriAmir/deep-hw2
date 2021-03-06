import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

cifar10_dataset_folder_path = 'cifar-10-batches-py'

# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# y_train = one_hot_encode(y_train, OUTPUT_LAYER_SIZE)
# y_test = one_hot_encode(y_test, OUTPUT_LAYER_SIZE)


def load_label_names():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def load_cfar10_batch(cifar10_dataset_folder_path, batch_id):
    with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
        # note the encoding type is 'latin1'
        batch = pickle.load(file, encoding='latin1')

    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']

    return features, labels

def display_stats(cifar10_dataset_folder_path, batch_id, sample_id):
    features, labels = load_cfar10_batch(cifar10_dataset_folder_path, batch_id)

    if not (0 <= sample_id < len(features)):
        print('{} samples in batch {}.  {} is out of range.'.format(len(features), batch_id, sample_id))
        return None

    print('\nStats of batch #{}:'.format(batch_id))
    print('# of Samples: {}\n'.format(len(features)))

    label_names = load_label_names()
    label_counts = dict(zip(*np.unique(labels, return_counts=True)))
    for key, value in label_counts.items():
        print('Label Counts of [{}]({}) : {}'.format(key, label_names[key].upper(), value))

    sample_image = features[sample_id]
    sample_label = labels[sample_id]

    print('\nExample of Image {}:'.format(sample_id))
    print('Image - Min Value: {} Max Value: {}'.format(sample_image.min(), sample_image.max()))
    print('Image - Shape: {}'.format(sample_image.shape))
    print('Label - Label Id: {} Name: {}'.format(sample_label, label_names[sample_label]))

def normalize(x):
    """
        argument
            - x: input image data in numpy array [32, 32, 3]
        return
            - normalized x
    """
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x

def one_hot_encode(x):
    """
        argument
            - x: a list of labels
        return
            - one hot encoding matrix (number of labels, number of class)
    """
    encoded = np.zeros((len(x), 10))

    for idx, val in enumerate(x):
        encoded[idx][val] = 1

    return encoded

def _preprocess_and_save(normalize, one_hot_encode, features, labels, filename):
    features = normalize(features)
    labels = one_hot_encode(labels)

    pickle.dump((features, labels), open(filename, 'wb'))

def preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode):
    n_batches = 5
    valid_features = []
    valid_labels = []

    for batch_i in range(1, n_batches + 1):
        features, labels = load_cfar10_batch(cifar10_dataset_folder_path, batch_i)

        # find index to be the point as validation data in the whole dataset of the batch (10%)
        index_of_validation = int(len(features) * 0.1)

        # preprocess the 90% of the whole dataset of the batch
        # - normalize the features
        # - one_hot_encode the lables
        # - save in a new file named, "preprocess_batch_" + batch_number
        # - each file for each batch
        _preprocess_and_save(normalize, one_hot_encode,
                             features[:-index_of_validation], labels[:-index_of_validation],
                             'preprocess_batch_' + str(batch_i) + '.p')

        # unlike the training dataset, validation dataset will be added through all batch dataset
        # - take 10% of the whold dataset of the batch
        # - add them into a list of
        #   - valid_features
        #   - valid_labels
        valid_features.extend(features[-index_of_validation:])
        valid_labels.extend(labels[-index_of_validation:])

    # preprocess the all stacked validation dataset
    _preprocess_and_save(normalize, one_hot_encode,
                         np.array(valid_features), np.array(valid_labels),
                         'preprocess_validation.p')

    # load the test dataset
    with open(cifar10_dataset_folder_path + '/test_batch', mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    # preprocess the testing data
    test_features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    test_labels = batch['labels']

    # Preprocess and Save all testing data
    _preprocess_and_save(normalize, one_hot_encode,
                         np.array(test_features), np.array(test_labels),
                         'preprocess_training.p')

def conv_net(x, keep_prob):
    conv1_filter = tf.Variable(tf.truncated_normal(shape=[5, 5, 3, 64], mean=0, stddev=0.01))
    conv2_filter = tf.Variable(tf.truncated_normal(shape=[5, 5, 64, 64], mean=0, stddev=0.01))

    # 1, 2
    conv1 = tf.nn.conv2d(x, conv1_filter, strides=[1,1,1,1], padding='SAME')
    conv1 = tf.nn.relu(conv1)
    conv1_pool = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,1,1,1], padding='SAME')
    # conv1_bn = tf.layers.batch_normalization(conv1_pool)
    conv1_bn = conv1_pool

    # 3, 4
    conv2 = tf.nn.conv2d(conv1_bn, conv2_filter, strides=[1,1,1,1], padding='SAME')
    conv2 = tf.nn.relu(conv2)
    conv2_pool = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    # conv2_bn = tf.layers.batch_normalization(conv2_pool)
    conv2_bn = conv2_pool

    # 9
    flat = tf.contrib.layers.flatten(conv2_bn)

    # 10
    full1 = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=512, activation_fn=tf.nn.relu)
    full1 = tf.nn.dropout(full1, keep_prob)
    # full1 = tf.layers.batch_normalization(full1)

    # 14
    out = tf.contrib.layers.fully_connected(inputs=full1, num_outputs=10, activation_fn=None)
    return out, conv1, conv2

def train_neural_network(x, y, keep_prob, session, optimizer, keep_probability, feature_batch, label_batch):
    session.run(optimizer,
                feed_dict={
                    x: feature_batch,
                    y: label_batch,
                    keep_prob: keep_probability
                })

def print_stats(x, y, keep_prob, session, feature_batch, label_batch, cost, accuracy, valid_features, valid_labels):
    loss = session.run(cost,
                    feed_dict={
                        x: feature_batch,
                        y: label_batch,
                        keep_prob: 1.
                    })
    valid_acc = session.run(accuracy,
                         feed_dict={
                             x: valid_features,
                             y: valid_labels,
                             keep_prob: 1.
                         })

    print('Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(loss, valid_acc))

def main():
    # Explore the dataset
    batch_id = 3
    sample_id = 7000
    display_stats(cifar10_dataset_folder_path, batch_id, sample_id)

    # Preprocess all the data and save it
    preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode)
    # features, labels = load_cfar10_batch(cifar10_dataset_folder_path, batch_i)

    # load the saved dataset
    valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))

    # Hyper parameters
    epochs = 10
    batch_size = 128
    keep_probability = 0.7
    learning_rate = 0.001

    # Remove previous weights, bias, inputs, etc..
    tf.reset_default_graph()

    # Inputs
    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='input_x')
    y = tf.placeholder(tf.float32, shape=(None, 10), name='output_y')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # Build model
    logits, conv1_filter, conv2_filter = conv_net(x, keep_prob)
    model = tf.identity(logits, name='logits') # Name logits Tensor, so that can be loaded from disk after training

    # Loss and Optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Accuracy
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

    # Training Phase
    save_model_path = './image_classification'

    costs = []
    accuracies = []
    print('Training...')
    with tf.Session() as sess:
        # Initializing the variables
        sess.run(tf.global_variables_initializer())

        # Training cycle
        for epoch in range(epochs):
            # Loop over all batches
            n_batches = 5
            for batch_i in range(1, n_batches + 1):
                batch_features, batch_labels = pickle.load(open('preprocess_batch_' + str(batch_i) + '.p', mode='rb'))
                batch_features = batch_features[:1000]
                batch_labels = batch_labels[:1000]

                # print("Batch features: ", batch_features)

                train_neural_network(x, y, keep_prob, sess, optimizer, keep_probability, batch_features, batch_labels)

                print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
                costs.append(cost)
                accuracies.append(accuracy)

                print_stats(x, y, keep_prob, sess, batch_features, batch_labels, cost, accuracy, valid_features, valid_labels)

        # Save Model
        saver = tf.train.Saver()
        save_path = saver.save(sess, save_model_path)


        # print(sess.run(conv1_filter))

        horse_image = None
        for valid_feature, valid_result in zip(valid_features, valid_labels):
            if valid_result[load_label_names().index('horse')] == 1:
                horse_image = valid_feature
                break

        plt.imsave('horse_orig', horse_image)
        #
        for i in range(8):
            for j in range(8):
                fig, axarr = plt.subplots(8, 8)
                for i in range(8):
                    for j in range(8):
                        conv1_filter_result = sess.run(conv1_filter, feed_dict={x: [horse_image]})
                        horse_conv1 = conv1_filter_result[0, :, :, i * 8 + j]
                        axarr[i, j].imshow(horse_conv1, cmap='gray')
                        axarr[i, j].xaxis.set_visible(False)
                        axarr[i, j].yaxis.set_visible(False)
                fig.savefig('first_layer_images')
                plt.close(fig)
        #
        horse_conv2 = sess.run(conv2_filter, feed_dict={x: [horse_image]})[0, :, :, 0]
        plt.imsave('second_layer_image', horse_conv2, cmap='gray')
        #
        print('Horse images generation finished.')


    plt.plot(accuracies)
    plt.ylabel('Training Accuracy')
    plt.show()

    plt.plot(costs)
    plt.ylabel('Cost')
    plt.show()
    
if __name__ == "__main__":
    main()
