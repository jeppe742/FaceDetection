
# coding: utf-8

#import Jeppe_functions as fn
#from PIL import Image
import numpy as np
import os
import tensorflow as tf
from sklearn.metrics import f1_score 
from tensorflow.python.client import timeline
import matplotlib.pyplot as plt

from time import gmtime, strftime

# Define paths for the files
n_face = len(os.listdir("./images/face"))
n_noface = len(os.listdir("./images/noface"))

face_paths = ["./images/face/"+str(i)+".jpg" for i in range(n_face)]
noface_paths = ["./images/noface/"+str(i)+".jpg" for i in range(2*n_face)]

# and the labes
face_labels = np.ones((n_face,2))*[1,0]
noface_labels = np.ones((2*n_face,2))*[0,1]


#Add the faces twice. Random pertupations should avoid dublicates
paths = face_paths + face_paths + noface_paths

#Concatenate labels to vector
labels = np.concatenate((face_labels, face_labels))
labels = np.concatenate((labels, noface_labels))

#Read image, and randomly flip it
def read_data(filename_queue):
    reader = tf.WholeFileReader()
    _, image = reader.read(filename_queue)
    image = tf.image.per_image_standardization(tf.image.decode_jpeg(image, channels=1))
    return tf.image.random_flip_left_right(image)

#Returns accuracy
def accuracy(predictions, labels):
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]

#Returns the accuracy you would get if you only predicted no face.(To check if model only predicts the same output)
def accuracy_only_noface(labels):
    return 100*np.sum(np.argmax(labels, 1))/len(labels)

#Usefull parameters
samples = len(labels)
batch_size = 100
epochs = samples/batch_size
learning_rate = 1e-04
train_steps = 2000 
kernel_debth = 16
num_hidden_nodes = 64

#setup queues
file_queue = tf.train.string_input_producer(paths, num_epochs=100, seed=42)
label_queue = tf.train.input_producer(labels, num_epochs=100, seed=42)

#Create image op
image = read_data(file_queue)
image.set_shape((36, 36, 1))
image = tf.cast(image, tf.float32)
images, labels= tf.train.shuffle_batch([image, label_queue.dequeue()], batch_size=batch_size, num_threads=1, capacity=2000, min_after_dequeue=1000)


#define model Variabels
x = tf.placeholder(tf.float32, shape=(batch_size,36, 36, 1), name='input')
y = tf.placeholder(tf.float32, shape=(batch_size, 2), name='output')

#Convolution variables
conv_w1 = tf.Variable(tf.truncated_normal([36, 36, 1, kernel_debth]), name='Convweight1') #36x36x1x16
conv_b1 = tf.Variable(tf.zeros([kernel_debth]), name='bias1') #16
conv_w2 = tf.Variable(tf.truncated_normal([36, 36, kernel_debth, 2*kernel_debth]), name='Convweight2') #36x36x16x16
conv_b2 = tf.Variable(tf.zeros([2*kernel_debth]), name='bias2') #16
conv_w3 = tf.Variable(tf.truncated_normal([36, 36, 2*kernel_debth, 4*kernel_debth]), name='Convweight3') #36x36x16x16
conv_b3 = tf.Variable(tf.zeros([4*kernel_debth]), name='bias2') #16

#fully connected variables
fc_w1 = tf.Variable(tf.truncated_normal([5*5*4*kernel_debth, num_hidden_nodes]), name='fullyConnectedWeight') #1296x64
fc_b1 = tf.Variable(tf.zeros([num_hidden_nodes]), name='fullyConnectedBias') #64
fc_w2 = tf.Variable(tf.truncated_normal([num_hidden_nodes, 2]), name='HiddenLayerWeight') #1296x64
fc_b2 = tf.Variable(tf.zeros([2]), name='HiddenLayerBias') #64

with tf.name_scope("Convlayer1"):
    conv = tf.nn.conv2d(x, conv_w1, [1,1,1,1], padding='SAME')
    conv = tf.nn.relu(conv + conv_b1)
    pool = tf.nn.max_pool(conv, strides=[1,2,2,1], ksize=[1,2,2,1], padding='SAME')

with tf.name_scope("Convlayer2"):
    conv = tf.nn.conv2d(pool, conv_w2, [1,1,1,1], padding='SAME')
    conv = tf.nn.relu(conv + conv_b2)
    pool = tf.nn.max_pool(conv, strides=[1,2,2,1], ksize=[1,2,2,1], padding='SAME')

with tf.name_scope("Convlayer3"):
    conv = tf.nn.conv2d(pool, conv_w3, [1,1,1,1], padding='SAME')
    conv = tf.nn.relu(conv + conv_b3)
    pool = tf.nn.max_pool(conv, strides=[1,2,2,1], ksize=[1,2,2,1], padding='SAME')

with tf.name_scope('FullyConnected'):
    shape = pool.get_shape().as_list()
    net = tf.reshape(pool, [shape[0], shape[1]*shape[2]*shape[3]])
    hidden = tf.nn.relu(tf.matmul(net,fc_w1)+fc_b1)

with tf.name_scope('Hiddenlayer'):
    logits = tf.matmul(hidden,fc_w2)+fc_b2

train_predictions = tf.nn.softmax(logits)

#Type I and II errors
true_positive = tf.reduce_mean(tf.cast(tf.logical_and(tf.equal(tf.argmax(train_predictions,1), 1),tf.equal(tf.argmax(y,1), 1)), tf.float32))
false_positive = tf.reduce_mean(tf.cast(tf.logical_and(tf.equal(tf.argmax(train_predictions,1), 1),tf.equal(tf.argmax(y,1), 0)), tf.float32))
true_negative = tf.reduce_mean(tf.cast(tf.logical_and(tf.equal(tf.argmax(train_predictions,1), 0),tf.equal(tf.argmax(y,1), 1)), tf.float32))
false_negative = tf.reduce_mean(tf.cast(tf.logical_and(tf.equal(tf.argmax(train_predictions,1), 0),tf.equal(tf.argmax(y,1), 0)), tf.float32))

#Loss, train and accuracy OP
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits,y))
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(train_predictions,1), tf.argmax(y,1)), tf.float32))

#Summaries
tf.summary.scalar('true_positive',true_positive)
tf.summary.scalar('false_positive',false_positive)
tf.summary.scalar('true_negative',true_negative)
tf.summary.scalar('false_negative',false_negative)
tf.summary.scalar('Loss', loss)
tf.summary.scalar('accuracy',accuracy)
summary_op = tf.summary.merge_all()

#training time
with tf.Session() as sess:
    #Initialize variables
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    
    #Handle summaries and dataReader
    summary_writer = tf.summary.FileWriter(("./logs/bs_%1d_lr_%1.0e" %(batch_size,learning_rate)), graph= tf.get_default_graph())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    #Magic happens here
    for i in range(train_steps):
        #Contruct training data
        train_images, train_labels = sess.run([images, labels])
        feed_dict={x:train_images, y:train_labels}
        
        # if False:  # Record execution stats
        #     run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        #     run_metadata = tf.RunMetadata()
        #     summary, _ = sess.run([summary_op, train_op],feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
        #     tl = timeline.Timeline(run_metadata.step_stats)
        #     ctf = tl.generate_chrome_trace_format()
        #     with open('timeline.json', 'w') as f:
        #         f.write(ctf)
        #     summary_writer.add_run_metadata(run_metadata, 'step%d' % i)
        #     summary_writer.add_summary(summary, i)

        #Reporting
        if i%50==0:

                _, l,train_pred, summary, acc, true_pos, false_pos = sess.run([train_op, loss, train_predictions, summary_op, accuracy, true_positive, false_positive], feed_dict=feed_dict)
                summary_writer.add_summary(summary,i)

                print("step: %d  loss=%3.4f"%(i,l))
                print("accuracy train : %3.4f"%(acc))

        else:
            _ = sess.run([train_op], feed_dict=feed_dict)



