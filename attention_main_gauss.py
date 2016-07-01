# -*- coding: utf-8 -*-
"""
Created on Sunday June 26

@author: Rob Romijnders


"""
import sys
sys.path.append('/home/rob/Dropbox/ml_projects/attention')
sys.path.append('/home/rob/Dropbox/ml_projects/FCN/')
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
import sklearn as sk
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn

from mpl_toolkits.mplot3d import axes3d

"""Load the data"""
direc = '/home/rob/Dropbox/ml_projects/FCN/'
X_test = np.loadtxt(direc+'X_test.csv', delimiter=',')
y_test = np.loadtxt(direc+'y_test.csv', delimiter=',')
X_train = np.loadtxt(direc+'X_train.csv', delimiter=',')
y_train = np.loadtxt(direc+'y_train.csv', delimiter=',')
F_train = np.load(direc+'feat_train.npy')
F_test = np.load(direc+'feat_test.npy')

N,H,W,C = F_train.shape
Ntest = F_test.shape[0]
print('Data is loaded')

"""Hyperparameters"""
hidden_size = H*W		#Number of hidden units in LSTM
num_layers = 1
max_iterations = 1000
batch_size = 64
dropout = 0.5       #Dropout rate in the fully connected layer
learning_rate = .001
sl = 10           #sequence length
plot_every = 100    #How often do you want terminal output for the performances
max_grad_norm = 1
num_classes = 10




#Proclaim the epochs
epochs = np.floor(batch_size*max_iterations / N)
print('Train with approximately %d epochs' %(epochs))

# Nodes for the input variables
x = tf.placeholder(tf.float32, shape=[batch_size, H,W,C], name = 'Input_data')
y_ = tf.placeholder(tf.int64, shape=[batch_size], name = 'Ground_truth')
keep_prob = tf.placeholder("float")


x1 = np.linspace(-3,3,H)
x2 = np.linspace(-3,3,W)
X1,X2 = np.meshgrid(x1,x2)
X1 = X1.flatten()
X2 = X2.flatten()

XX1 = np.tile(X1,(batch_size,1))
XX2 = np.tile(X2,(batch_size,1))

M1 = tf.constant(XX1,dtype=tf.float32)
M2 = tf.constant(XX2,dtype=tf.float32)
def tf_2d_normal(x1, x2, mu1, mu2, s1, s2, rho):
  #Inspired from Hardmaru's implementation on Github
  norm1 = tf.sub(x1, mu1)
  norm2 = tf.sub(x2, mu2)
  s1s2 = tf.mul(s1, s2)
  z = tf.square(tf.div(norm1, s1))+tf.square(tf.div(norm2, s2))-2*tf.div(tf.mul(rho, tf.mul(norm1, norm2)), s1s2)
  negRho = 1-tf.square(rho)
  result = tf.exp(tf.div(-z,2*negRho))
  denom = 2*np.pi*tf.mul(s1s2, tf.sqrt(negRho))
  result = tf.div(result, denom)
  return result




with tf.name_scope("LSTM") as scope:
  cell = rnn_cell.LSTMCell(hidden_size)
  #cell = rnn_cell.MultiRNNCell([cell] * num_layers)
  cell = rnn_cell.DropoutWrapper(cell,output_keep_prob=keep_prob)
  #XW_plus_b
  W_a = tf.Variable(tf.random_normal([hidden_size,5], stddev=0.01))
  b_a = tf.Variable(tf.constant(0.5, shape=[5]))
  #Initial state
  initial_state = cell.zero_state(batch_size, tf.float32)
  #initial input vector is a sum over the activation map
  x_in = tf.reduce_sum(x,[1,2])
  time = sl*tf.ones([batch_size,1])
  x_in = tf.concat(1,[x_in,time])
  outputs = []
  masks = []
  state = initial_state
  for time_step in range(sl):
    if time_step > 0: tf.get_variable_scope().reuse_variables()
    (cell_output, state) = cell(x_in, state)
    #Extract features that parametrize the Gaussian
    act_a = tf.nn.xw_plus_b(cell_output,W_a,b_a)   #Activations for the attention
    mu1,mu2,s1,s2,rho = tf.split(1,5,act_a)
    #Set the range of the variances and correlation
    s1 = tf.exp(s1)
    s2 = tf.exp(s2)
    rho = tf.tanh(rho)
    #obtain mask and reshape
    mask = tf_2d_normal(M1,M2,mu1,mu2,s1,s2,rho)
    p_mask = tf.reshape(mask,[batch_size,H,W,1])   #reshape into the mask
    masks.append(p_mask)   #Save masks for later visualization
    ex = tf.mul(p_mask,x)
    x_in = tf.reduce_sum(ex,[1,2])
    time = (sl-time_step-1)*tf.ones([batch_size,1])
    x_in = tf.concat(1,[x_in,time])

    outputs.append(cell_output)
  #outputs is now a list of length seq_len with tensors [ batch_size by hidden_size ]

with tf.name_scope("SoftMax") as scope:
  final = outputs[-1]
  W_c = tf.Variable(tf.random_normal([hidden_size,num_classes], stddev=0.01))
  b_c = tf.Variable(tf.constant(0.1, shape=[num_classes]))
  h_c = tf.nn.xw_plus_b(final, W_c,b_c)
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(h_c,y_)
  cost = tf.reduce_mean(loss)

with tf.name_scope("train") as scope:
  tvars = tf.trainable_variables()
  #We clip the gradients to prevent explosion
  grads = tf.gradients(cost, tvars)
  grads, _ = tf.clip_by_global_norm(grads,0.5)
  optimizer = tf.train.AdamOptimizer(learning_rate)
  gradients = zip(grads, tvars)
  train_step = optimizer.apply_gradients(gradients)
  # The following block plots for every trainable variable
  #  - Histogram of the entries of the Tensor
  #  - Histogram of the gradient over the Tensor
  #  - Histogram of the grradient-norm over the Tensor
  numel = tf.constant([[0]])
  for gradient, variable in gradients:
    if isinstance(gradient, ops.IndexedSlices):
      grad_values = gradient.values
    else:
      grad_values = gradient

    numel +=tf.reduce_sum(tf.size(variable))

    h1 = tf.histogram_summary(variable.name, variable)
    h2 = tf.histogram_summary(variable.name + "/gradients", grad_values)
    h3 = tf.histogram_summary(variable.name + "/gradient_norm", clip_ops.global_norm([grad_values]))


with tf.name_scope("Evaluating_accuracy") as scope:
  correct_prediction = tf.equal(tf.argmax(h_c,1), y_)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print('Finished computation graph')

#Define one op to call all summaries
merged = tf.merge_all_summaries()

# For now, we collect performances in a Numpy array.
# In future releases, I hope TensorBoard allows for more
# flexibility in plotting
perf_collect = np.zeros((6,int(np.floor(max_iterations /plot_every))))

sess = tf.Session()

#with tf.Session() as sess:
if True:
  writer = tf.train.SummaryWriter("/home/rob/Dropbox/ml_projects/attention/log_tb", sess.graph)

  sess.run(tf.initialize_all_variables())

  step = 0      # Step is a counter for filling the numpy array perf_collect
  for i in range(max_iterations):
    batch_ind = np.random.choice(N,batch_size,replace=False)

#    debug = sess.run(initial_state,feed_dict={x:F_train[batch_ind], y_: y_train[batch_ind], keep_prob: dropout})
#    print(np.max(debug[0]))
#    print(np.max(debug[1]))


    if i%plot_every == 0:
      #Check training performance
      result = sess.run([accuracy,cost],feed_dict = { x: F_train[batch_ind], y_: y_train[batch_ind], keep_prob: 1.0})
      perf_collect[0,step] = result[0]
      perf_collect[1,step] = cost_train = result[1]

      #Check testidation performance
      batch_ind_test = np.random.choice(Ntest,batch_size,replace=False)
      result = sess.run([accuracy,cost,merged], feed_dict={ x: F_test[batch_ind_test], y_: y_test[batch_ind_test], keep_prob: 1.0})
      acc = result[0]
      perf_collect[2,step] = acc
      perf_collect[3,step] = cost_test = result[1]

      #Write information to TensorBoard
      summary_str = result[2]
      writer.add_summary(summary_str, i)
      writer.flush()  #Don't forget this command! It makes sure Python writes the summaries to the log-file
      print("At %6s / %6s test acc %5.3f and AUC is %5.3f trainloss %5.3f" % (i,max_iterations, acc, 0.0,cost_train ))
      step +=1
    sess.run(train_step,feed_dict={x:F_train[batch_ind], y_: y_train[batch_ind], keep_prob: dropout})


"""Visualize the soft attention"""
batch_ind_test = np.random.choice(Ntest,batch_size,replace=False)
masks_fetch = sess.run(masks, feed_dict={ x: F_test[batch_ind_test], y_: y_test[batch_ind_test], keep_prob: 1.0})

ind =23
bind = batch_ind_test[ind]
plt.subplot(3,4,1)
plt.imshow(np.reshape(X_test[bind],[28,28]).T,origin='upper')
for p in range(sl):
  plt.subplot(3,4,p+2)   #p starts at 0, subplots start at 2
  plt.imshow(masks_fetch[p][ind,:,:,0].T)

plt.subplot(3,4,12)
plt.imshow(np.sum(F_test[bind,:,:,:],axis=2).T)
