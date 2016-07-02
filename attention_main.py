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
#Load any form of MNIST here.
#X_ is expected to be in [number_samples, 784]
#y_ is expected to be in [number_samples,]

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
hidden_size = H*W      #Number of hidden units in LSTM
num_layers = 2
max_iterations = 10000
batch_size = 64
dropout = 0.8         #Dropout rate in the fully connected layer
learning_rate = .005
sl = 10               #sequence length
plot_every = 100      #How often do you want terminal output for the performances
max_grad_norm = 2     #Maximum gradient before clipping
num_classes = 10      #number of classes, 10 for MNIST

#Proclaim the epochs
epochs = np.floor(batch_size*max_iterations / N)
print('Train with approximately %d epochs' %(epochs))

def softmax_2d(value):
  """Calculates Softmax over dim 1 and 2"""
  value = tf.sub(value,tf.reduce_max(value,[1,2,3],keep_dims=True))  #substract max to prevent numeric instability
  nom = tf.exp(value)
  den = tf.reduce_sum(nom,[1,2,3],keep_dims=True)
  softmax = tf.div(nom,den)
  return softmax


with tf.name_scope("Placeholders") as scope:
  x = tf.placeholder("float", shape=[batch_size, H,W,C], name = 'Input_data')
  y_ = tf.placeholder(tf.int64, shape=[batch_size], name = 'Ground_truth')
  keep_prob = tf.placeholder("float")

with tf.name_scope("LSTM") as scope:
  cell = rnn_cell.LSTMCell(hidden_size)
  cell = rnn_cell.MultiRNNCell([cell] * num_layers)
  cell = rnn_cell.DropoutWrapper(cell,output_keep_prob=keep_prob)
  #XW_plus_b
  W_a = tf.Variable(tf.random_normal([hidden_size,C], stddev=0.01))
  b_a = tf.Variable(tf.constant(0.5, shape=[C]))
  #Initial state
  initial_state = cell.zero_state(batch_size, tf.float32)
  #initial input vector is a sum over the activation map


  x_in = tf.reduce_sum(x,[1,2])

  #Concatenate the time as an integer
  time = sl*tf.ones([batch_size,1])
  x_in = tf.concat(1,[x_in,time])

  outputs = []
  masks = []
  state = initial_state
  for time_step in range(sl):
    if time_step > 0: tf.get_variable_scope().reuse_variables()
    (cell_output, state) = cell(x_in, state)
    pat = tf.nn.xw_plus_b(cell_output,W_a,b_a)   #The pattern that the attention is looking for in [batch_size,C,]
    pat_ex = tf.expand_dims(tf.expand_dims(pat,1),2)
    pat_match = tf.mul(x,pat_ex)  #x was in [batch_size, H,W,C]
    pat_templ = tf.reduce_sum(pat_match,3,keep_dims=True)   #now in [batch_size, H,W,1]
    p_mask = softmax_2d(pat_templ)    #now in [batch_size, H,W,1]
    masks.append(p_mask)   #Save masks for later visualization
    ex = tf.mul(p_mask,x)   #now in [batch_size, H,W,C]
    x_in = tf.reduce_sum(ex,[1,2])    #now in [batch_size, C]
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
  grads, _ = tf.clip_by_global_norm(grads,max_grad_norm)
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

    if i%plot_every == 0:   #plot_every is how often you want a print to terminal
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
      print("At %6s / %6s test acc %5.3f and AUC is %5.3f loss %5.3f(%5.3f)" % (i,max_iterations, acc, 0.0,cost_train ,cost_test))
      step +=1
    sess.run(train_step,feed_dict={x:F_train[batch_ind], y_: y_train[batch_ind], keep_prob: dropout})


"""Visualize the soft attention"""
batch_ind_test = np.random.choice(Ntest,batch_size,replace=False)
masks_fetch = sess.run(masks, feed_dict={ x: F_test[batch_ind_test], y_: y_test[batch_ind_test], keep_prob: 1.0})


ind = 11   #Which index to visualize
assert ind < batch_size,'Please provide an index within the batch_size'
bind = batch_ind_test[ind]
fig, axes = plt.subplots(nrows=3, ncols=4)
for p,ax in enumerate(axes.flat):
  if p == 11:
    ax.imshow(np.reshape(X_test[bind],[28,28]).T,origin='upper')
    ax.set_title('Original')
  elif p == 10:
    ax.imshow(np.sum(F_test[bind,:,:,:],axis=2).T)
    ax.set_title('Sum over all input channels')
  else:
    im = ax.imshow(masks_fetch[p][ind,:,:,0].T,vmin=0.0,vmax=0.02)
    ax.set_title('Sequence step %.0f'%p)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
plt.title('Colorbar for first 10 maps')
plt.show()

"""Plot the evolution of cost and accuracy"""
plt.figure()
plt.plot(perf_collect[0,:],label='train acc')
plt.plot(perf_collect[2,:],label='val acc')
plt.legend()
plt.title('Accuracy evolution')
plt.show()

plt.figure()
plt.plot(perf_collect[1,:],label='train cost')
plt.plot(perf_collect[3,:],label='val cost')
plt.legend()
plt.title('Cost evolution')
plt.show()
