# 29 May, 2019
# Train a 2-layer MLP with task-based modulation (L1 -> L2, L2 -> L3, and their combinations) for object detection tasks
# Works with Python 2.7 and Tensorflow 1.3.0 (CPU version)

from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True) # Import dataset, can change it to 'fMNIST' or 'kMNIST' if need be

import tensorflow as tf
import numpy as np
import scipy
from scipy.ndimage.interpolation import zoom
from random import shuffle
import os  

## Function definitions

def create_permuted_mnist(dat,nums,ord_create=True,filename='order_mat_perm_mnist.npy'): 
# Outputs a dictionary of permuted datasets
# feed the mnist struct to dat, nums is the number of permuted instances to be created, ord_create=True implies a new set of permutations
# are generated and stored in 'filename'. If False, the file in 'filename' is read
  if ord_create:
    order_mat = np.zeros([nums,784])
    for i in range(nums):
      if i == 0:
        order_mat[i,:] = np.arange(784)
      else:
        order_mat[i,:] = np.arange(784)
        shuffle(order_mat[i,:])
    np.save(filename,order_mat)
  else:
    if os.path.exists(filename):
      order_mat = np.load(filename)
    else:
      print('File does not exist!\n')
  dict_mnist = {}
  dict_inst = {}
  dict_inst.update({'train_images':dat.train.images.copy(),'train_labels':dat.train.labels.copy(),'val_images':dat.validation.images.copy(),
    'val_labels':dat.validation.labels.copy(),'test_images':dat.test.images.copy(),'test_labels':dat.test.labels.copy()})
  dict_inst.update({'order':order_mat[0,:]})
  dict_mnist.update({str(0):dict_inst})
  for i in range(nums-1):
    inst_ord = order_mat[i+1,:].astype('int32')
    dict_inst = {}
    dict_inst.update({'order':inst_ord})
    dict_inst.update({'train_images':dat.train.images[:,inst_ord].copy(),'train_labels':dat.train.labels.copy(),
      'val_images':dat.validation.images[:,inst_ord].copy(), 'val_labels':dat.validation.labels.copy(),
      'test_images':dat.test.images[:,inst_ord].copy(),'test_labels':dat.test.labels.copy()})
    dict_mnist.update({str(i+1):dict_inst})
  return dict_mnist#,order_mat

def get_batch(bat_sz,modh,cat,lay,nums,dat,operatn): 
# Outputs a batch of size bat_sz with half samples with positive matches between cues and objects in the images.
# input cat_sel in cat, input mode_h in modh, bat_sz should be even, input lay_sel in lay,
# input number of permutations in nums, input the dataset as dat, input train/val/test as operatn
  x = np.zeros([bat_sz,784])
  y = np.zeros([bat_sz,2])
  yc1 = np.zeros([bat_sz,10*nums+1])
  yc2 = np.zeros([bat_sz,10*nums+1])
  str_images_h = operatn+'_images'
  str_labels_h = operatn+'_labels'
  if modh == 1:
    dat_h = np.int(np.floor((cat-1)*1./10*1.))
    dat_sel_h = np.zeros(bat_sz)
    dat_sel_h[0:bat_sz/2] = dat_h
    dat_sel_h[bat_sz/2:] = np.random.randint(0,nums,bat_sz/2)
    ex_sel_h = np.where(np.argmax(dat[str(np.int(dat_h))][str_labels_h],1)==cat-1)[0]
    shuffle(ex_sel_h)
    x[:bat_sz/2,:] = dat[str(np.int(dat_h))][str_images_h][ex_sel_h[:bat_sz/2],:]
    y[:bat_sz/2,0] = 1.
    for i in range(bat_sz/2):
      ex_sel_inst = np.random.randint(np.shape(dat[str(np.int(dat_sel_h[bat_sz/2+i]))][str_labels_h])[0])
      if dat_sel_h[bat_sz/2+i] == dat_h:
        while np.argmax(dat[str(np.int(dat_sel_h[bat_sz/2+i]))][str_labels_h][ex_sel_inst,:]) == cat-1:
          ex_sel_inst = np.random.randint(np.shape(dat[str(np.int(dat_h))][str_labels_h])[0])
      x[bat_sz/2+i,:] = dat[str(np.int(dat_sel_h[bat_sz/2+i]))][str_images_h][ex_sel_inst,:]
    y[bat_sz/2:,1] = 1.
    yc1[:,-1] = 1.
    yc2[:,-1] = 1.
  elif (modh == 2) or (modh == 3):
    dat_sel_h = np.random.randint(0,nums,bat_sz)
    for i in range(bat_sz/2):
      ex_sel_inst = np.random.randint(np.shape(dat[str(np.int(dat_sel_h[i]))][str_labels_h])[0])
      x[i,:] = dat[str(np.int(dat_sel_h[i]))][str_images_h][ex_sel_inst,:]
      y[i,0] = 1.
      if lay == 1:
        yc1[i,np.int(dat_sel_h[i])*10+np.argmax(dat[str(np.int(dat_sel_h[i]))][str_labels_h][ex_sel_inst,:])] = 1.
        yc2[i,-1] = 1.
      elif lay == 2:
        yc1[i,-1] = 1.
        yc2[i,np.int(dat_sel_h[i])*10+np.argmax(dat[str(np.int(dat_sel_h[i]))][str_labels_h][ex_sel_inst,:])] = 1.
      elif lay == 3:
        yc1[i,np.int(dat_sel_h[i])*10+np.argmax(dat[str(np.int(dat_sel_h[i]))][str_labels_h][ex_sel_inst,:])] = 1.
        yc2[i,np.int(dat_sel_h[i])*10+np.argmax(dat[str(np.int(dat_sel_h[i]))][str_labels_h][ex_sel_inst,:])] = 1.
      elif lay == 4:
        yc1[i,np.int(dat_sel_h[i])*10+np.argmax(dat[str(np.int(dat_sel_h[i]))][str_labels_h][ex_sel_inst,:])] = 1.
        yc1[i,-1] = 1.
        yc2[i,np.int(dat_sel_h[i])*10+np.argmax(dat[str(np.int(dat_sel_h[i]))][str_labels_h][ex_sel_inst,:])] = 1.
    for i in range(bat_sz/2):
      ex_sel_inst = np.random.randint(np.shape(dat[str(np.int(dat_sel_h[bat_sz/2+i]))][str_labels_h])[0])
      x[bat_sz/2+i,:] = dat[str(np.int(dat_sel_h[bat_sz/2+i]))][str_images_h][ex_sel_inst,:]
      y[bat_sz/2+i,1] = 1.
      if lay == 1:
        pos_sel_h = np.random.randint(10*nums)
        while pos_sel_h == np.int(dat_sel_h[bat_sz/2+i])*10+np.argmax(dat[str(np.int(dat_sel_h[bat_sz/2+i]))][str_labels_h][ex_sel_inst,:]):
          pos_sel_h = np.random.randint(10*nums)
        yc1[bat_sz/2+i,pos_sel_h] = 1.
        yc2[bat_sz/2+i,-1] = 1.
      elif lay == 2:
        pos_sel_h = np.random.randint(10*nums)
        while pos_sel_h == np.int(dat_sel_h[bat_sz/2+i])*10+np.argmax(dat[str(np.int(dat_sel_h[bat_sz/2+i]))][str_labels_h][ex_sel_inst,:]):
          pos_sel_h = np.random.randint(10*nums)
        yc2[bat_sz/2+i,pos_sel_h] = 1.
        yc1[bat_sz/2+i,-1] = 1.
      elif lay == 3:
        pos_sel_h = np.random.randint(10*nums)
        while pos_sel_h == np.int(dat_sel_h[bat_sz/2+i])*10+np.argmax(dat[str(np.int(dat_sel_h[bat_sz/2+i]))][str_labels_h][ex_sel_inst,:]):
          pos_sel_h = np.random.randint(10*nums)
        yc2[bat_sz/2+i,pos_sel_h] = 1.
        yc1[bat_sz/2+i,pos_sel_h] = 1.
      elif lay == 4:
        pos_sel_h = np.random.randint(10*nums)
        while pos_sel_h == np.int(dat_sel_h[bat_sz/2+i])*10+np.argmax(dat[str(np.int(dat_sel_h[bat_sz/2+i]))][str_labels_h][ex_sel_inst,:]):
          pos_sel_h = np.random.randint(10*nums)
        yc2[bat_sz/2+i,pos_sel_h] = 1.
        yc1[bat_sz/2+i,-1] = 1.
        yc1[bat_sz/2+i,pos_sel_h] = 1.
  return x,y,yc1,yc2

def mod_images_noise(x,noiser=0.4): # add uniform random noise to images
  x1 = 0.*(x.copy())
  x1 = (1.-noiser)*x.copy() + noiser*np.random.random([np.shape(x)[0],np.shape(x)[1]])
  return x1

def mod_images_normal(x,zoomer=24./28.,noiser=0.2): # add uniform random noise and translation (with scaling) to images
  x1 = x.copy()
  x1 = 0.*x1
  for i in range(np.shape(x)[0]):
    dum1 = zoom(np.reshape(x[i,:],[28,28]),zoomer)
    dum2 = 0.*np.reshape(x[i,:],[28,28])
    dum_x = np.random.randint(np.int(np.floor(28.-zoomer*28.)))
    dum_y = np.random.randint(np.int(np.floor(28.-zoomer*28.)))
    dum2[dum_x:dum_x+np.int(np.floor(zoomer*28.)),dum_y:dum_y+np.int(np.floor(zoomer*28.))] = dum1
    x1[i,:] = (1.-noiser)*np.reshape(dum2,[1,784]) + noiser*np.random.random([1,784])
  return x1

def weight_variable_fc(shape,name):
  initial = tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer()) 
  return initial

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

## Configuring the NN variant

mode_h = 2 # 1 is train on a specified category, 2 is train on all categories with learned modulations, 3 is train on all categories with random modulations
cat_sel = 6 # select category for when mode_h == 1
lay_sel = 3 # 1 for L1->L2 modulation, 2 for L2->L3 modulation, 3 for both L1->L2 & L2->L3 modulation, 4 for L1->L2 on L2->L3 (pre-trained)
cond_r = 0 # 0 for no retrain, 1 for retrain
n_hl1 = 32 # no. of neurons in L2
n_hl2 = 2 # no. of neurons in L3
batch_size = 100
n_perms = 25 # how many mnist permutations to use (1/10 the number of total tasks)
train_dropout = 0.5
g_train = 0. # gain multiplier - set to 0 if gain modulation shouldn't be used
b_train = 1. # bias multiplier - set to 0 if bias modulation shouldn't be used
rand_case = 2 # 1 is cheung et al. like random modulations, 2 is normally-distributed random modulations

str_data_loader = 'order_mat_perm_mnist_'+str(n_perms)+'.npy'
perm_mnist = create_permuted_mnist(mnist,n_perms,False,str_data_loader) # set to False if an existing set is to be used; create datasets once and then reuse them for multiple variants
print('Dataset created')

## Define the NN

x  = tf.placeholder(tf.float32, [None, 28*28], name='x') # Input image - 28x28
y_ = tf.placeholder(tf.float32, [None, n_hl2],  name='y_')
y_c1 = tf.placeholder(tf.float32, [None, 10*n_perms+1],  name='y_c1') # 10*n_perms cats, 1 general (to be used when lay_sel is 1 or 2)
y_c2 = tf.placeholder(tf.float32, [None, 10*n_perms+1],  name='y_c2') # 10*n_perms cats, 1 general (to be used when lay_sel is 1 or 2)
keep_prob  = tf.placeholder(tf.float32)

W_fc1 = weight_variable_fc([28*28, n_hl1],'W_fc1')
b_fc1 = bias_variable([n_hl1])
W_fc2 = weight_variable_fc([n_hl1, n_hl2],'W_fc2')
b_fc2 = bias_variable([n_hl2])
W_fc1_b = weight_variable_fc([10*n_perms+1, n_hl1],'W_fc1_b')
W_x_g = weight_variable_fc([10*n_perms+1, 28*28],'W_x_g')
W_fc2_b = weight_variable_fc([10*n_perms+1, n_hl2],'W_fc2_b')
W_fc1_g = weight_variable_fc([10*n_perms+1, n_hl1],'W_fc1_g')

saver = tf.train.Saver({"W_fc1": W_fc1, "b_fc1": b_fc1, "W_fc2": W_fc2, "b_fc2": b_fc2,
  "W_fc1_b": W_fc1_b, "W_fc2_b": W_fc2_b, "W_x_g": W_x_g, "W_fc1_g": W_fc1_g})
temp = set(tf.all_variables())

x_mod = tf.multiply(x,1.+g_train*tf.matmul(y_c1, W_x_g))
h_fc1 = tf.multiply(tf.nn.relu(tf.matmul(x_mod, W_fc1) + tf.multiply(1.+b_train*tf.matmul(y_c1, W_fc1_b), b_fc1)),1.+g_train*tf.matmul(y_c2, W_fc1_g))
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + tf.multiply(1.+b_train*tf.matmul(y_c2, W_fc2_b), b_fc2), name = 'y')

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ *tf.log(tf.clip_by_value(y,1e-10,1.0)), reduction_indices=[1]))
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

if mode_h == 1:
  lr = 1e-3
  momt = 0.9
  max_steps = 5000001
  train_step = tf.train.MomentumOptimizer(lr,momt).minimize(cross_entropy)
  tf.summary.scalar('cross_entropy_ind', cross_entropy)
  tf.summary.scalar('accuracy_ind', accuracy)
elif mode_h == 2:
  lr = 1e-5
  momt = 0.9
  max_steps = 10000001
  if lay_sel == 4:
    train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy,var_list=[W_fc1_b,W_x_g])
  else:
    train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
  tf.summary.scalar('cross_entropy_all', cross_entropy)
  tf.summary.scalar('accuracy_all', accuracy)
elif mode_h == 3:
  lr = 1e-5
  momt = 0.9
  max_steps = 10000001
  train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy,var_list=[W_fc1, b_fc1, W_fc2, b_fc2])
  tf.summary.scalar('cross_entropy_all_rand', cross_entropy)
  tf.summary.scalar('accuracy_all_rand', accuracy)
merged = tf.summary.merge_all()

## Train the NN

with tf.Session() as sess:

  if mode_h == 1:
    train_str = 'pm_train_ind_' + str(cat_sel)
    val_str = 'pm_val_ind_' + str(cat_sel)
  elif mode_h == 2:
    train_str = 'pm_train_all_' + str(lay_sel) + '_' + str(n_hl1) + "_bg_" + str(b_train) + str(g_train) + '_perms_' + str(n_perms) + '_bs_' + str(batch_size) + '_lr_' + str(lr)
    val_str = 'pm_val_all_' + str(lay_sel) + '_' + str(n_hl1) + "_bg_" + str(b_train) + str(g_train) + '_perms_' + str(n_perms) + '_bs_' + str(batch_size) + '_lr_' + str(lr)
  elif mode_h == 3:
    train_str = 'pm_train_all_rand_' + str(lay_sel) + '_' + str(n_hl1) + "_bg_" + str(b_train) + str(g_train) + '_perms_' + str(n_perms) + '_bs_' + str(batch_size) + '_lr_' + str(lr) +'_randc_' + str(rand_case)
    val_str = 'pm_val_all_rand_' + str(lay_sel) + '_' + str(n_hl1) + "_bg_" + str(b_train) + str(g_train) + '_perms_' + str(n_perms) + '_bs_' + str(batch_size) + '_lr_' + str(lr) +'_randc_' + str(rand_case)
  train_writer = tf.summary.FileWriter(train_str,sess.graph)
  val_writer = tf.summary.FileWriter(val_str)

  if cond_r == 0:
    if mode_h == 2:
      if lay_sel == 4:
        str_h = "./models_new/perm_mnist_" + str(n_perms) + "_hl" + str(n_hl1) + "_allcat" + "_fbcase_" + str(2) + "_bg_" + str(b_train) + str(g_train) + ".cpkt"
        saver.restore(sess,str_h)
        sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))
      else:
        sess.run(tf.global_variables_initializer())
    else:
      sess.run(tf.global_variables_initializer())
  else:
    if mode_h == 1:
      str_h = "./models_new/perm_mnist_" + str(n_perms) + "_hl" + str(n_hl1) + "_cat" + str(cat_sel) + "_base.cpkt"
    elif mode_h == 2:
      str_h = "./models_new/perm_mnist_" + str(n_perms) + "_hl" + str(n_hl1) + "_allcat" + "_fbcase_" + str(lay_sel) + "_bg_" + str(b_train) + str(g_train) + ".cpkt"
    elif mode_h == 3:
      str_h = "./models_new/perm_mnist_" + str(n_perms) + "_hl" + str(n_hl1) + "_allcat" + "_fbcase_rand_" + str(lay_sel) + "_bg_" + str(b_train) + str(g_train) + ".cpkt"
    saver.restore(sess,str_h)
    sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))

  if mode_h == 3:
    if cond_r == 0:
      if rand_case == 1:
      ## Cheung et al. 2019 binary vectors
        dumh11 = np.zeros([10*n_perms+1, n_hl1])
        W_fc1_b.load(dumh11, sess)
        dumh11 = np.random.randint(0,2,[10*n_perms+1, 28*28])*2.-2.
        W_x_g.load(dumh11, sess)
        dumh11 = np.zeros([10*n_perms+1, n_hl2])
        W_fc2_b.load(dumh11, sess)
        dumh11 = np.random.randint(0,2,[10*n_perms+1, n_hl1])*2.-2.
        W_fc1_g.load(dumh11, sess)
      elif rand_case == 2:
      ## Random vectors sampled from random normal distribution 
        dumh11 = np.random.randn(10*n_perms+1, n_hl1)
        W_fc1_b.load(dumh11, sess)
        dumh11 = np.random.randn(10*n_perms+1, 28*28)
        W_x_g.load(dumh11, sess)
        dumh11 = np.random.randn(10*n_perms+1, n_hl2)
        W_fc2_b.load(dumh11, sess)
        dumh11 = np.random.randn(10*n_perms+1, n_hl1)
        W_fc1_g.load(dumh11, sess)

  for step in range(max_steps):

    batch_xs, batch_ys, batch_yc1, batch_yc2 = get_batch(batch_size,mode_h,cat_sel,lay_sel,n_perms,perm_mnist,'train')
    batch_xs = mod_images_normal(batch_xs)

    if (step % 200) == 0: 
      summary,dum1,dum2 = sess.run([merged,accuracy,cross_entropy], feed_dict={x: batch_xs, y_: batch_ys, y_c1: batch_yc1, y_c2: batch_yc2, keep_prob: 1.})
      train_writer.add_summary(summary, step)
      print('Train_acc: ',step, dum1, dum2)

    if (step % 1000) == 0:
      batch_xsv, batch_ysv, batch_ysvc1, batch_ysvc2 = get_batch(800,mode_h,cat_sel,lay_sel,n_perms,perm_mnist,'val')
      batch_xsv = mod_images_normal(batch_xsv)
      summary,dum1,dum2 = sess.run([merged,accuracy,cross_entropy], feed_dict={x: batch_xsv, y_: batch_ysv, y_c1: batch_ysvc1, y_c2: batch_ysvc2, keep_prob: 1.})
      val_writer.add_summary(summary, step)
      print('Val_acc: ',step, dum1, dum2)

    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, y_c1: batch_yc1, y_c2: batch_yc2, keep_prob: train_dropout})

  if mode_h == 1:
    batch_xst, batch_yst, batch_ystc1, batch_ystc2 = get_batch(1600,mode_h,cat_sel,lay_sel,n_perms,perm_mnist,'test')
    batch_xst = mod_images_normal(batch_xst)
    summary,dum1,dum2 = sess.run([merged,accuracy,cross_entropy], feed_dict={x: batch_xst, y_: batch_yst, y_c1: batch_ystc1, y_c2: batch_ystc2, keep_prob: 1.})
    print('Test_acc: ',max_steps, dum1, dum2)
  elif (mode_h == 2) or (mode_h == 3):
    batch_xst, batch_yst, batch_ystc1, batch_ystc2 = get_batch(10000,mode_h,cat_sel,lay_sel,n_perms,perm_mnist,'test')
    batch_xst = mod_images_normal(batch_xst)
    summary,dum1,dum2 = sess.run([merged,accuracy,cross_entropy], feed_dict={x: batch_xst, y_: batch_yst, y_c1: batch_ystc1, y_c2: batch_ystc2, keep_prob: 1.})
    print('Test_acc: ',max_steps, dum1, dum2)
    for i in range(10):
      batch_xst, batch_yst, batch_ystc1, batch_ystc2 = get_batch(1600,1,i+1,lay_sel,n_perms,perm_mnist,'test')
      batch_xst = mod_images_normal(batch_xst)
      if lay_sel == 1:
        batch_ystc1 = 0.*batch_ystc1
        batch_ystc1[:,i] = 1.
        acc_h = sess.run(accuracy, feed_dict={x: batch_xst, y_: batch_yst, y_c1: batch_ystc1, y_c2: batch_ystc2, keep_prob: 1.})
        print('Cat:',i+1,'Test_acc: ',acc_h)
      elif lay_sel == 2:
        batch_ystc2 = 0.*batch_ystc2
        batch_ystc2[:,i] = 1.
        acc_h = sess.run(accuracy, feed_dict={x: batch_xst, y_: batch_yst, y_c1: batch_ystc1, y_c2: batch_ystc2, keep_prob: 1.})
        print('Cat:',i+1,'Test_acc: ',acc_h)
      elif lay_sel == 3:
        batch_ystc2 = 0.*batch_ystc2
        batch_ystc2[:,i] = 1.
        batch_ystc1 = 0.*batch_ystc1
        batch_ystc1[:,i] = 1.
        acc_h = sess.run(accuracy, feed_dict={x: batch_xst, y_: batch_yst, y_c1: batch_ystc1, y_c2: batch_ystc2, keep_prob: 1.})
        print('Cat:',i+1,'Test_acc: ',acc_h)

## Save the NN

  saver = tf.train.Saver({"W_fc1": W_fc1, "b_fc1": b_fc1, "W_fc2": W_fc2, "b_fc2": b_fc2,
    "W_x_g": W_x_g, "W_fc1_b": W_fc1_b, "W_fc1_g": W_fc1_g, "W_fc2_b": W_fc2_b},write_version=tf.train.SaverDef.V1)

  if mode_h == 1:
    str_h = "./models_new/perm_mnist_" + str(n_perms) + "_hl" + str(n_hl1) + "_cat" + str(cat_sel) + "_base.cpkt"
  elif mode_h == 2:
    str_h = "./models_new/perm_mnist_" + str(n_perms) + "_hl" + str(n_hl1) + "_allcat" + "_fbcase_" + str(lay_sel) + "_bg_" + str(b_train) + str(g_train) + ".cpkt"
  elif mode_h == 3:
    str_h = "./models_new/perm_mnist_" + str(n_perms) + "_hl" + str(n_hl1) + "_allcat" + "_fbcase_rand_" + str(lay_sel) + "_bg_" + str(b_train) + str(g_train) +'_randc_' + str(rand_case) + ".cpkt"
  save_path = saver.save(sess, str_h)

