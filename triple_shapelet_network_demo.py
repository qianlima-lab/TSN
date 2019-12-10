import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.utils import np_utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from matplotlib.pyplot import savefig

def transfer_labels(labels):
	indexes = np.unique(labels)
	num_classes = indexes.shape[0]
	num_samples = np.shape(labels)[0]
	for i in range(num_samples):
		new_label = np.argwhere(indexes == labels[i])[0][0]
		labels[i] = new_label
	labels = np_utils.to_categorical(labels, num_classes)
	return labels, num_classes

def open_file(path_data,dataset, info):
	data_x = []
	data_y = []
	count = 0;
	for line in open(path_data):
		count = count + 1
		row = [[np.float32(x)] for x in line.strip().split(',')]
		label = np.int32(row[0])
		row = np.array(row[1:])
		row_shape = np.shape(row)
		row_mean = np.mean(row[0:])
		data_x.append(row-np.kron(np.ones((row_shape[0],row_shape[1])),row_mean))
		data_y.append(label[0])	
	return data_x, data_y
	
def loading_ucr(index):
	dir_path = './UCR_TS_Archive_2015'
	list_dir = os.listdir(dir_path)
	dataset = list_dir[index]	
	train_data = dir_path+'/'+dataset+'/'+dataset+'_TRAIN'
	test_data = dir_path+'/'+dataset+'/'+dataset+'_TEST'
	train_x, train_y = open_file(train_data,dataset,'train')
	test_x, test_y   = open_file(test_data,dataset,'test')
	return train_x, train_y, test_x, test_y, dataset

dir_path = './UCR_TS_Archive_2015'
list_dir = os.listdir(dir_path)
index = list_dir.index('ECGFiveDays')
train_x, train_y, test_x, test_y, dataset_name = loading_ucr(index=index)
nb_train, len_series =  np.shape(train_x)[0], np.shape(train_x)[1]
nb_test = np.shape(test_x)[0]
train_x = np.reshape(train_x,[nb_train,len_series])
test_x = np.reshape(test_x,[nb_test,len_series])
train_y, nb_class = transfer_labels(train_y)
test_y, _ = transfer_labels(test_y)
learning_rate=0.001
batch_size=15
nb_shapelet=90
nb_epoch=200
dropout=0.75
nb_shapelet_per=np.int(np.ceil(np.float(nb_shapelet)/nb_class))
nb_shapelet_cls=nb_shapelet_per*nb_class
len_prop=[0.7,0.8]
len_shapelet=[np.int(len_prop[0]*len_series),np.int(len_prop[1]*len_series)]
nb_slice = [len_series-i+1 for i in len_shapelet]
dim_cls=nb_shapelet_per*len(len_shapelet)
lam=1

nb_batch_train = np.int32(np.floor(nb_train/batch_size))
if nb_train%batch_size==0:
	nb_total = nb_train
else:
	nb_total = np.int32((nb_batch_train+1)*batch_size)
delta = np.int32(nb_total - nb_train)

nb_batch_test = np.int32(np.floor(nb_test/batch_size))
if nb_test%batch_size==0:
	nb_total_test = nb_test
else:
	nb_total_test = np.int32((nb_batch_test+1)*batch_size)
delta_test = np.int32(nb_total_test - nb_test)
test_data_full = np.zeros((nb_total_test, len_series))
test_label_full = np.zeros((nb_total_test, nb_class))
for m in range(nb_test):
	test_data_full[m,:] = test_x[m,:]
	test_label_full[m,:] = test_y[m,:]
for m in range(delta_test):
	test_data_full[nb_test+m,:] = test_x[m,:]
	test_label_full[nb_test+m,:] = test_y[m,:]
test_slice_full_0 = np.zeros((nb_total_test, len_shapelet[0], nb_slice[0]))
test_slice_full_1 = np.zeros((nb_total_test, len_shapelet[1], nb_slice[1]))
for m in range(nb_total_test):
	for k in range(nb_slice[0]):
		test_slice_full_0[m,:,k]=test_data_full[m,k:k+len_shapelet[0]]
for m in range(nb_total_test):
	for k in range(nb_slice[1]):
		test_slice_full_1[m,:,k]=test_data_full[m,k:k+len_shapelet[1]]

with tf.name_scope('input_layer'):
	x=tf.placeholder(tf.float32,[None,len_series])
	y=tf.placeholder(tf.float32,[None,nb_class])
	keep_prob=tf.placeholder(tf.float32)
	x_full_slice_0=tf.placeholder(tf.float32,[None,len_shapelet[0],nb_slice[0]])
	x_full_slice_1=tf.placeholder(tf.float32,[None,len_shapelet[1],nb_slice[1]])
	x_slice_reshape_0=tf.reshape(x_full_slice_0,shape=[-1,len_shapelet[0],1,nb_slice[0]])
	x_slice_reshape_1=tf.reshape(x_full_slice_1,shape=[-1,len_shapelet[1],1,nb_slice[1]])

with tf.name_scope('General_shapelet'):
	general_shapelet_0=tf.Variable(tf.truncated_normal([1,len_shapelet[0],1,nb_shapelet],stddev=0.001))
	general_shapelet_1=tf.Variable(tf.truncated_normal([1,len_shapelet[1],1,nb_shapelet],stddev=0.001))

with tf.name_scope('class_specific_shapelet'):
	class_shapelet_0=tf.Variable(tf.truncated_normal([1,len_shapelet[0],1,nb_shapelet_cls],stddev=0.001))
	class_shapelet_1=tf.Variable(tf.truncated_normal([1,len_shapelet[1],1,nb_shapelet_cls],stddev=0.001))

with tf.name_scope('sample_specific_shapelet'):	
	with tf.name_scope('Conv_Shapelet0_Generator'):
		w_gen0 = tf.Variable(tf.truncated_normal([3,1,nb_slice[0],nb_shapelet],stddev=0.001))
		b_gen0 = tf.Variable(tf.constant(0.1, shape=[nb_shapelet]))
		sample_shapelet_0=tf.nn.conv2d(x_slice_reshape_0,w_gen0,strides=[1,1,1,1],padding='SAME')
		sample_shapelet_0=tf.nn.bias_add(sample_shapelet_0, b_gen0)
	with tf.name_scope('Conv_Shapelet1_Generator'):
		w_gen1 = tf.Variable(tf.truncated_normal([3,1,nb_slice[1],nb_shapelet],stddev=0.001))
		b_gen1 = tf.Variable(tf.constant(0.1, shape=[nb_shapelet]))
		sample_shapelet_1=tf.nn.conv2d(x_slice_reshape_1,w_gen1,strides=[1,1,1,1],padding='SAME')
		sample_shapelet_1=tf.nn.bias_add(sample_shapelet_1, b_gen1)

with tf.name_scope('shapelet_transformation'):
	with tf.name_scope('shapelet_0'):
		x_slice_0=tf.reshape(x_full_slice_0,shape=[-1,len_shapelet[0],nb_slice[0],1])
		x_slice_0=tf.tile(x_slice_0,[1,1,1,2*nb_shapelet+nb_shapelet_cls])
		shapelet_0=tf.concat([tf.tile(class_shapelet_0,[batch_size,1,1,1]),tf.tile(general_shapelet_0,[batch_size,1,1,1]),sample_shapelet_0],3)
		shapelet_0=tf.tile(shapelet_0,[1,1,nb_slice[0],1])
		shapelet_transform_0=tf.reduce_min(tf.sqrt(tf.reduce_sum(tf.square(x_slice_0-shapelet_0),1)),1)
	with tf.name_scope('shapelet_1'):
		x_slice_1=tf.reshape(x_full_slice_1,shape=[-1,len_shapelet[1],nb_slice[1],1])
		x_slice_1=tf.tile(x_slice_1,[1,1,1,2*nb_shapelet+nb_shapelet_cls])
		shapelet_1=tf.concat([tf.tile(class_shapelet_1,[batch_size,1,1,1]),tf.tile(general_shapelet_1,[batch_size,1,1,1]),sample_shapelet_1],3)
		shapelet_1=tf.tile(shapelet_1,[1,1,nb_slice[1],1])
		shapelet_transform_1=tf.reduce_min(tf.sqrt(tf.reduce_sum(tf.square(x_slice_1-shapelet_1),1)),1)
	shapelet_transform=tf.concat([shapelet_transform_0,shapelet_transform_1],1)
	fc1=shapelet_transform
	
with tf.name_scope('auxiliary_classifier'):
	cls_out=[]
	for i in range(nb_class):
		w_h = tf.Variable(tf.truncated_normal([dim_cls,1]))
		b_h = tf.Variable(tf.constant(0.1, shape=[1]))
		cls_h_i = tf.concat([shapelet_transform_0[:,i*nb_shapelet_per:(i+1)*nb_shapelet_per],shapelet_transform_1[:,i*nb_shapelet_per:(i+1)*nb_shapelet_per]],1)
		cls_out_i=tf.add(tf.matmul(cls_h_i,w_h),b_h)
		cls_out.append(cls_out_i)
	cls_out=tf.concat(cls_out,1)
	cls_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=cls_out, labels=y))

with tf.name_scope('out'):
	with tf.name_scope('weights'):
		wout = tf.Variable(tf.truncated_normal([4*nb_shapelet+2*nb_shapelet_cls,nb_class]))
	with tf.name_scope('bias'):
		bout = tf.Variable(tf.constant(0.1, shape=[nb_class]))
	fc1=tf.nn.dropout(fc1,keep_prob)
	out_tmp1=tf.add(tf.matmul(fc1,wout),bout)
	out_tmp2=tf.nn.softmax(out_tmp1)

with tf.name_scope('model_loss'):
	model_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_tmp1,labels=y))
	loss=model_loss+lam*cls_loss

with tf.name_scope('train'):
	optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

with tf.name_scope('accuracy'):
	correct_pred=tf.equal(tf.argmax(out_tmp2,1),tf.argmax(y,1))
	accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init = tf.global_variables_initializer()

gpu_options = tf.GPUOptions(allow_growth=True)
saver = tf.train.Saver()
test_accuracy_collect=[]
train_accuracy_collect=[]
train_loss_collect=[]
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
	sess.run(init)
	for i in range(nb_epoch):
		print 'Epoch:'+str(i)+'/'+str(nb_epoch)
		train_data_full = np.zeros((nb_total, len_series))
		train_label_full = np.zeros((nb_total, nb_class))
		L_train = [x_train for x_train in range(nb_train)]
		np.random.shuffle(L_train)
		for m in range(nb_train):
			train_data_full[m,:] = train_x[L_train[m],:]
			train_label_full[m,:] = train_y[L_train[m],:]
		for m in range(delta):
			train_data_full[nb_train+m,:] = train_data_full[m,:]
			train_label_full[nb_train+m,:] = train_label_full[m,:]
		train_slice_full_0 = np.zeros((nb_total, len_shapelet[0], nb_slice[0]))
		train_slice_full_1 = np.zeros((nb_total, len_shapelet[1], nb_slice[1]))
		for m in range(nb_total):
			for k in range(nb_slice[0]):
				train_slice_full_0[m,:,k]=train_data_full[m,k:k+len_shapelet[0]]
			for k in range(nb_slice[1]):
				train_slice_full_1[m,:,k]=train_data_full[m,k:k+len_shapelet[1]]
		train_accuracy_tmp=[]
		train_loss_tmp=[]
		for j in range(nb_total/batch_size):
			batch_x = train_data_full[j*batch_size:(j+1)*batch_size]
			batch_y = train_label_full[j*batch_size:(j+1)*batch_size]
			batch_slice_0 = train_slice_full_0[j*batch_size:(j+1)*batch_size]
			batch_slice_1 = train_slice_full_1[j*batch_size:(j+1)*batch_size]
			sess.run(optimizer,feed_dict={x:batch_x,y:batch_y,x_full_slice_0:batch_slice_0,x_full_slice_1:batch_slice_1,keep_prob:dropout})
			los, acc = sess.run([model_loss, accuracy], feed_dict={x: batch_x,y: batch_y,x_full_slice_0:batch_slice_0,x_full_slice_1:batch_slice_1,keep_prob: 1.0})
			train_accuracy_tmp.append(acc)
			train_loss_tmp.append(los)
			print("epoch: " + str(i) + ", batch: " + str(j) + ", Accuracy= " + "{:.5f}".format(acc))	
		train_accuracy_collect.append(np.mean(train_accuracy_tmp))
		train_loss_collect.append(np.mean(train_loss_tmp))
		test_accuracy_tmp=[]
		for j in range(nb_total_test/batch_size):
			prediction_acc=sess.run(correct_pred,feed_dict={x:test_data_full[j*batch_size:(j+1)*batch_size],y:test_label_full[j*batch_size:(j+1)*batch_size],x_full_slice_0:test_slice_full_0[j*batch_size:(j+1)*batch_size],x_full_slice_1:test_slice_full_1[j*batch_size:(j+1)*batch_size],keep_prob: 1.0})
			test_accuracy_tmp=np.concatenate((test_accuracy_tmp,prediction_acc))
		test_accuracy=np.mean(test_accuracy_tmp[0:nb_test])
		test_accuracy_collect.append(test_accuracy)
		print("testing accuracy:",test_accuracy)
	saver.save(sess, './save/model.ckpt')
print("min tain loss accuracy:",test_accuracy_collect[train_loss_collect.index(min(train_loss_collect))])
print('batch_size:',batch_size)
print('nb_shapelet:',nb_shapelet)
print('len_shapelet:',len_prop)
print('dataset:',list_dir[index])
plt.figure(figsize=(9,4))
plt.plot(train_accuracy_collect,linewidth=0.5)
plt.plot(test_accuracy_collect,linewidth=0.5)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_acc', 'test_acc'], loc='upper left')
savefig('./plot/'+dataset_name+'_acc.pdf')
plt.show()
