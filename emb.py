from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import math
import tensorflow as tf
import numpy as np
import facenet

def main():
	with tf.Graph().as_default():
		with tf.Session() as sess:
			data_set = facenet.get_dataset("../../datasets/lfw/lfw_mtcnnpy_160")
			paths, label_list = facenet.get_image_paths_and_labels(data_set)
			
			paths = paths[0:5] # just 5 pictures for speed
			
			print("Loading model")
			facenet.load_model("20170511-185253")
			images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
			embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
			phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
			
			image_size = 160
			embedding_size = embeddings.get_shape()[1]
			
			print('Runnning forward pass')
			batch_size = 10
			nrof_images = len(paths)
			nrof_batches = int(math.ceil(1.0*nrof_images / batch_size))
			emb_array = np.zeros((nrof_images, embedding_size))
			
			for i in range(nrof_batches):
				start_index = i*batch_size
				end_index = min((i+1)*batch_size, nrof_images)
				paths_batch = paths[start_index:end_index]
				images = facenet.load_data(paths_batch, False, False, image_size)
				feed_dict = { images_placeholder:images, phase_train_placeholder:False }
				emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
			
			n_images = len(paths)
			print('Distance matrix')
			print('    ', end='')
			for i in range(n_images):
				print('    %1d     ' % i, end='')
			print('')
			for i in range(n_images):
				print('%1d  ' % i, end='')
				for j in range(n_images):
					dist = np.sqrt(np.sum(np.square(np.subtract(emb_array[i,:], emb_array[j,:]))))
					print('  %1.4f  ' % dist, end='')
				print('')	
	
if __name__ == '__main__':
    main()