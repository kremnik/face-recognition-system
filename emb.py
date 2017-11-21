from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import math
import tensorflow as tf
import numpy as np
import facenet
from sklearn.manifold import TSNE # for t-SNE dimension reduction
from sklearn.decomposition import PCA # for PCA dimension reduction

def main():
	with tf.Graph().as_default():
		with tf.Session() as sess:
			data_set = facenet.get_dataset("../../datasets/ownpeople/ownpeople_mtcnnpy_160")
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
			threshold = 1.0
			print('Distance matrix')
			print('    ', end='')
			for i in range(len(paths)):
				print('    %3d     ' % i, end='')
			print('')
			for i in range(len(paths)):
				print('%1d  ' % i, end='')
				for j in range(len(paths)):
					dist = np.linalg.norm(emb_array[i,:] - emb_array[j,:])
					print('  %1.4f' % dist, end='')
					if (label_list[i] == label_list[j]) and (dist <= threshold):
						print("(TP)", end='')
					elif (label_list[i] != label_list[j]) and (dist > threshold):
						print("(TN)", end='')
					elif (label_list[i] == label_list[j]) and (dist > threshold):
						print("(FN)", end='')
					elif (label_list[i] != label_list[j]) and (dist <= threshold):
						print("(FP)", end='')
				print('')

			# t-SNE dimension reduction
			print('')
			print("t-SNE dimension reduction")
			emb_tsne = TSNE(n_components=2).fit_transform(emb_array)
			print(emb_tsne)
			
			# PCA dimension reduction
			print('')
			print("PCA dimension reduction")
			pca = PCA(n_components=5) # number of components must be either less or equal to number of training (or test) examples
			emb_pca = pca.fit(emb_array).transform(emb_array)
			print(emb_pca)
			
			# Fisher's discriminant
			print('')
			print('Fisher\'s discriminant')
			E = np.linalg.inv(np.cov(emb_pca.T))
			num = 7
			y = emb_pca[num,:] # separate y from others
			
			emb_del = np.delete(emb_pca, (num), axis=0)
			
			D = (y - (1 / (len(emb_pca) - 1) * np.sum(emb_del)))
			w = np.dot(E, D)
			c = np.dot(w / np.linalg.norm(w), y)
			
			lng = len(emb_pca)
			l = []
			for i in range(lng - 1):
				l.append(np.dot(w / np.linalg.norm(w), emb_del[i,:]) - c)
				
			label_del = np.delete(label_list, (num), axis=0)
			
			label = label_list[num]
			print("class:", label)
			
			for i in range(lng-1):
				if label != label_del[i] and l[i] < 0:
					mark = "TP"
				elif label != label_del[i] and l[i] >= 0:
					mark = "FP"
				elif label == label_del[i] and l[i] < 0:
					mark = "FN"
				elif label == label_del[i]  and l[i] >= 0:
					mark = "TN"
				print(l[i], label_del[i], mark)
	
if __name__ == '__main__':
    main()