# perceptron.py
# module to train a perceptron on training data


import dataParser
import numpy as np


# main training method
def train_perceptron(classes, X_train, y_train):

	# create k weight vectors, for our n features
	# initialize first column (bias) to 1
	W = np.zeros((classes.shape[0],y.shape[0] + 1))
	W[:,0] = np.ones((classes.shape[0]))

	"""
	Main Training Loop for Perceptron:

	Iterate through each row of the training data.
	For each row, take the dot product between row and each weight
	in W. We take the weight that gives us the highest valued result as our classifier weight.

	If the predictor is mislabeled, then we decrement the value of this weight, and increase the value
	our of target weight
	""" 

	for j in range(0,X_train.shape[0]):

		# initalize our weight and class prediction
		example 	= X_train[j,:]
		y_target 	= y_train[j]
		w_predictor = W[j,:]
		y_hat 		= y_train[j]
		result 		= w_predictor * np.transpose(example)
		index 		= 0 # keep track of index of best result

		for i in range(1 , W.shape[0]):
			# check for a better result
			tmp_result = W[i,:] * np.transpose(example) 
			if tmp_result > result:
				result 		= tmp_result
				w_predictor = W[i,:]
				y_hat 		= y_train[i]
				index = i

		# check for correction prediction
		if y_hat != y_target:
			W[i,:] = W[i,:] - np.transpose(example) # 
			W[j,:] = W[j,:] + np.transpose(exmaple) # 

	return W

