# perceptron.py
# module to train a perceptron on training data


import dataParser
import numpy as np


# main training method
def train_perceptron(classes, X_train, y_train, y_cuisine,all_classes):

    # insert a bias column into training examples
	X_train = np.insert( X_train, 0,1, 1)

	# create k weight vectors, for our n features
	# initialize first column (bias) to 1
	W = np.zeros((classes.shape[0],X_train.shape[1]))
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

		
		example 	= X_train[j,:] # get the jth example
		y_target 	= y_cuisine[j] # cuisine target is jth cuisine
		w_index 	= all_classes.index(y_target) # index of associated weight vector
		w_predictor = W[w_index,:] # weight vector
		result 		= np.dot(w_predictor, np.transpose(example))
		y_hat 		= y_cuisine[j] # initialize our prediction
		index 		= j 		   # index for wrong weights


		# loop through all weights, to find highest value one
		for i in range(1, W.shape[0]):

			# check for a better result
			tmp_result = np.dot(W[i,:], np.transpose(example)) 
			if  tmp_result >= result:
				result 		= np.dot(W[i,:], np.transpose(example)) 
				w_predictor = W[i,:]
				y_hat 		= y_cuisine[i]
				index = i

		# check for correction prediction
		if not np.array_equiv(y_hat, y_target):
			W[index,:] = W[index,:] - np.transpose(example) # 
			W[w_index,:] = W[w_index,:] + np.transpose(example) # 

	return W

classes, ingredients, X, y, y_cuisine, all_classes = dataParser.parse_input('train.json')
print train_perceptron(classes,X,y, y_cuisine, all_classes)

