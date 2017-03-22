# perceptron.py
# module to train a perceptron on training data


import dataParser
import numpy as np

# classification
# take a test matrix, trained weights and classify results
# ith weight with highest result is index of cuisine
def classify(X_test,W,all_classes):
	""" 
	Iterate through each example, and find weight value with
	highest result. use index of this weight value to index cuisine
	from y_cuisine
	"""

	print "Now we classifying..."
	classifications = []
	for i in range(0,X_test.shape[0]):

		w_predictor = W[0,:]
		w_index 	= 0
		result 		= np.dot(w_predictor, np.transpose(W[0,:]))
		y_hat 		= all_classes[0] # initialize our prediction
		index 		= 0 		   # index for wrong weights

		for j in range(1,W.shape[0]):
			tmp_result = np.dot(w_predictor, np.transpose(W[0,:]))
			if tmp_result > result:
				result = tmp_result
				w_predictor = W[j,:]
				y_hat = all_classes[j]
		classifications.append(y_hat)
	return np.array(classifications)


# compute number of correctly classified results
# compare two numpy arrays
def compute_accuracy(targets, predictors):
	print "Compute that accuracy..."
	correct = 0
	for i in range(0,targets.shape[0]):
		print targets[i], predictors[i]
		if targets[i] == predictors[i]:
			correct += 1
	print correct
	total = targets.shape[0]

	return correct/total
	



# main training method
def train_perceptron(classes, X_train, y_train, y_cuisine,all_classes):
	print "Train that perceptron..."

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
				result 		= tmp_result
				w_predictor = W[i,:]
				y_hat 		= y_cuisine[i]
				index = i

		# check for correction prediction
		if not np.array_equiv(y_hat, y_target):
			W[index,:] = W[index,:] - np.transpose(example) # 
			W[w_index,:] = W[w_index,:] + np.transpose(example) # 

	return W



def main():
	classes, ingredients, X, y, y_cuisine, all_classes = dataParser.parse_input('train.json')
	W = train_perceptron(classes,X,y, y_cuisine, all_classes)

	predictions = classify(X,W,all_classes)
	accuracy = compute_accuracy(y,predictions)
	print "\nThe accuracy of this model is %.4f" % accuracy

if __name__ == '__main__':
	main()


