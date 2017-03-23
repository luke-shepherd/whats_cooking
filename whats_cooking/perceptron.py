# perceptron.py
# module to train a perceptron on training data


import dataParser
import numpy as np
import sys
import pdb

ITERATIONS = 1


# classification
# take a test matrix, trained weights and classify results
# ith weight with highest result is index of cuisine
def classify(X_test,W,all_classes):
	""" 
	Iterate through each example, and find weight value with
	highest result. use index of this weight value to index cuisine
	from y_cuisine
	"""
	# insert a bias column into training examples
	X_test = np.insert( X_test, 0,1, 1)

	print "Now we classifying..."
	classifications = []
	for i in range(0,X_test.shape[0]):

		example 	= X_test[i,:] # ith row-example of feature matrix
		w_predictor = W[0,:] # initialize the weight we think will yield highest value
		w_index 	= 0 	 # maintain its index
		result 		= 0 # calc result of weight and example
		y_hat 		= all_classes[0] # initialize our prediction

		for j in range(0,W.shape[0]):
			tmp_result = np.dot(w_predictor, np.transpose(example))
			if tmp_result >= result:
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
	for i in range(0,len(targets)):
		print targets[i], predictors[i]
		if targets[i] == predictors[i]:
			correct += 1
	print correct
	total = len(targets)
	print total

	return float(correct)/float(total)
	



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
	n = 1
	for k in xrange(ITERATIONS):
		if True:
			print "Training Iteration: ", k
		for j in range(0,X_train.shape[0]):

			example 	= X_train[j,:] # get the jth example
			y_target 	= y_cuisine[j] # cuisine target is jth cuisine
			w_index 	= 0 # assume associated weight vector is first one
			w_predictor = W[w_index,:] # weight vector
			result 		= 0
			y_hat 		= y_cuisine[w_index] # initialize our prediction
			index 		= 0 		   # index for wrong weights


			# loop through all weights, to find highest value one
			for i in range(0, W.shape[0]):

				# check for a better result
				tmp_result = np.dot(W[i,:], example) 
				if  tmp_result >= result:
					result 		= tmp_result
					w_predictor = W[i,:]
					y_hat 		= y_cuisine[i]
					index = i

			# check for correction prediction
			if y_hat != y_target:
				#pdb.set_trace()
				W[index,:] -=example# 
				W[w_index,:] += example # 
		k += 1

	return W



def main():
	if len(sys.argv) <2: quit()
	classes, ingredients, X, y, y_cuisine, all_classes = dataParser.parse_input(sys.argv[1])
	W = train_perceptron(classes,X,y, y_cuisine, all_classes)

	predictions = classify(X,W,all_classes)
	accuracy = compute_accuracy(y_cuisine,predictions)
	print "\nThe accuracy of this model is %.4f" % accuracy

if __name__ == '__main__':
	main()


