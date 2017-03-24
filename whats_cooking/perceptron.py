# perceptron.py
# module to train a perceptron on training data


import dataParser
import numpy as np
import sys
import pdb
import random
from sets import Set

ITERATIONS 		= 100000
RATIO 			= .1
BATCH_SIZE		= 100
LEARNING_RATE	= .01                     
DIM 			= 2


# PCA
def PCA(X):
	"""
	center the data
	"""

	print "Principal Component Analysis"
	for i in range(0,X.shape[1]):
		total = np.sum(X[:,i])
		average = total / X.shape[0]
		X[:,i] = np.subtract(X[:,i],average)

	cov_matrix = np.matmul(np.transpose(X),X)

	# get eigen values/vectors
	e_vals, e_vecs = np.linalg.eig(cov_matrix)
	U_reduced = np.zeros((e_vecs.shape[0],DIM))
	best_vals = np.argsort(e_vals)


	for i in range(0,DIM):
		eval_index = best_vals[i]
		U_reduced[:,i] = np.add(U_reduced[:,i],e_vecs[:,eval_index])

	phi = np.matmul(X,U_reduced)

	new_X = np.matmul(phi,np.transpose(U_reduced))

	return new_X





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
	output = open("results.txt",'a')
	for i in range(0,len(targets)):
		##print targets[i], predictors[i]
		output.write("targets: %s 	predictor: %s" % (targets[i],predictors[i]))
		if targets[i] == predictors[i]:
			correct += 1
	total = len(targets)
	output.close()

	return correct, total
	



# main training method
def train_perceptron(classes, X_train, y_cuisine,all_classes):
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
		if (k % 1000 == 0):
			print "Training Iteration: ", k

		batch_indicies = random.sample(xrange(X_train.shape[0]),BATCH_SIZE) # list of indices to train on
		for j in batch_indicies:

			example 	= X_train[j,:] # get the jth example
			y_target 	= y_cuisine[j] # cuisine target is jth cuisine
			w_index 	= all_classes.index(y_target) # weight vector that should return highest value
			result 		= 0
			y_hat 		= all_classes[w_index] # initialize our prediction
			index 		= 0 		   # index highest valued weight result


			# loop through all weights, to find highest value one
			for i in range(0, W.shape[0]):

				# check for a better result
				tmp_result = np.dot(W[i,:], example) 
				if  tmp_result >= result:
					result 		= tmp_result
					y_hat 		= all_classes[i]
					index = i

			# check for correction prediction
			if y_hat != y_target:
				#pdb.set_trace()
				W[index,:] -= np.multiply(LEARNING_RATE, example) # 
				W[w_index,:] += np.multiply(LEARNING_RATE, example) # 
		k += 1

	return W



def main():
	output = open("traingResults.txt",'a')

	if len(sys.argv) <2: quit()
	classes, ingredients, X, y, y_cuisine, all_classes = dataParser.parse_input(sys.argv[1])
	x_train, y_train = dataParser.split_data(X,y_cuisine,RATIO)

	# size = x_train.shape[1] # get number of features in x
	# cols_to_get = random.sample(xrange(size),size/4)
	# x_train = x_train[:,cols_to_get]
	#x_train = PCA(x_train)

	W = train_perceptron(classes, x_train,y_cuisine,all_classes)


	predictions = classify(x_train,W,all_classes)
	correct, total = compute_accuracy(y_train,predictions)	

	print "\nITERATIONS : %d\nRATIO : %.2f\nBATCH_SIZE %d\nLEARNING_RATE : %.2f\nDIMENSIONS: %d" % (ITERATIONS,RATIO,BATCH_SIZE,LEARNING_RATE,DIM)
	print "\nThere were %d correct classifications at of %d" % (correct,total)
	accuracy = float(correct)/float(total)
	print "\nThe accuracy of this training model is %.4f" % accuracy

	output.close()

if __name__ == '__main__':
	main()


