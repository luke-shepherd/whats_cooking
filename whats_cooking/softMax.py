


### SOFT MAX CODE ###


import numpy as np
from dataParser import parse_input
from dataParser import split_training_data
import random
# most of this code is written by Arthur Juliani
# https://medium.com/@awjuliani/simple-softmax-in-python-tutorial-d6b4c4ed5c16#.xpni65dgy

def getLoss(w,x,y,lam):
    m = x.shape[0] #First we get the number of training examples
     
    scores = np.dot(x,w) #Then we compute raw class scores given our input and current weights
    prob = softmax(scores) #Next we perform a softmax on these scores to get their probabilities
    loss = (-1 / m) * np.sum(y * np.log(prob)) + (lam/2)*np.sum(w*w) #We then find the loss of the probabilities
    grad = (-1 / m) * np.dot(x.T,(y - prob)) + lam*w #And compute the gradient for that loss
    return loss,grad


def softmax(z):
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z),axis=1)).T
    return sm

def getProbsAndPreds(someX, w):
    probs = softmax(np.dot(someX,w))
    preds = np.argmax(probs,axis=1)
    #print 'probs shape ', probs.shape
    #print 'preds is ', preds.shape
    return probs,preds


def getAccuracy(someX,someY, w):
    prob,prede = getProbsAndPreds(someX,w)
    accuracy = sum(prede == someY)/(float(len(someY)))
    return accuracy


print 'Fetching inputs...'
(classes, ingredients, X, y, y_cuisine, all_classes) = parse_input('train.json')
print 'Loaded inputs'

X = np.insert(X, 0, 1, axis=1)

print 'Splitting data into train and test sets...'
(X_train, X_test, y_train, y_test, y_tr_labels, y_te_labels) = split_training_data(X, y, all_classes, y_cuisine, .75)

te, n = X_train.shape
num_classes = y_train.shape[1]


print 'te: ', te
print 'n: ', n
print 'num_classes: ', num_classes

print '\nTraining loop'

w = np.zeros([n, num_classes])
lam = 1
iterations = 80000
learningRate = 1e-5
batch_size = 100
for i in range(0,iterations):
	X_batch= []
	y_batch = []
	for exnum in np.random.randint(0, high=X_train.shape[0], size=batch_size):
		X_batch.append(X_train[exnum, :])
		y_batch.append(y_train[exnum, :])

	Xin = np.array(X_batch)
	yin = np.array(y_batch)

   	loss,grad = getLoss(w,Xin,yin,lam)
   	if i%200 == 0: 
            print 'loss at iteration ',i,'/', iterations, ': ',loss
            if i%1000 == 0: print 'weight vector: ', w
	w = w - (learningRate * grad)


print 'Training error: ', getAccuracy(X_train, y_tr_labels, w)
print 'Testing error: ', getAccuracy(X_test, y_te_labels, w)
