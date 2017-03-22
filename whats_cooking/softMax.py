import numpy as np
from dataParser import parse_input
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
(all_classes, ingredients, X, y) = parse_input('train.json')
print 'Loaded inputs'

X_train = []
X_test = []
y_train = []
y_test = []
for i in range(0, X.shape[0]):
    r = random.random()
    if r < .75:
        X_train.append(X[i, :])
        y_train.append(y[i, :])
    else:
        X_test.append(X[i, :])
        y_test.append(y[i, :])

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

print 'split data'

print 'train loop'
w = np.zeros([X_train.shape[1], y_train.shape[1]])
lam = 1
iterations = 1000
learningRate = .002
losses = []
for i in range(0,iterations):
    loss,grad = getLoss(w,X_train,y_train,lam)
    print 'loss at iteration ',i,': ',loss     
    losses.append(loss)
    w = w - (learningRate * grad)


y_train_labels = []
y_test_labels = []
print 'formatting y'
for row in y_train:
    x,y = np.where(y_train==1)
    y_train_labels.append(y[0])
    
    
print 'Training error: ', getAccuracy(X_train, y_train_labels, w)
#print 'Testing error: ', getAccuracy(X_test, y_test, w)
