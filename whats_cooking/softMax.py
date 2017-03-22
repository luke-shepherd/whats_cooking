import numpy as np
from dataParser import parse_input

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

def getProbsAndPreds(someX):
    probs = softmax(np.dot(someX,w))
    preds = np.argmax(probs,axis=1)
    return probs,preds



(all_classes, ingredients, X, y) = parse_input('train.json')
print 'Loaded inputs'

w = np.zeros([X.shape[1], y.shape[1])
lam = 1
iterations = 1000
learningRate = 1e-5
losses = []
for i in range(0,iterations):
    loss,grad = getLoss(w,X,y,lam)
    print 'loss at iteration ',i,': ',loss
    losses.append(loss)
    w = w - (learningRate * grad)
