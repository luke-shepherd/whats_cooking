import json
from pprint import pprint
from sets import Set
import numpy as np
import random

RATIO = 0.75


# method to parse data from json file
def parse_input(filename):
    with open(filename) as data_file:  

        data = json.load(data_file)

	# data structures
        all_classes = [] # target values
        all_ingredients = [] # all features
        NUM_EXAMPLES = len(data)

	# iterate over the cuisines
	# add cuisine to list if not seen yet
        for i in range(0, NUM_EXAMPLES):
            cuisine = data[i]['cuisine']
            if cuisine not in all_classes:
                all_classes.append(cuisine)

	    # grab all the ingredients from json file
 	    # for cuisine
            ingredients = data[i]['ingredients']
            for ingredient in ingredients:
		# add new ingredient to list
                if ingredient not in all_ingredients:
                    all_ingredients.append(ingredient)
        
        # create class column vector
        classes = np.array(all_classes)
        classes = np.resize(classes, (classes.size, 1))
        
        
        print 'There are ', classes.shape, ' classes\n' 
        print classes

        ingredients = np.array(all_ingredients)
        ingredients = np.resize(ingredients, (ingredients.size, 1))
        print 'There are ', ingredients.size, ' different ingredients\n'
        print ingredients


        print '\n Dims of ingredients'
        print ingredients.shape

        
        NUM_ING = ingredients.size
        train_rows = []
        y_rows = []
        
        for training_example in data:
            ing = training_example['ingredients']
            f_vec = np.zeros((1, NUM_ING))
            y_vec = np.zeros((1, classes.size))

            for ingredient in ing:
                np.put(f_vec, all_ingredients.index(ingredient), 1)
            train_rows.append(f_vec)
            np.put(y_vec, all_classes.index(training_example['cuisine']), 1)
            y_rows.append(y_vec)

        X = np.array(train_rows)
        y = np.array(y_rows)

        print 'X: all training examples'
        print X

        print 'y'
        print y


        return (all_classes, ingredients, X, y)
             

