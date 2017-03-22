import json
from pprint import pprint
from sets import Set
import numpy as np

# method to parse data from json file
def parse_input(filename):
    with open(filename) as data_file:  

        data = json.load(data_file)

	# data structures
        all_classes = [] # target values
        all_ingredients = [] # all features
        ing_indexes = {} # maintain column index of features
        curr_index = 1 # counter for feature columne indices
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
                    ing_indexes[ingredient] = curr_index
                    curr_index += 1
        
        # create class column vector
        all_classes = np.array(all_classes)
        all_classes = np.resize(all_classes, (all_classes.size, 1))
        print 'There are ', all_classes.shape, ' classes\n' 
        print all_classes

        all_ingredients = np.array(all_ingredients)
        all_ingredients = np.resize(all_ingredients, (all_ingredients.size, 1))
        print 'There are ', all_ingredients.size, ' different ingredients\n'
        print all_ingredients


        print '\n Dims of ingredients'
        print all_ingredients.shape

        NUM_ING = all_ingredients.size
        train_rows = []
        y_rows = []
        
        for training_example in data:
            ing = training_example['cuisine']
            row = np.zeros((1, NUM_ING))
            for ingredient in ing:
                

             

(classes, X_train) = parse_input('train.json')

