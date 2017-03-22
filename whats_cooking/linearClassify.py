import json
from pprint import pprint
from sets import Set
import numpy as np


def parse_input(filename):
    with open(filename) as data_file:  

        data = json.load(data_file)
        #print('First data point')
        #pprint(data[0])

        #print('\nSecond point')
        #pprint(data[1])

        #print('\nsecond cuisine')
        
        #print(data[1]['cuisine'])

        all_classes = []
        all_ingredients = []
        ing_indexes = {}
        curr_index = 0
        NUM_EXAMPLES = len(data)
        for i in range(0, NUM_EXAMPLES):
            cuisine = data[i]['cuisine']
            if cuisine not in all_classes:
                all_classes.append(cuisine)

            ingredients = data[i]['ingredients']
            for ingredient in ingredients:
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

