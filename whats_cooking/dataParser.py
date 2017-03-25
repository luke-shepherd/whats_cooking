import json
from pprint import pprint
from sets import Set
import numpy as np
import random
import pdb


# split data into training and testing data
def split_data(X,y, ratio = .75):
   # create data matrix
   # add classes to first column of matrix
   # add examples to rest of columns
   size = X.shape[0]
   index_to_split = int(size * ratio)
   print "Fetch first %d examples in data" % index_to_split   
   X_train = X[0:index_to_split,:]
   y_train = y[0:index_to_split]

   return X_train,y_train



def split_training_data(X, y, all_classes, y_cuisine, ratio):
    X_tr = []
    X_te = []
    y_tr = []
    y_te = []
    y_tr_labels = []
    y_te_labels = []
    for i in range(0, X.shape[0]):
        r = random.random()
        if r < ratio:
            X_tr.append(X[i, :])
            y_tr.append(y[i, :])
            y_tr_labels.append(all_classes.index(y_cuisine[i]))
        else:
            X_te.append(X[i, :])
            y_te.append(y[i, :])
            y_te_labels.append(all_classes.index(y_cuisine[i]))

    return (np.array(X_tr), np.array(X_te), np.array(y_tr), np.array(y_te), y_tr_labels, y_te_labels)



# method to parse data from json file
def parse_input(filename):
    with open(filename) as data_file:  

        print "Load in the data..."
        data = json.load(data_file)

	# data structures
        all_classes = [] # target values
        all_ingredients = [] # all features
        NUM_EXAMPLES = len(data)

	# iterate over the cuisines
	# add cuisine to list if not seen yet
        print "Discover cuisines and ingredients..."
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

        ingredients = np.array(all_ingredients)
        ingredients = np.resize(ingredients, (ingredients.size, 1))
        
        NUM_ING = ingredients.size
        train_rows = []
        y_rows = []
        y_cuisine = [] # list of cuisine for example
        
        print "Build the example and target matrices..."
        for training_example in data:
            ing = training_example['ingredients']
            f_vec = np.zeros((1, NUM_ING))
            y_vec = np.zeros((1, classes.size))
            y_cuisine.append(training_example['cuisine'])

            for ingredient in ing:
                np.put(f_vec, all_ingredients.index(ingredient), 1)
            train_rows.append(f_vec)
            np.put(y_vec, all_classes.index(training_example['cuisine']), 1)
            y_rows.append(y_vec)

        X = np.array(train_rows).squeeze()
        y = np.array(y_rows).squeeze()

        print "Classes: %d\nFeatures: %d\nExamples: %d"% (len(all_classes), X.shape[1],X.shape[0])

	   # return tuple of: class array, ingredients array, example array
	   # 'hot-vector' array, list of target cuisines, list of all cuisines
        return (classes, ingredients, X, y,y_cuisine,all_classes)
             
def convert_weka(filename):
    with open(filename) as data_file:  

        print "Load in the data..."
        data = json.load(data_file)

        # data structures
        all_classes = [] # target values
        all_ingredients = [] # all features
        NUM_EXAMPLES = len(data)

        print "Discover cuisines and ingredients..."
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

        
        print 'creating json...'
        all_attr = []
        NUM_INGREDIENTS = len(all_ingredients)
        dataList = []
        example_counter = 0
        for example in data:
            if example_counter % 100 == 0: print example_counter
            example_counter += 1
            curr_ingredients = example['ingredients']
            cd = {}
            attr = {}

            cd["weight"] = 1
            cd["sparse"] = False
      
            attr['name'] = 'R' + str(i)
            attr['type'] = 'numeric'
            attr['class'] = False
            attr['weight'] = 1.0

            values = []
            for i in range(0, NUM_INGREDIENTS):
                if all_ingredients[i] in curr_ingredients:
                    values.append('1')
                else: 
                    values.append('0')
            exclass = all_classes.index(example['cuisine'])
            values.append(str(exclass))
            cd['values'] = values
            dataList.append(cd)

        class_attr = {}
        class_attr['name'] = 'Class'
        class_attr['type'] = 'nominal'
        class_attr['class'] = True
        class_attr['weight'] = 1.0
        class_attr['labels'] = ['1', '2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20']
        all_attr.append(class_attr)

        out = {}
        header = {}
        header['relation'] = 'Foods'
        header['attributes'] = all_attr
        header['data'] = dataList
        out['header'] = header

        print 'writing...' 
        with open('trainWeka.json', 'w') as json_weka:
            json_weka.write(json.dumps(out))
