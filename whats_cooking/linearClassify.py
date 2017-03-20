import json
from pprint import pprint
from sets import Set
import numpy as np

with open('train.json') as data_file:  

        data = json.load(data_file)
        #print('First data point')
        #pprint(data[0])

        #print('\nSecond point')
        #pprint(data[1])

        #print('\nsecond cuisine')
        
        #print(data[1]['cuisine'])

        all_classes = Set()
        all_ingredients = []

        for i in range(0, len(data)):
            cuisine = data[i]['cuisine']
            if cuisine not in all_classes:
                all_classes.add(cuisine)

            ingredients = data[i]['ingredients']
            for ingredient in ingredients:
                if ingredient not in all_ingredients:
                    all_ingredients.append(ingredient)
        
        
        print 'There are ', len(all_classes), ' classes\n' 
        print all_classes

        print 'There are ', len(all_ingredients), ' different ingredients\n'
        print all_ingredients



