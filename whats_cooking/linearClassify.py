import json
from pprint import pprint
from sets import Set
with open('train.json') as data_file:  

        data = json.load(data_file)
        print('First data point')
        pprint(data[0])

        print('\nSecond point')
        pprint(data[1])

        print('\nsecond cuisine')
        
        print(data[1]['cuisine'])

        all_classes = Set()

        for i in range(0, len(data)):
            cuisine = data[i]['cuisine']
            if cuisine not in all_classes:
                all_classes.add(cuisine)

        print(all_classes)
        print 'There are ', len(all_classes), ' classes' 

