import pandas

df = pandas.read_json('~/train.json')
df.to_csv('~/train.csv')
