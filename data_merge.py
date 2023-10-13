import json
import os

example = []

data_path = 'ise-dsc01/cache'

list_items = os.listdir(data_path)
for i in range(len(list_items)):
    example.append(json.load(open(os.path.join(data_path, list_items[i]))))

with open(os.path.join(data_path, 'train.json'), 'w', encoding ='utf8') as json_file: 
    json.dump(example, json_file)
