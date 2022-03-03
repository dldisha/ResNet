import json
import os
import pandas as pd

results = dict()
for files in os.listdir('outputs'):
    if files.endswith('.json'):
        with open('outputs/' + files, 'r') as f:
            metric = json.load(f)
        results[len(results)] = metric

results = pd.DataFrame(results).T.to_csv('results.csv')
