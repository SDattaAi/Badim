import pandas as pd
import json


with open('best_no_gradient_models.json', 'r') as f:
    best_no_gradient_models = json.load(f)

print(best_no_gradient_models)
for key, value in best_no_gradient_models.items():
    print(key)
    print(pd.DataFrame(value))