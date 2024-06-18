import pandas as pd
import json


with open('best_no_gradient_models.json', 'r') as f:
    best_no_gradient_models = json.load(f)

print(len(best_no_gradient_models))