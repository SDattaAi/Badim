import pandas as pd
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


with open('no_gradient_models.pickle', 'rb') as f:
    best_no_gradient_models = pickle.load(f)
# replace -1 with nan
df = pd.DataFrame(best_no_gradient_models['20_category']['MAE']).replace(-1, np.nan)
# plot heat map
best_models = df.mean(axis=1).sort_values(ascending=True)
# sort the df
df = df.loc[best_models.index]
print(df)

sns.heatmap(df, cmap='viridis')
plt.show()
