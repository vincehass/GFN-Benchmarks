"""
  Gradient boosted regressor sEH proxy model.
  Trained on neural net proxy's predictions on
  34M molecules from block18, stop6.
  Attains pearsonr=0.90 on data set.
"""
import pandas as pd, numpy as np
import sklearn

import pickle

data_fn = 'datasets/sehstr/block_18_stop6.pkl'
with open(data_fn, 'rb') as f:
  x_to_r = pickle.load(f)



json_fn = 'datasets/sehstr/block_18.json'
blocks_df = pd.read_json(json_fn)

xs = list(x_to_r.keys())


rs = list(x_to_r.values())
print("keys", xs[:10])
print("rewards", rs[:10])
print("number of molecules (34M)", len(rs))

# Featurize

# symbols = '0123456789abcdefghijklmnopqrstuvwxyz' + \
#               'ABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\()*+,-./:;<=>?@[\]^_`{|}~'
# num_blocks = len(blocks_df)

# import functools

# @functools.cache
# def symbol_ohe(symbol):
#   zs = np.zeros(num_blocks)
#   zs[symbols.index(symbol)] = 1.0
#   return zs

# print(symbol_ohe('1'))

# # Featurization

# def featurize(x):
#   return np.concatenate([symbol_ohe(c) for c in x])

# print(featurize(xs[0]))

# Y = np.array(list(x_to_r.values()))

# from tqdm import tqdm

# X = []
# for x in tqdm(xs):
#   X.append(featurize(x))
# X = np.array(X)

# from sklearn.ensemble import HistGradientBoostingRegressor

# # N_SUBSET = 1000000
# N_SUBSET = len(X)

# model = HistGradientBoostingRegressor()
# model.fit(X[:N_SUBSET], Y[:N_SUBSET])
# print(model.score(X[:N_SUBSET], Y[:N_SUBSET]))


# from scipy.stats import pearsonr

# PEARSONR_SUBSET = 1000000

# pearsonr(model.predict(X[:PEARSONR_SUBSET]), Y[:PEARSONR_SUBSET])

# with open('sehstr_gbtr.pkl', 'wb') as f:
#   pickle.dump(model, f)
# print('Saved to file.')

#pearson_score = 0.8065185482836459










