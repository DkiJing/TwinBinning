import pandas as pd
from pandas.core import indexing

feature_vectors_all = pd.read_csv('../data/test_dataset/3-mer/kmer.csv', header=None)
feature_vectors = feature_vectors_all.loc[((feature_vectors_all.iloc[:, -1] < 1) | (feature_vectors_all.iloc[:, -1] >= 5)) & (feature_vectors_all.iloc[:, -1] != 10)]

feature_vectors.to_csv('../data/test_dataset/3-mer/train.csv', header=None, index=None)