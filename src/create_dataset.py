import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import csv
import numpy as np

seed = 7
# output file
pairs_file = '../data/test_dataset/3-mer/pairs.csv'
test_file = '../data/test_dataset/3-mer/test.csv'

feature_vectors = pd.read_csv('../data/test_dataset/3-mer/kmer.csv', header=None)
feature_vectors_0 = feature_vectors.loc[((feature_vectors.iloc[:, -1] < 1) | (feature_vectors.iloc[:, -1] >= 5)) & (feature_vectors.iloc[:, -1] != 10)]
# dataset = feature_vectors.values
dataset = feature_vectors_0.values
X = dataset[:, :-1]
Y = dataset[:, -1]
print(X.shape)
print(Y.shape)
print(set(Y))

# split dataset to train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=seed)

X_test = preprocessing.MinMaxScaler().fit_transform(X_test)
    
pairs = []
for i in range(len(X_test)):
    for j in range(i + 1, len(X_test)):
        if Y_test[i] == Y_test[j]:
            label = 1
            pairs.append([i, j, label])
        else:
            label = 0
            pairs.append([i, j, label])

pairs = np.array(pairs)
print(pairs.shape)

print("Start writing pairs to csv")
# write pairs to csv file
with open(pairs_file, 'a', newline='') as f:
    f_csv = csv.writer(f)
    f_csv.writerows(pairs)

f.close()
print("Writing pairs finish!")
print("---------------------")
# prepare test set
Y_test = Y_test.reshape((-1, 1))
test = np.hstack((X_test, Y_test))
print(test.shape)
print("Start writing test data to csv")
# write test feature vector to csv file
with open(test_file, 'a', newline='') as f:
    f_csv = csv.writer(f)
    f_csv.writerows(test)

f.close()
print("Writing test data finish!")
