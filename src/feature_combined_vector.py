import pandas as pd
import numpy as np
import csv
import argparse
from sklearn import preprocessing

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('-k', '--kmer', type=str, default=None)
parser.add_argument('-a', '--abundance', type=str, default=None)
parser.add_argument('-o', '--output', type=str, default=None)
args = parser.parse_args()

output_file = args.output

kmer_vectors = pd.read_csv(args.kmer, header=None)
abundance_vectors = pd.read_csv(args.abundance, header=None)
kmer = kmer_vectors.values
abundance = abundance_vectors.values

X = np.hstack((kmer[:, :-1], abundance))
Y = kmer[:, -1].reshape((-1, 1))

vector = np.hstack((X, Y))
print(vector.shape)

# write feature vector to csv file
with open(output_file, 'a', newline='') as f:
    f_csv = csv.writer(f)
    f_csv.writerows(vector)
