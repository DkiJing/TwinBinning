import argparse

from Bio import SeqIO
from sklearn.feature_extraction.text import CountVectorizer
import csv
import numpy as np
from random import randrange

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('-i', '--input', type=str, default=None)
parser.add_argument('-o', '--output', type=str, default=None)
parser.add_argument('-k', '--kmer', type=int, default=5)
parser.add_argument('-l', '--label', type=int, default=-1)
args = parser.parse_args()

# parameters
input_path = args.input
output_path = args.output
kmer_size = args.kmer
label = args.label

# generate kmers sub-string
def build_kmers(sequence, ksize):
    kmers = []
    n_kmers = len(sequence) - ksize + 1
    for i in range(n_kmers):
        kmer = sequence[i:i + ksize]
        kmers.append(kmer)
    return kmers

# get many sequences from fasta file
sequences = []
# labels assignment
labels = []
if label == -1:
    print('randomly generate label')
    for seq_record in SeqIO.parse(input_path, "fasta"):
        if len(seq_record.seq) >= 2000:
            sequences.append(str(seq_record.seq))
            labels.append(randrange(10))
else:
    for seq_record in SeqIO.parse(input_path, "fasta"):
        if len(seq_record.seq) >= 2000:
            sequences.append(str(seq_record.seq))
            labels.append(label)

# kmers of size
for i in range(len(sequences)):
    sequences[i] = ' '.join(build_kmers(sequences[i], kmer_size))

# creating the Bag of Words model using CountVectorizer()
cv = CountVectorizer()
X = cv.fit_transform(sequences)
# encode sequences by counts of kmers
vector = cv.transform(sequences).toarray()

# add labels to vector
labels = np.array(labels).reshape((-1, 1))
vector = np.hstack((vector, labels))

print("K-mer feature shape: " + str(vector.shape))

# write feature vector to csv file
with open(output_path, 'a', newline='') as f:
    f_csv = csv.writer(f)
    f_csv.writerows(vector)

f.close()

