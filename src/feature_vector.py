import argparse

from Bio import SeqIO
from sklearn.feature_extraction.text import CountVectorizer
import csv
import numpy as np
from random import randrange
import sys

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

if kmer_size == 3:
    words = ['aaa', 'aac', 'aag', 'aat', 'aca', 'acc', 'acg', 'act', 'aga', 'agc', 'agg', 'agt', 'ata', 'atc', 'atg', 'att', 'caa', 'cac', 'cag', 'cat', 'cca', 'ccc', 'ccg', 'cct', 'cga', 'cgc', 'cgg', 'cgt', 'cta', 'ctc', 'ctg', 'ctt', 'gaa', 'gac', 'gag', 'gat', 'gca', 'gcc', 'gcg', 'gct', 'gga', 'ggc', 'ggg', 'ggt', 'gta', 'gtc', 'gtg', 'gtt', 'taa', 'tac', 'tag', 'tat', 'tca', 'tcc', 'tcg', 'tct', 'tga', 'tgc', 'tgg', 'tgt', 'tta', 'ttc', 'ttg', 'ttt']
elif kmer_size == 4:
    words = ['aaaa', 'aaac', 'aaag', 'aaat', 'aaca', 'aacc', 'aacg', 'aact', 'aaga', 'aagc', 'aagg', 'aagt', 'aata', 'aatc', 'aatg', 'aatt', 'acaa', 'acac', 'acag', 'acat', 'acca', 'accc', 'accg', 'acct', 'acga', 'acgc', 'acgg', 'acgt', 'acta', 'actc', 'actg', 'actt', 'agaa', 'agac', 'agag', 'agat', 'agca', 'agcc', 'agcg', 'agct', 'agga', 'aggc', 'aggg', 'aggt', 'agta', 'agtc', 'agtg', 'agtt', 'ataa', 'atac', 'atag', 'atat', 'atca', 'atcc', 'atcg', 'atct', 'atga', 'atgc', 'atgg', 'atgt', 'atta', 'attc', 'attg', 'attt', 'caaa', 'caac', 'caag', 'caat', 'caca', 'cacc', 'cacg', 'cact', 'caga', 'cagc', 'cagg', 'cagt', 'cata', 'catc', 'catg', 'catt', 'ccaa', 'ccac', 'ccag', 'ccat', 'ccca', 'cccc', 'cccg', 'ccct', 'ccga', 'ccgc', 'ccgg', 'ccgt', 'ccta', 'cctc', 'cctg', 'cctt', 'cgaa', 'cgac', 'cgag', 'cgat', 'cgca', 'cgcc', 'cgcg', 'cgct', 'cgga', 'cggc', 'cggg', 'cggt', 'cgta', 'cgtc', 'cgtg', 'cgtt', 'ctaa', 'ctac', 'ctag', 'ctat', 'ctca', 'ctcc', 'ctcg', 'ctct', 'ctga', 'ctgc', 'ctgg', 'ctgt', 'ctta', 'cttc', 'cttg', 'cttt', 'gaaa', 'gaac', 'gaag', 'gaat', 'gaca', 'gacc', 'gacg', 'gact', 'gaga', 'gagc', 'gagg', 'gagt', 'gata', 'gatc', 'gatg', 'gatt', 'gcaa', 'gcac', 'gcag', 'gcat', 'gcca', 'gccc', 'gccg', 'gcct', 'gcga', 'gcgc', 'gcgg', 'gcgt', 'gcta', 'gctc', 'gctg', 'gctt', 'ggaa', 'ggac', 'ggag', 'ggat', 'ggca', 'ggcc', 'ggcg', 'ggct', 'ggga', 'gggc', 'gggg', 'gggt', 'ggta', 'ggtc', 'ggtg', 'ggtt', 'gtaa', 'gtac', 'gtag', 'gtat', 'gtca', 'gtcc', 'gtcg', 'gtct', 'gtga', 'gtgc', 'gtgg', 'gtgt', 'gtta', 'gttc', 'gttg', 'gttt', 'taaa', 'taac', 'taag', 'taat', 'taca', 'tacc', 'tacg', 'tact', 'taga', 'tagc', 'tagg', 'tagt', 'tata', 'tatc', 'tatg', 'tatt', 'tcaa', 'tcac', 'tcag', 'tcat', 'tcca', 'tccc', 'tccg', 'tcct', 'tcga', 'tcgc', 'tcgg', 'tcgt', 'tcta', 'tctc', 'tctg', 'tctt', 'tgaa', 'tgac', 'tgag', 'tgat', 'tgca', 'tgcc', 'tgcg', 'tgct', 'tgga', 'tggc', 'tggg', 'tggt', 'tgta', 'tgtc', 'tgtg', 'tgtt', 'ttaa', 'ttac', 'ttag', 'ttat', 'ttca', 'ttcc', 'ttcg', 'ttct', 'ttga', 'ttgc', 'ttgg', 'ttgt', 'ttta', 'tttc', 'tttg', 'tttt']
else:
    sys.exit("This script only supports 3-mer and 4-mer. If you want to customize more kmers, please update words list.")

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
cv = CountVectorizer(vocabulary=words)
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

