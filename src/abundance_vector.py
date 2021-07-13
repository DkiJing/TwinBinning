import argparse

import numpy as np
import re
import csv

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('-l', '--length', type=int, default=4000)
parser.add_argument('-i', '--input', type=str, default=None)
parser.add_argument('-o', '--output', type=str, default=None)
args = parser.parse_args()

contig_len = args.length
abundance_file = args.input
output_file = args.output
flag = 0
prev_idx = 0
coverage_vector = [0 for i in range(contig_len)]
coverage_list = []

for n, line in enumerate(open(abundance_file)):
    name = line.strip().split()[0]
    idx = int(line.strip().split()[0].split('|')[0])
    pos = int(line.strip().split()[1])
    coverage = int(line.strip().split()[2])
    if n == 0:
        prev_idx = idx
    if prev_idx != idx:
        flag = 1
    prev_idx = idx
    # new contig comes
    if flag == 1:
        coverage_list.append(coverage_vector.copy())
        coverage_vector = [0 for i in coverage_vector]
        flag = 0
    # add converage to the coverage vector
    coverage_vector[pos - 1] = coverage
coverage_list.append(coverage_vector.copy())

vector = np.array(coverage_list)
print(vector.shape)

# write feature vector to csv file
with open(output_file, 'a', newline='') as f:
    f_csv = csv.writer(f)
    f_csv.writerows(vector)