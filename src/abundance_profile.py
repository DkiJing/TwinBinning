import pandas as pd
import numpy as np
import csv
import heapq
import argparse

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('-i', '--input', type=str, default=None)
parser.add_argument('-o', '--output', type=str, default=None)
# parser.add_argument('-m', '--meandepth', type=int, default=None)
args = parser.parse_args()

abundance_vectors = pd.read_csv(args.input, header=None)
abundance = abundance_vectors.values
output_file = args.output

maxDepth = abundance.max()
minDepth = abundance.min()
# meanDepth = args.meandepth
maxCandidate = 100

print('max depth: ' + str(maxDepth))
print('min depth: ' + str(minDepth))

if maxCandidate >= maxDepth - minDepth:
    freqProfile = np.zeros((abundance.shape[0], maxCandidate))
    for i in range(abundance.shape[0]):
        for j in range(abundance.shape[1]):
            if (abundance[i][j] != 0 and abundance[i][j] < maxDepth):
                freqProfile[i][abundance[i][j] - 1 - minDepth] += 1
# elif maxCandidate >= meanDepth - minDepth:
#     freqProfile = np.zeros((abundance.shape[0], maxCandidate))
#     for i in range(abundance.shape[0]):
#         for j in range(abundance.shape[1]):
#             if (abundance[i][j] != 0 and abundance[i][j] < maxCandidate):
#                 freqProfile[i][abundance[i][j] - 1 - minDepth] += 1
elif maxCandidate < maxDepth - minDepth:
    maxFreqProfile = np.zeros((abundance.shape[0], maxDepth))
    freqProfile = np.zeros((abundance.shape[0], maxCandidate))
    step = int(maxDepth / maxCandidate)
    end = step * maxCandidate
    for i in range(abundance.shape[0]):
        for j in range(abundance.shape[1]):
            if (abundance[i][j] != 0 and abundance[i][j] < maxDepth):
                maxFreqProfile[i][abundance[i][j] - 1] += 1
        freqProfile[i] = maxFreqProfile[i][0:end:int(step)]

print(freqProfile.shape)

# write feature vector to csv file
with open(output_file, 'a', newline='') as f:
    f_csv = csv.writer(f)
    f_csv.writerows(freqProfile)
