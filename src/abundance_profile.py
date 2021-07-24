import pandas as pd
import numpy as np
import csv
import heapq
import argparse

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('-i', '--input', type=str, default=None)
parser.add_argument('-o', '--output', type=str, default=None)
parser.add_argument('-m', '--meandepth', type=float, default=None)
parser.add_argument('-l', '--length', type=int, default=4000)
args = parser.parse_args()

contig_len = args.length
abundance_file = args.input
output_file = args.output
prev_name = ""
coverage_list = []

def parse_depth_file(abundance_file):
    flag = 0
    coverage_vector = [0 for i in range(contig_len)]
    for n, line in enumerate(open(abundance_file)):
        name = line.strip().split()[0]
        pos = int(line.strip().split()[1])
        coverage = int(line.strip().split()[2])
        if n == 0:
            prev_name = name
        if prev_name != name:
            flag = 1
        prev_name = name
        # new contig comes
        if flag == 1:
            coverage_list.append(coverage_vector.copy())
            coverage_vector = [0 for i in coverage_vector]
            flag = 0
        # add converage to the coverage vector
        coverage_vector[pos - 1] = coverage
    coverage_list.append(coverage_vector.copy())
    abundance = np.array(coverage_list)
    return abundance

def get_abundance_profile(abundance, maxDepth, minDepth, meanDepth):
    if maxCandidate >= maxDepth - minDepth:
        freqProfile = np.zeros((abundance.shape[0], maxCandidate))
        for i in range(abundance.shape[0]):
            for j in range(abundance.shape[1]):
                if (abundance[i][j] != 0 and abundance[i][j] < maxDepth):
                    freqProfile[i][abundance[i][j] - 1 - minDepth] += 1
    elif (meanDepth != None) and (maxCandidate >= meanDepth - minDepth):
        freqProfile = np.zeros((abundance.shape[0], maxCandidate))
        for i in range(abundance.shape[0]):
            for j in range(abundance.shape[1]):
                if (abundance[i][j] != 0 and abundance[i][j] < maxCandidate):
                    freqProfile[i][abundance[i][j] - 1 - minDepth] += 1
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
    return freqProfile

if __name__ == '__main__':
    # Parse depth file produced from Samtools
    abundance = parse_depth_file(abundance_file)
    maxDepth = abundance.max()
    minDepth = abundance.min()
    meanDepth = int(args.meandepth)
    maxCandidate = 192

    print('max depth: ' + str(maxDepth))
    print('min depth: ' + str(minDepth))
    print('mean depth: ' + str(meanDepth))

    # Obatin abundance frequency profile
    freqProfile = get_abundance_profile(abundance, maxDepth, minDepth, meanDepth)
    print("Abundance frequency profile shape: " + str(freqProfile.shape))

    # write feature vector to csv file
    with open(output_file, 'a', newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerows(freqProfile)
