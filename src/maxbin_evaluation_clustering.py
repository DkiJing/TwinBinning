from Bio import SeqIO
import os
import re
from sklearn.metrics import rand_score, adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score, pair_confusion_matrix, contingency_matrix
from sklearn.metrics.cluster import homogeneity_score, completeness_score, v_measure_score
import numpy as np

num_labels = 10
classes_dict = {'Bacteroides_uniformis': 0, 'Bacteroides_xylanisolvens': 1,
'Cohaesibacter_sp': 2, 'Halomonas_sp_HL_four': 3,
'Marinobacter_spone': 4, 'M_echinaurantiaca': 5, 
'Muricauda_sp': 6, 'Propionibact_b': 7, 'Psychrobacter_sp': 8, 'Thioclava_sp': 9}

def get_truth(seqName, binNumber):
    # get main sequence
    seq_list = [s.seq for s in SeqIO.parse(seqName, "fasta")]
    contigs = [s.id.split('|')[1] for s in SeqIO.parse(seqName, "fasta")]

    labels_truth = []
    labels = []
    for contig in contigs:
        labels_truth.append(classes_dict[contig])
        labels.append(binNumber)
    return labels_truth, labels
        
if __name__ == '__main__':
    directory = "../benchmarks/metabat2/myout"
    rand_index = []
    truth_list = []
    pred_list = []
    binNumber = 0
    for filename in os.listdir(directory):
        if filename.endswith(".fasta") or filename.endswith(".fa"):
            seqName = os.path.join(directory, filename)
            truth, pred = get_truth(seqName, binNumber)
            truth_list.extend(truth)
            pred_list.extend(pred)
            rand_index.append(rand_score(truth, pred))
            binNumber += 1
    print("Rand index for each bin:")
    print(rand_index)
    print('Contingency Matrix')
    classes, class_idx = np.unique(truth_list, return_inverse=True)
    print(classes)
    print(contingency_matrix(truth_list, pred_list))
    print('Pair Confusion Matrix')
    pm = pair_confusion_matrix(truth_list, pred_list)
    print(pm)
    precision = pm[1][1] / (pm[1][1] + pm[0][1])
    recall = pm[1][1] / (pm[1][1] + pm[1][0])
    print("Pricision: " + str(precision))
    print("Recall: " + str(recall))
    print("Rand index: " + str(rand_score(truth_list, pred_list)))
    print("Adjusted rand index: " + str(adjusted_rand_score(truth_list, pred_list)))
    print("Normalized mutual information: " + str(normalized_mutual_info_score(truth_list, pred_list)))
    print("Homogeneity: " + str(homogeneity_score(truth_list, pred_list)))
    print("Completeness: " + str(completeness_score(truth_list, pred_list)))
    print("V measure: " + str(v_measure_score(truth_list, pred_list)))