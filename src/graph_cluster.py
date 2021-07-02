from graph_tool import *
import graph_tool.all as gt
from graph_tool.spectral import adjacency
from scipy.sparse.construct import rand
from sknetwork.clustering import KMeans
from sklearn.metrics import adjusted_rand_score, rand_score
from sklearn.metrics.cluster import contingency_matrix, pair_confusion_matrix, normalized_mutual_info_score
from sklearn.metrics.cluster import homogeneity_score, completeness_score, v_measure_score
import time
import numpy as np
import argparse
from sknetwork.clustering import Louvain
import markov_clustering as mc

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('-m', '--mode', type=str, default='louvain')
parser.add_argument('-g', '--graph', type=str, default='../models/threeSpecies.gt')
parser.add_argument('-gt', '--groundtruth', type=str, default='../models/threeSpecies_groundtruth.gt')
parser.add_argument('-r', '--resolution', type=float, default=1.0)
parser.add_argument('-k', '--numclusters', type=int, default=None)
parser.add_argument('-min', '--minBlock', type=int, default=None)
parser.add_argument('-max', '--maxBlock', type=int, default=None)
parser.add_argument('-inf', '--inflation', type=float, default=None)
args = parser.parse_args()

MODE = args.mode
GRAPH_PATH = args.graph
GROUND_TRUTH_PATH = args.groundtruth

def cluster(mode, adjacency_matrix, adjacency_groundtruth):
    if mode == 'kmeans':
        num_clusters = args.numclusters
        if num_clusters == None:
            print('Please input the number of clusters by indicating -k')
            return
        kmeans = KMeans(n_clusters=num_clusters)
        labels = kmeans.fit_transform(adjacency_matrix)
        labels_groundtruth = kmeans.fit_transform(adjacency_groundtruth)
    elif mode == 'louvain':
        resolution = args.resolution
        louvain = Louvain(resolution=resolution)
        labels = louvain.fit_transform(adjacency_matrix)
        labels_groundtruth = louvain.fit_transform(adjacency_groundtruth)
    elif mode == 'mcmc':
        min_Blocks = args.minBlock
        max_Blocks = args.maxBlock
        if min_Blocks == None or max_Blocks == None:
            print('Please input the number of min clusters and max clusters by indicating -min and -max')
            return
        state = gt.minimize_blockmodel_dl(g, B_min=min_Blocks, B_max=max_Blocks)
        labels = state.get_blocks().get_array()
        state_groundtruth = gt.minimize_blockmodel_dl(g_groundtruth)
        labels_groundtruth = state_groundtruth.get_blocks().get_array()
    elif mode == 'mcl':
        inflation = args.inflation
        if inflation == None:
            print('Please input the value of inflation by indicating -inf')
        result = mc.run_mcl(A, inflation=inflation)          
        clusters = mc.get_clusters(result) 
        result_groundtruth = mc.run_mcl(A_groundtruth)
        clusters_groundtruth = mc.get_clusters(result_groundtruth)
        # get the number of nodes
        nodes = A.shape[0]
        # map node to cluster id for labels
        cluster_map = {node: i for i, cluster in enumerate(clusters) for node in cluster}
        labels = [cluster_map[i] for i in range(nodes)]
        # map nodes to ground truth cluster id for labels 
        cluster_map_groundtruth = {node: i for i, cluster in enumerate(clusters_groundtruth) for node in cluster}
        labels_groundtruth = [cluster_map_groundtruth[i] for i in range(nodes)]
    else:
        print('Please specify one of the clustering algorithms: kmeans, louvain, mcmc mcl')
        return
    return labels, labels_groundtruth

if __name__ == '__main__':
    # Load graph
    g = load_graph(GRAPH_PATH)
    A = gt.adjacency(g)
    g_groundtruth = load_graph(GROUND_TRUTH_PATH)
    A_groundtruth = gt.adjacency(g_groundtruth)

    start_time = time.time()
    labels, labels_groundtruth = cluster(MODE, A, A_groundtruth)
    end_time = time.time()

    # evaluate
    print('Contingency Matrix')
    print(contingency_matrix(labels_groundtruth, labels))
    print('Pair Confusion Matrix')
    pm = pair_confusion_matrix(labels_groundtruth, labels)
    print(pm)
    precision = pm[1][1] / (pm[1][1] + pm[0][1])
    recall = pm[1][1] / (pm[1][1] + pm[1][0])
    print('Pricision: ' + str(precision))
    print('Recall: ' + str(recall))
    print('Adjusted Rand Index: ' + str(adjusted_rand_score(labels_groundtruth, labels)))
    print('Rand Index: ' + str(rand_score(labels_groundtruth, labels)))
    print('Normalized Mutual Information: ' + str(normalized_mutual_info_score(labels_groundtruth, labels)))
    print('Homogeneity: ' + str(homogeneity_score(labels_groundtruth, labels)))
    print('Completeness ' + str(completeness_score(labels_groundtruth, labels)))
    print('V measure ' + str(v_measure_score(labels_groundtruth, labels)))
    print('----------------------------------------------')
    print('Running time for graph clustering: ' + str(end_time - start_time) + ' second')