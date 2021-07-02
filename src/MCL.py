import markov_clustering as mc
import networkx as nx
from graph_tool import *
import graph_tool.all as gt
from sklearn.metrics import confusion_matrix, classification_report, rand_score
import time

start_time = time.time()

g = load_graph("../models/threeSpecies.gt")
pos = gt.sfdp_layout(g)

A = gt.adjacency(g)
g_groundtruth = load_graph("../models/threeSpecies_groundtruth.gt")
A_groundtruth = gt.adjacency(g_groundtruth)

result = mc.run_mcl(A, inflation=2.5)           # run MCL with default parameters
clusters = mc.get_clusters(result)    # get clusters
end_time = time.time()

result_groundtruth = mc.run_mcl(A_groundtruth)
clusters_groundtruth = mc.get_clusters(result_groundtruth)

print('begin...')
# # perform clustering using different inflation values from 1.3 and 2.5
# # for each clustering run, calculate the modularity
# for inflation in [i / 10 for i in range(13, 26)]:
#     result = mc.run_mcl(A, inflation=inflation)
#     clusters = mc.get_clusters(result)
#     Q = mc.modularity(matrix=result, clusters=clusters)
#     print("inflation:", inflation, "modularity:", Q)

# get the number of nodes
nodes = A.shape[0]
# map node to cluster id for labels
cluster_map = {node: i for i, cluster in enumerate(clusters) for node in cluster}
labels = [cluster_map[i] for i in range(nodes)]

# map nodes to ground truth cluster id for labels

cluster_map_groundtruth = {node: i for i, cluster in enumerate(clusters_groundtruth) for node in cluster}
labels_groundtruth = [cluster_map_groundtruth[i] for i in range(nodes)]

# evaluate
print('Confusion Matrix')
print(confusion_matrix(labels_groundtruth, labels))
print('Classification Report')
print(classification_report(labels_groundtruth, labels))
print('Rand index')
print(rand_score(labels_groundtruth, labels))

print('----------------------------------------------')
print('Running time: ' + str(end_time - start_time) + ' second')

# # draw grpah
# mc.draw_graph(A, clusters, pos=pos, node_size=50, with_labels=False, edge_color="silver")

