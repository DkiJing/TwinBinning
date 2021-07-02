from graph_tool import *
import graph_tool.all as gt
from sknetwork.clustering import Louvain
from sklearn.metrics import confusion_matrix, classification_report, rand_score
from sknetwork.clustering import modularity
import numpy as np
import time

start_time = time.time()

g = load_graph("../models/threeSpecies.gt")
A = gt.adjacency(g)
g_groundtruth = load_graph("../models/threeSpecies_groundtruth.gt")
A_groundtruth = gt.adjacency(g_groundtruth)

louvain = Louvain()
labels = louvain.fit_transform(A)
end_time = time.time()
labels_groundtruth = louvain.fit_transform(A_groundtruth)

modularity = np.round(modularity(A, labels), 2)

# evaluate
print('Confusion Matrix')
print(confusion_matrix(labels_groundtruth, labels))
print('Classification Report')
print(classification_report(labels_groundtruth, labels))
print('Modularity: ' + str(modularity))
print('Rand index')
print(rand_score(labels_groundtruth, labels))

print('----------------------------------------------')
print('Running time: ' + str(end_time - start_time) + ' second')