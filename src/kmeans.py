from graph_tool import *
import graph_tool.all as gt
from sknetwork.clustering import KMeans
from sklearn.metrics import confusion_matrix, classification_report, rand_score
import time

start_time = time.time()

g = load_graph("../models/threeSpecies.gt")
A = gt.adjacency(g)
g_groundtruth = load_graph("../models/threeSpecies_groundtruth.gt")
A_groundtruth = gt.adjacency(g_groundtruth)

kmeans = KMeans(n_clusters=3)
labels = kmeans.fit_transform(A)
end_time = time.time()
labels_groundtruth = kmeans.fit_transform(A_groundtruth)

print(len(labels))
print(len(labels_groundtruth))

# evaluate
print('Confusion Matrix')
print(confusion_matrix(labels_groundtruth, labels))
print('Classification Report')
print(classification_report(labels_groundtruth, labels))
print('Rand index')
print(rand_score(labels_groundtruth, labels))

print('----------------------------------------------')
print('Running time: ' + str(end_time - start_time) + ' second')