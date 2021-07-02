from graph_tool import *
import graph_tool.all as gt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, rand_score
import time

PLOT = True

start_time = time.time()

# predicted graph
g = load_graph("../models/threeSpecies.gt")
pos = gt.sfdp_layout(g)
control = g.new_edge_property("vector<double>")
deg = g.degree_property_map("total")
state = gt.minimize_blockmodel_dl(g, B_min=1, B_max=3)

for e in g.edges():
    d = np.sqrt(sum((pos[e.source()].a - pos[e.target()].a) ** 2)) / 5
    control[e] = [0.3, d, 0.7, d]

labels = state.get_blocks().get_array()
end_time = time.time()
# print(labels)

if PLOT == True:
    # with color
    state.draw(pos=pos, 
            output_size=(5000, 5000), 
            edge_control_points=control, # some curvy edges
            output="../images/threeSpecie.pdf")

# groundtruth graph
g_groundtruth = load_graph("../models/threeSpecies_groundtruth.gt")
pos = gt.sfdp_layout(g_groundtruth)
state_groundtruth = gt.minimize_blockmodel_dl(g_groundtruth)

labels_groundtruth = state_groundtruth.get_blocks().get_array()

if PLOT == True:
    state_groundtruth.draw(pos=pos, 
            output_size=(5000, 5000), 
            output="../images/threeSpecie_groundtruth.pdf")
print(labels_groundtruth)

# evaluate
print('Confusion Matrix')
print(confusion_matrix(labels_groundtruth, labels))
print('Classification Report')
print(classification_report(labels_groundtruth, labels))
print('Rand index')
print(rand_score(labels_groundtruth, labels))

print('----------------------------------------------')
print('Running time: ' + str(end_time - start_time) + ' second')