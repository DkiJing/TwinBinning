import argparse
from graph_tool import *
import pandas as pd
import torch.utils.data
from dataset import AllPairsData
from losses import ContrastiveLoss
from network import EmbeddingNet, SiameseNet
from trainer import prediction
from metrics import SiameseAccuracyMetric
import time

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('-m', '--model', type=str, default='../models/3-mer/siamese_net.pth')
parser.add_argument('-i', '--data', type=str, default='../data/test_dataset/3-mer')
parser.add_argument('-o', '--graph', type=str, default='../models/threeSpecies.gt')
parser.add_argument('-gt', '--groundtruth', type=str, default='../models/threeSpecies_groundtruth.gt')
args = parser.parse_args()

# parameters
seed = 7
batch_size = 128
cuda = torch.cuda.is_available()
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
SIAMESE_PATH = args.model
DATA_PATH = args.data
GRAPH_PATH = args.graph
GROUND_TRUTH_PATH = args.groundtruth

if __name__ == '__main__':
    start_time = time.time()
    # load data from csv file
    feature_vectors = pd.read_csv(DATA_PATH + '/test.csv', header=None)
    pairs = pd.read_csv(DATA_PATH + '/pairs.csv', header=None)
    pairs = pairs.values
    dataset = feature_vectors.values
    X = dataset[:, :-1]
    Y = dataset[:, -1]
    
    if cuda:
        X_test, Y_test = torch.cuda.FloatTensor(X), torch.cuda.LongTensor(Y)
    else:
        X_test, Y_test = torch.FloatTensor(X), torch.LongTensor(Y)

    X_test = X_test.unsqueeze(1)

    # construct dataset
    test_dataset = torch.utils.data.TensorDataset(X_test, Y_test)

    # load data set
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=batch_size, **kwargs)

    pairs_test_dataset = AllPairsData(dataset=test_dataset, pairs=pairs)

    pairs_test_loader = torch.utils.data.DataLoader(pairs_test_dataset, shuffle=False, batch_size=batch_size, **kwargs)

    # initilize model
    embedding_net = EmbeddingNet()
    model = SiameseNet(embedding_net)

    # load the model parameters
    model.load_state_dict(torch.load(SIAMESE_PATH))

    # define loss function
    margin = 5
    loss_fn = ContrastiveLoss(margin)

    print("Prediction begins ... It may take some minutes.")
    pred_result, metrics = prediction(pairs_test_loader, model, loss_fn, cuda, metrics=[SiameseAccuracyMetric()])
    for metric in metrics:
        print(str(metric.name()) + "  " + str(metric.value()))
        print('confusion matrix:')
        print(metric.confusion_matrix())
    print("Prediction ends!!!")

    # Construct the graph
    g = Graph(directed=False)
    g.add_vertex(pairs_test_dataset.feature_length())
    print("Start building predicted graph.")
    for i in range(len(pairs_test_dataset)):
        v1_idx = pairs_test_dataset.get_seq_pair(i)[0]
        v2_idx = pairs_test_dataset.get_seq_pair(i)[1]
        if pred_result[i // batch_size][i % batch_size].data.numpy()[0] == 1:
            g.add_edge(g.vertex(v1_idx), g.vertex(v2_idx))
    print("Building predicted graph finish.")
    g.save(GRAPH_PATH)

    # Construct the ground truth graph
    g_true = Graph(directed=False)
    g_true.add_vertex(pairs_test_dataset.feature_length())
    print("Start building ground truth graph.")
    for i in range(len(pairs_test_dataset)):
        v1_idx = pairs_test_dataset.get_seq_pair(i)[0]
        v2_idx = pairs_test_dataset.get_seq_pair(i)[1]
        if pairs_test_dataset.get_seq_pair(i)[2] == 1:
            g_true.add_edge(g_true.vertex(v1_idx), g_true.vertex(v2_idx))
    print("Building ground truth graph finish.")
    g_true.save(GROUND_TRUTH_PATH)

    # Calculate the recall
    TP = 0
    FN = 0
    for i in range(len(pairs_test_dataset)):
        if pairs_test_dataset.get_seq_pair(i)[2] == 1:
            if pred_result[i // batch_size][i % batch_size].data.numpy()[0] == 1:
                TP += 1
            elif pred_result[i // batch_size][i % batch_size].data.numpy()[0] == 0:
                FN += 1
    recall = TP / (TP + FN)
    print("recall: " + str(recall))

    # Calculate the precision
    TP = 0
    FP = 0
    for i in range(len(pairs_test_dataset)):
        if pred_result[i // batch_size][i % batch_size].data.numpy()[0] == 1:
            if pairs_test_dataset.get_seq_pair(i)[2] == 1:
                TP += 1
            elif pairs_test_dataset.get_seq_pair(i)[2] == 0:
                FP += 1
    precision = TP / (TP + FP)
    print("precision: " + str(precision))
    
    end_time = time.time()
    print('----------------------------------------------')
    print('Running time for generating graph: ' + str(end_time - start_time) + ' second')
