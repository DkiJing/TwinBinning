import pandas as pd
from scipy.sparse import data
import torch
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import torch.utils.data
from dataset import SiameseData
from network import EmbeddingNet, SiameseNet
from losses import ContrastiveLoss
import torch.optim as optim
from torch.optim import lr_scheduler
from trainer import fit, test_epoch
import matplotlib.pyplot as plt
import numpy as np
from metrics import AccumulatedAccuracyMetric, SiameseAccuracyMetric
import time

# parameters
seed = 7
batch_size = 128
cuda = torch.cuda.is_available()
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
SIAMESE_PATH = '../models/3-mer/siamese_net.pth'

# classes = ['Bacteroides caecimuris', 'Bacteroides coprosuis', 'Bacteroides fragilis',
#            'Bacteroides helcogenes', 'Bacteroides heparinolyticus', 'Bacteroides ovatus',
#            'Bacteroides thetaiotaomicron', 'Bacteroides vulgatus', 'Bacteroides zoogleoformans', 
#            'Phocaeicola dorei', 'Phocaeicola salanitronis']

# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
#           '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
#           '#bcbd22', '#17becf', '#99ff99']

# classes = ['Acidaminococcus fermentans', 'Bacteroides uniformis', 'Bacteroides xylanisolvens',
#            'Eggerthella lenta', 'Enterobacter cancerogenus', 'Enterocloster bolteae',
#            'Proteus mirabilis', 'Providencia alcalifaciens', 'Providencia_rettgeri', 'Rickettsia typhi']

# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
#           '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
#           '#bcbd22', '#17becf']

# classes = ['Cohaesibacter sp', 'Halomonas sp HL-4', 'Halomonas sp HL-93',
#            'Marinobacter sp1', ' Marinobacter sp8', 'M coxensis',
#            'M echinaurantiaca', 'M echinofusca ', 'Muricauda sp',
#            'Propionibact b', 'Psychrobacter sp', 'Thioclava sp']

# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
#           '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
#           '#bcbd22', '#17becf', '#99ff99', '#000099']

classes = ['Cohaesibacter sp', 'Halomonas sp HL-93', 'M coxensis',
           'M echinaurantiaca', 'M echinofusca ', 'Muricauda sp',
           'Propionibact b', 'Thioclava sp']

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
          '#bcbd22', '#17becf', '#99ff99', '#000099']

n_classes = 12

mode = "train"

# visualization
def plot_embeddings(embeddings, targets, xlim=None, ylim=None):
    plt.figure(figsize=(10, 10))
    for i in range(n_classes):
        inds = np.where(targets == i)[0]
        plt.scatter(embeddings[inds, 0], embeddings[inds, 1], alpha=0.5, color=colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(classes)


def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 2))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for seq, target in dataloader:
            if cuda:
                seq = seq.cuda()
            embeddings[k:k + len(seq)] = model.get_embedding(seq).data.cpu().numpy()
            labels[k:k + len(seq)] = target.numpy()
            k += len(seq)
    return embeddings, labels

def plot_model_history(loss_train, loss_test, accuracy_train, accuracy_test):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # loss of model
    axs[0].plot(range(1,len(loss_train)+1),loss_train)
    axs[0].plot(range(1,len(loss_test)+1),loss_test)
    axs[0].set_title('Model Loss')
    axs[0].set_ylabel('Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(loss_train)+1), len(loss_train)/10)
    axs[0].legend(['train', 'val'], loc='best')
    # accuracy of model
    axs[1].plot(range(1,len(accuracy_train)+1),accuracy_train)
    axs[1].plot(range(1,len(accuracy_test)+1),accuracy_test)
    axs[1].set_title('Model Accuracy')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(accuracy_train)+1),len(accuracy_train)/10)
    axs[1].legend(['train', 'val'], loc='best')
    fig.savefig('../images/' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '.png')

if __name__ == '__main__':
    # load data from csv file
    feature_vectors = pd.read_csv('../data/3-mer/train.csv', header=None)
    # feature_vectors = pd.read_csv('../data/test_dataset/3-mer/kmer.csv', header=None)
    dataset = feature_vectors.values
    X = dataset[:, :-1]
    Y = dataset[:, -1]

    # split dataset to train and test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)
    X_train, X_test = preprocessing.MinMaxScaler().fit_transform(X_train), preprocessing.MinMaxScaler().fit_transform(
        X_test)

    if cuda:
        X_train, Y_train = torch.cuda.FloatTensor(X_train), torch.cuda.LongTensor(Y_train)
        X_test, Y_test = torch.cuda.FloatTensor(X_test), torch.cuda.LongTensor(Y_test)
    else:
        X_train, Y_train = torch.FloatTensor(X_train), torch.LongTensor(Y_train)
        X_test, Y_test = torch.FloatTensor(X_test), torch.LongTensor(Y_test)

    X_train = X_train.unsqueeze(1)
    X_test = X_test.unsqueeze(1)

    # construct dataset
    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, Y_test)

    # load data set
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    # construct dataset in pairs
    siamese_train_dataset = SiameseData(dataset=train_dataset, train=True)
    siamese_test_dataset = SiameseData(dataset=test_dataset, train=False)

    # load data set in pairs
    siamese_train_loader = torch.utils.data.DataLoader(siamese_train_dataset, batch_size=batch_size, shuffle=True,
                                                       **kwargs)
    siamese_test_loader = torch.utils.data.DataLoader(siamese_test_dataset, batch_size=batch_size, shuffle=False,
                                                      **kwargs)

    if mode == "train":
        # initilize model
        embedding_net = EmbeddingNet()
        model = SiameseNet(embedding_net)

        # GPU acceleration
        if cuda:
            model.cuda()

        # define loss function
        margin = 5
        loss_fn = ContrastiveLoss(margin)

        # initilize optimizer
        lr = 1e-3
        optimizer = optim.Adam(model.parameters(), lr=lr)

        scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
        n_epochs = 20
        log_interval = 100

        # train the model
        loss_values_train, loss_values_test, accuracy_values_train, accuracy_values_test = fit(siamese_train_loader, siamese_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda,
            log_interval, metrics=[SiameseAccuracyMetric()])

        # save the model parameters
        torch.save(model.state_dict(), SIAMESE_PATH)

        # save model losses result
        plot_model_history(loss_values_train, loss_values_test, accuracy_values_train, accuracy_values_test)

    elif mode == "test":
        # test siamese net
        # initilize model
        embedding_net = EmbeddingNet()
        model = SiameseNet(embedding_net)
        
        # load the model parameters
        model.load_state_dict(torch.load(SIAMESE_PATH))

        # define loss function
        margin = 5
        loss_fn = ContrastiveLoss(margin)

        val_loss, metrics = test_epoch(siamese_test_loader, model, loss_fn, cuda, metrics=[SiameseAccuracyMetric()])
        print(len(siamese_train_dataset))
        print('length of test dataset: ' + str(len(siamese_test_dataset)))
        for metric in metrics:
            print(str(metric.name()) + "  " + str(metric.value()))
            print('confusion matrix:')
            print(metric.confusion_matrix())

        # plot embedding vectors
        train_embeddings_cl, train_labels_cl = extract_embeddings(train_loader, model)
        plot_embeddings(train_embeddings_cl, train_labels_cl)
        val_embeddings_cl, val_labels_cl = extract_embeddings(test_loader, model)
        plot_embeddings(val_embeddings_cl, val_labels_cl)

        plt.show()
