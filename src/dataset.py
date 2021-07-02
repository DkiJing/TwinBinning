import numpy as np
from torch.utils.data import Dataset

class SiameseData(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, dataset, train):
        self.dataset = dataset
        self.train = train

        if self.train:
            self.train_labels = self.dataset.tensors[1]
            self.train_data = self.dataset.tensors[0]
            self.labels_set = set(self.train_labels.cpu().numpy())
            self.label_to_indices = {label: np.where(self.train_labels.cpu().numpy() == label)[0]
                                      for label in self.labels_set}

        else:
            # generate fixed pairs for testing
            self.test_labels = self.dataset.tensors[1]
            self.test_data = self.dataset.tensors[0]
            self.labels_set = set(self.test_labels.cpu().numpy())
            self.label_to_indices = {label: np.where(self.test_labels.cpu().numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            # sequence pair from same class, output 1
            # [idx1, idx2, 1], where idx1 is even number
            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                               1]
                              for i in range(0, len(self.test_data), 2)]

            # sequence pair from different classes, output 0
            # [idx1, idx2, 0], where idx1 is odd number
            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.test_labels[i].item()]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.test_data), 2)]
            self.test_pairs = positive_pairs + negative_pairs

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            seq1, label1 = self.train_data[index], self.train_labels[index].item()
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            seq2 = self.train_data[siamese_index]
        else:
            seq1 = self.test_data[self.test_pairs[index][0]]
            seq2 = self.test_data[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]
        return (seq1, seq2), target

    def get_seq_pair(self, index):
        return self.test_pairs[index]

class AllPairsData(Dataset):
    def __init__(self, dataset, pairs):
        self.dataset = dataset
        self.pairs = pairs
        self.data = self.dataset.tensors[0]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        target = self.pairs[index][2]
        seq1 = self.data[self.pairs[index][0]]
        seq2 = self.data[self.pairs[index][1]]
        return (seq1, seq2), target

    def get_seq_pair(self, index):
        return self.pairs[index]

    def feature_length(self):
        return len(self.data)