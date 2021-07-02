from Bio import SeqIO
import numpy as np
from Bio.SeqRecord import SeqRecord

seqName = "../benchmarks/combined.fa"

# get main sequence
seq_list = [s.seq for s in SeqIO.parse(seqName, "fasta")]
seq_label = [str(s.id.split('|')[1]) for s in SeqIO.parse(seqName, "fasta")]
seq_idx = [str(s.id.split('|')[0]) for s in SeqIO.parse(seqName, "fasta")]
labels_set = set(np.array(seq_label))

random_state = np.random.RandomState(29)
label_to_indices = {label: np.where(np.array(seq_label) == label)[0]
                                      for label in labels_set}

# sequence pair from same class
# [idx1, idx2, 1], where idx1 is even number
positive_pairs = [[i,
                   random_state.choice(label_to_indices[seq_label[i]]),
                   1]
                  for i in range(0, len(seq_list), 2)]

# sequence pair from different classes
# [idx1, idx2, 0], where idx1 is odd number
negative_pairs = [[i,
                   random_state.choice(label_to_indices[
                                           np.random.choice(
                                               list(labels_set - set([seq_label[i]]))
                                           )
                                       ]),
                   0]
                  for i in range(1, len(seq_list), 2)]

# positive pairs of contig
contig_positive_pairs = []
for pos_pair in positive_pairs:
    # first contig of pair
    contig = seq_list[pos_pair[0]]
    contig_positive_pairs.append(SeqRecord(
        contig,
        id='P' + str(pos_pair[0]) + '|' +
           seq_label[pos_pair[0]] + seq_idx[pos_pair[0]] + '-'
           + seq_label[pos_pair[1]] + seq_idx[pos_pair[1]] + '|',
        description="pairs of contig from same genome",
    ))

# negative pairs of contig
contig_negative_pairs = []
for neg_pair in negative_pairs:
    # first contig of pair
    contig = seq_list[neg_pair[0]]
    contig_negative_pairs.append(SeqRecord(
        contig,
        id='N' + str(neg_pair[0]) + '|' +
           seq_label[neg_pair[0]] + seq_idx[neg_pair[0]] + '-'
           + seq_label[neg_pair[1]] + seq_idx[neg_pair[1]] + '|',
        description='pairs of contig from different genome',
    ))

# output: write contigs into file
SeqIO.write(contig_positive_pairs, "../benchmarks/positive_test.fa", "fasta")
SeqIO.write(contig_negative_pairs, "../benchmarks/negative_test.fa", "fasta")
