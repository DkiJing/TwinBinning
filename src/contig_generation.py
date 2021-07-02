import argparse
import random
from re import sub
from Bio import SeqIO
import math
import sys
from Bio.Seq import Seq

from Bio.SeqRecord import SeqRecord

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('-s', '--seq', type=str, default=None)
parser.add_argument('-b', '--base', type=int, default=4000)
parser.add_argument('-t', '--threshold', type=int, default=0)
parser.add_argument('-o', '--output', type=str, default=None)
parser.add_argument('-min', '--minbase', type=int, default=2000)
args = parser.parse_args()

# parameters
seqName = args.seq
substr_base_len = args.base
substr_threshold_len = args.threshold
substr_min_len = args.minbase

# get substring
def get_random_str(main_str, start, substr_len):
    sub_seq = main_str[start : (start + substr_len)]
    if "N" in sub_seq:
        sub_seq = sub_seq.tomutable()
        sub_seq = str(sub_seq)
        sub_seq = sub_seq.replace('N', '')
        sub_seq = Seq(sub_seq)
    else:
        # simulate sequence mutation with probability 0.001
        if random.random() < 0.001:
            r = random.random()
            sub_seq = sub_seq.tomutable()
            if r < 0.2:
                sub_seq[random.randint(0, substr_len - 1)] = "A"
            if 0.2 <= r < 0.4:
                sub_seq[random.randint(0, substr_len - 1)] = "T"
            if 0.4 <= r < 0.6:
                sub_seq[random.randint(0, substr_len - 1)] = "C"
            if 0.6 <= r < 0.8:
                sub_seq[random.randint(0, substr_len - 1)] = "G"
            if r >= 0.8:
                if 0.8 < r <= 0.85:
                    sub_seq.remove("A")
                if 0.85 < r <= 0.9:
                    sub_seq.remove("T")
                if 0.9 < r <= 0.95:
                    sub_seq.remove("C")
                if r > 0.95:
                    sub_seq.remove("G")
            sub_seq = sub_seq.toseq()
    return sub_seq

# get main sequence
seq_list = [s.seq for s in SeqIO.parse(seqName, "fasta")]
sequence = Seq('').join(seq_list)

# list of contigs
contigs = []

penality = 0
# the number of contigs
iter_times = math.floor(len(sequence) / (substr_base_len + substr_threshold_len))
for i in range(iter_times):
    sub_seq = get_random_str(sequence, i * substr_base_len, substr_base_len + random.randint(-substr_threshold_len, substr_threshold_len))
    idx = str(i - penality)
    if(len(sub_seq) == 0 or len(sub_seq) < substr_min_len):
        penality += 1
        continue
    contigs.append(SeqRecord(
        sub_seq,
        id=idx + '|' + seqName.split('/')[-2] + '|',
        description=seqName.split('/')[-2],
))

# output: write contigs into file
SeqIO.write(contigs, args.output, "fasta")


