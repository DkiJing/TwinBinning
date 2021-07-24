#!/bin/bash
parentdir="$(dirname `pwd`)"
src=$parentdir/src
contigs=$parentdir/data/test_dataset/output_contigs
threemer=$parentdir/data/test_dataset/3-mer/kmer.csv
fourmer=$parentdir/data/test_dataset/4-mer/test.csv
feature=$parentdir/data/test_dataset/3-mer/test.csv
covfreq=$parentdir/data/test_dataset/3-mer/abundance_profile.csv
covmyout=$parentdir/data/test_dataset/coverage
benchmark=$parentdir/benchmarks
contig_len=4000

echo "Kmer extraction..."
label=0
for i in `ls $contigs`
do
  python feature_vector.py -i $contigs/$i -k 3 -l $label -o $threemer
  python feature_vector.py -i $contigs/$i -k 4 -l $label -o $fourmer
  label=$((label+1))
done
echo "Generate abundance profile..."
cd $covmyout
samtools depth $benchmark/coverage/metabat2/sample/myout.sorted.bam > output.txt
average=`awk '{sum+=$3} END{print sum/NR}' output.txt`
echo "Average depth is $average"
cd $src
python abundance_profile.py -l $contig_len -i $covmyout/output.txt -o $covfreq -m $average
python feature_combined_vector.py -k $threemer -a $covfreq -o $feature 
