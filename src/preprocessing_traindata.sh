#!/bin/bash
parentdir="$(dirname `pwd`)"
src=$parentdir/src
species=$parentdir/data/baseline
contigs=$parentdir/output_contigs
metabat_contigs=$parentdir/benchmarks/metabat2/myout
threemer=$parentdir/data/3-mer/kmer.csv
fourmer=$parentdir/data/4-mer/train.csv
feature=$parentdir/data/3-mer/train.csv
covfreq=$parentdir/data/3-mer/abundance_profile.csv
covsample=$parentdir/benchmarks/metabat2/sample_input
reads=$parentdir/benchmarks/reads
contig_len=4000

echo "Cut contigs..."
for i in `ls $species`
do
  python contig_generation.py -s $species/$i/`ls "$species/$i"` -b 4000 -o $contigs/$i.fa
done
echo "Kmer extraction..."
label=0
for i in `ls $contigs`
do
  python feature_vector.py -i $contigs/$i -k 4 -l $label -o $fourmer
  label=$((label+1))
done
count=$label
for i in `ls $metabat_contigs`
do
  python feature_vector.py -i $metabat_contigs/$i -k 4 -l $label -o $fourmer
  python feature_vector.py -i $metabat_contigs/$i -k 3 -l $((label-count)) -o $threemer
  label=$((label+1))
done
echo "Generate abundance profile..."
cat $metabat_contigs/*.fa > $covsample/combined_pairs.fa 
cd $covsample
bowtie2-build combined_pairs.fa myout.idx 1>myout.sam.bowtie2build.out 2>myout.sam.bowtie2build.err
bowtie2 -p 4 -x myout.idx -U $reads/combined_all_reads.fq -S myout.sam 1>myout.sam.bowtie2.out 2>myout.sam.bowtie2.err
samtools view -bS myout.sam > myout.bam
samtools sort myout.bam -o myout.sorted.bam
samtools depth myout.sorted.bam > output.txt
average=`awk '{sum+=$3} END{print sum/NR}' output.txt`
echo "Average depth is $average"
cd $src
python abundance_profile.py -l $contig_len -i $covsample/output.txt -o $covfreq -m $average
python feature_combined_vector.py -k $threemer -a $covfreq -o $feature 
