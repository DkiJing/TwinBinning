#!/bin/bash
src="/home/jli347/src"
species="/home/jli347/data/10_genomes"
contigs="/home/jli347/output_contigs"
metabat_contigs="/home/jli347/benchmarks/cov_diff_5/metabat2/myout"
threemer="/home/jli347/data/3-mer/kmer.csv"
fourmer="/home/jli347/data/4-mer/train.csv"
feature="/home/jli347/data/3-mer/train.csv"
cov="/home/jli347/data/3-mer/abundance.csv"
covfreq="/home/jli347/data/3-mer/abundance_profile.csv"
covsample="/home/jli347/benchmarks/cov_diff_5/metabat2/sample_input"
reads="/home/jli347/benchmarks/cov_diff_5/reads"

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
cat $metabat_contigs/*.fa > $covsample/Bacteroides_pairs.fa 
cd $covsample
bowtie2-build Bacteroides_pairs.fa myout.idx 1>myout.sam.bowtie2build.out 2>myout.sam.bowtie2build.err
bowtie2 -p 4 -x myout.idx -U $reads/Bacteroides_all_reads.fq -S myout.sam 1>myout.sam.bowtie2.out 2>myout.sam.bowtie2.err
samtools view -bS myout.sam > myout.bam
samtools sort myout.bam -o myout.sorted.bam
samtools depth myout.sorted.bam > output.txt
average=`awk '{sum+=$3} END{print sum/NR}' output.txt`
echo "Average depth is $average"
cd $src
python abundance_vector.py -l 4000 -i $covsample/output.txt -o $cov
python abundance_profile.py -i $cov -o $covfreq -m $average
python feature_combined_vector.py -k $threemer -a $covfreq -o $feature 
