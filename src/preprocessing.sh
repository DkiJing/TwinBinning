#!/bin/bash
echo "Cut contigs..."
src="/home/jli347/src"
species="/home/jli347/data/test_dataset/species"
contigs="/home/jli347/data/test_dataset/output_contigs"
kmer="/home/jli347/data/test_dataset/3-mer/kmer.csv"
train="/home/jli347/data/test_dataset/3-mer/train.csv"
cov="/home/jli347/data/test_dataset/3-mer/abundance.csv"
covfreq="/home/jli347/data/test_dataset/3-mer/abundance_profile.csv"
covmyout="/home/jli347/data/test_dataset/cov_diff_5/myout"
for i in `ls $species`
do
  python contig_generation.py -s $species/$i/`ls "$species/$i"` -b 4000 -o $contigs/$i.fa
done
echo "Kmer extraction..."
label=0
for i in `ls $contigs`
do
  python feature_vector.py -i $contigs/$i -k 3 -l $label -o $kmer
  label=$((label+1))
done
echo "Generate abundance profile..."
cat $contigs/*.fa > $covmyout/Bacteroides_pairs.fa 
cd $covmyout
bowtie2-build Bacteroides_pairs.fa myout.idx 1>myout.sam.bowtie2build.out 2>myout.sam.bowtie2build.err
bowtie2 -p 4 -x myout.idx -U Bacteroides_all_reads.fq -S myout.sam 1>myout.sam.bowtie2.out 2>myout.sam.bowtie2.err
samtools view -bS myout.sam > myout.bam
samtools sort myout.bam -o myout.sorted.bam
samtools depth myout.sorted.bam > output.txt
cd $src
python abundance_vector.py -l 4000 -i $covmyout/output.txt -o $cov
python abundance_profile.py -i $cov -o $covfreq
python feature_combined_vector.py -k $kmer -a $covfreq -o $train 
