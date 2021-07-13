#!/bin/bash
contigs="/home/jli347/data/test_dataset/output_contigs"
benchmark="/home/jli347/benchmarks"
covdepth="/home/jli347/benchmarks/cov_diff_5/metabat2/sample"
reads="/home/jli347/benchmarks/cov_diff_5/reads"
output_bins="/home/jli347/benchmarks/cov_diff_5/metabat2/myout"

echo "Generate abundance profile for MetaBAT2..."
cat $contigs/*.fa > $benchmark/Bacteroides_pairs_combined.fa 
cd $covdepth
bowtie2-build $benchmark/Bacteroides_pairs_combined.fa myout.idx 1>myout.sam.bowtie2build.out 2>myout.sam.bowtie2build.err
bowtie2 -p 4 -x myout.idx -U $reads/Bacteroides_all_reads.fq -S myout.sam 1>myout.sam.bowtie2.out 2>myout.sam.bowtie2.err
samtools view -bS myout.sam > myout.bam
samtools sort myout.bam -o myout.sorted.bam
echo "MetaBAT2 binning..."
jgi_summarize_bam_contig_depths --outputDepth depth.txt myout.sorted.bam
metabat2 -i $benchmark/Bacteroides_pairs_combined.fa -a depth.txt -o $output_bins/bin
