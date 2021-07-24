#!/bin/bash
parentdir="$(dirname `pwd`)"
species=$parentdir/data/test_dataset/selected_species
contigs=$parentdir/data/test_dataset/output_contigs
benchmark=$parentdir/benchmarks
covdepth=$parentdir/benchmarks/metabat2/sample
reads=$parentdir/benchmarks/coverage/reads
output_bins=$parentdir/benchmarks/metabat2/myout

echo "Cut contigs..."
for i in `ls $species`
do
  python contig_generation.py -s $species/$i/`ls "$species/$i"` -b $contig_len -o $contigs/$i.fa
done
echo "Generate abundance profile for MetaBAT2..."
cat $contigs/*.fa > $benchmark/combined_pairs.fa 
cd $covdepth
bowtie2-build $benchmark/combined_pairs.fa myout.idx 1>myout.sam.bowtie2build.out 2>myout.sam.bowtie2build.err
bowtie2 -p 4 -x myout.idx -U $reads/combined_all_reads.fq -S myout.sam 1>myout.sam.bowtie2.out 2>myout.sam.bowtie2.err
samtools view -bS myout.sam > myout.bam
samtools sort myout.bam -o myout.sorted.bam
echo "MetaBAT2 binning..."
jgi_summarize_bam_contig_depths --outputDepth depth.txt myout.sorted.bam
metabat2 -i $benchmark/combined_pairs.fa -a depth.txt -o $output_bins/bin