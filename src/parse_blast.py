from Bio.Blast import NCBIXML

result_handle = open("../HMPdata/blast_reports/myseq0.xml")
blast_records = NCBIXML.parse(result_handle)

genome_names = []

for blast_record in blast_records:
    genome_names.append(blast_record.alignments[0].title + '\n')

with open('../HMPdata/blast_reports/myseq0.txt', 'a') as file:
    file.writelines(genome_names)