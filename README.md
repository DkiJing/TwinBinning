# TwinBinning  
This repository contains the srouce code of my master dissertation in CityU.  
## Prerequisite  
### Software  
+ Python 3.6 and above  
+ Pytorch 1.9.0  
+ CUDA 10.2 and above (optional)  
+ pandas  
+ numpy  
+ sklearn  
+ matplotlib  
+ BioPython  
+ samtools  
+ bowtie2  
+ MetaBAT2  
### Operating system  
Now we only support Linux and MacOS  
## Installation  
1. Download the source code  
```
git clone https://github.com/DkiJing/TwinBinning.git   
```  
2. Put contig "combined_pairs.fa" and reads "combined_all_reads.fq" file in corresponding positions  
```  
├── benchmarks  
│   ├── combined_pairs.fa   
│   └── reads  
│       └── combined_all_reads.fq  
```  
3. Run the program  
```  
cd src  
./metabat_binning.sh  
./preprocessing_testdata.sh  
./preprocessing_traindata.sh  
python main.py  
```  
4. Play the code "Transfer twin learning.ipynb" on jupyter notebook!  
