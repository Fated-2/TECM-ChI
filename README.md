# TECM-ChI


## Table of Contents

1. [Introduction](#introduction)
2. [Python Environment](#python-environment)
3. [Project Structure](#Project-Structure)
   - [Framework](#Framework)
   - [Dataset](#Dataset)
   - [Code](#Code)
   - [Test](#Test)
---

## 1. Introduction

We propose a novel deep learning model called TECM-ChI, which combines gene sequences and genomic features to identify chromatin interactions. First, we designed the FCR method within the model to balance positive and negative samples from the K562, IMR90, and GM12878 datasets, achieving a 1:1 ratio. To effectively extract relevant information from gene sequences, we developed a preprocessing Three-Encoding module that converts gene sequences into 45-dimensional feature vectors using three encoding methods (KNF + NAD + NCS). Secondly, we introduced the CMANet network model, which integrates multiple convolutional layers with attention mechanisms; CMANet effectively captures local features from sequence information and enhances focus on critical regions, improving the ability to identify chromatin interactions. Furthermore, to evaluate the effectiveness of TECM-ChI, we conducted model variant experiments and loss performance analysis, comparing it with existing computational methods across three cell lines. Experimental results demonstrate that TECM-ChI improves accuracy by 4.68%, 1.31%, and 2.41% on the K562, IMR90, and GM12878 datasets, respectively, compared to the current optimal models. Here, we provide the code for implementing, training, and testing BERT-TFBS.

## 2. Python Environment

Python 3.6 and packages version:

- tensorflow-gpu=2.0.0
- keras=2.3.1
- scikit-learn  
- imbalanced-learn
- numpy  
- h5py
- matplotlib

## 3. Project Structure

### 3.1 Framework

![模型架构](https://github.com/Fated-2/TECM-ChI/blob/main/model/model.png)

### 3.2 Dataset

- The raw data files are from https://github.com/shwhalen/tf2.

- The folder "data" contains the DNA sequences in ".bed" format and labels. You can use the Bedtools tool to convert a BED file to a FASTA file.  

- The folder "feature" contains other features, including genomic features, distance, CTCF motif and conservation score.  

- The file "K562_sequence.rar" is an example file for the K562 cell line, containing DNA sequences in ".fasta" format and labels. Used to test the code.  

- The file "K562_genomics.rar" is an example file for the K562 cell line, containing genomic features and other features. Used to test the code.  

### 3.3 Code
- The "encoding.py" code is used for encoding sequences.  
- The "data_load.py" code is used to read sequences.  
- The "model.py" code is used to construct the model architecture.  
- The "TECM-ChI.py" code is used to train and validate the TECM-ChI model and evaluate its performance.  

### 3.4 Test
- Use the K562 dataset for code testing. Run the following script to compile and execute TECM-ChI: 
	
  ```bash
	python TECM-ChI.py
	```
	
- At least 64GB of memory is required.