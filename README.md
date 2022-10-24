# PmxPred

The emergence of multidrug-resistant Gram-negative bacteria presents a worldwide threat to human health; over recent last decades, polymyxins have re-emerged in clinical practice due to their high activity against multidrug-resistant bacteria. However, the nephrotoxicity and neurotoxicity of polymyxins seriously hinder their practical use in the clinic. Based on the quantitative structure-activity relationship (QSAR), analogue design is an efficient strategy for discovering biologically active compounds with fewer adverse effects. To accelerate the polymyxin analogues discovery process and find the polymyxin analogues with high antimicrobial activity against Gram-negative bacteria, here we developed PmxPred, a Simplified Molecular Input Line Entry System (SMILES)-based machine learning framework. Morgan fingerprint and RDKit descriptors were used for the molecular representation, and the ensemble learning model was utilized for the antimicrobial activity prediction. This framework was trained and evaluated on multiple Gram-negative bacteria datasets, including Acinetobacter baumannii, Escherichia coli, Klebsiella pneumoniae, Pseudomonas aeruginosa and a general Gram-negative bacteria dataset achieving an AUROC of 0.872, 0.907, 0.947, 0.899 and 0.873 on the independent test, respectively. PmxPred outperformed the transfer learning method that trained on 10 million molecules.


<div align=center><img  src ="https://user-images.githubusercontent.com/49023946/193976363-26da793b-7cc9-4d2e-a0e1-3c3c7cae64cc.png" alt="Framework of PmxPred"></div>




## Dependency
* python 3.8
* xgboost 1.5.2 
* scikit_learn 0.22.2 post1
* imbalanced-learn 0.9.1
* rdkit 2022.3.5
* catboost 1.1

## Dataset
* A. baumannii ATCC 19606_Binary.csv
* E. coli ATCC 25922_Binary.csv
* K. pneumoniae ATCC 13883_Binary.csv
* P. aeruginosa ATCC 27853_Binary.csv
* all_Binary.csv



## Installation Guide

*  Install from Github 
```python
git clone https://github.com/yanwu20/PmxPred.git
cd PmxPred
pip install -r requirements.txt
```

*Benchmark
```
python PmxPred.py --type benchmark --strain <strain name> 
```

* Make the prediction 
```
python PmxPred.py --type predict --strain <strain name> --SMILES_file <SMILES file>  
```



* output results format

||Morgan figerprint|RDKit descriptor|mean|pred|
| ---------- | :-----------:  | :-----------: | :-----------: | :-----------: |
|id|score of Morgan fingerprint model|score of RDKit descriptor model|final score|predict class|

## Reference
  
