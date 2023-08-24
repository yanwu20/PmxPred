# PmxPred

The multidrug-resistant Gram-negative bacteria has evolved into a worldwide threat to human health; over recent decades, polymyxins have re-emerged in clinical practice due to their high activity against multidrug-resistant bacteria. Nevertheless, the nephrotoxicity and neurotoxicity of polymyxins seriously hinder their practical use in the clinic. Based on the quantitative structure-activity relationship (QSAR), analogue design is an efficient strategy for discovering biologically active compounds with fewer adverse effects. To accelerate the polymyxin analogues discovery process and find the polymyxin analogues with high antimicrobial activity against Gram-negative bacteria, here we developed PmxPred, a GCN and catBoost-based machine learning framework. The RDKit descriptors were used for the molecule and residues representation, and the ensemble learning model was utilized for the antimicrobial activity prediction. . This framework was trained and evaluated on multiple Gram-negative bacteria datasets, including Acinetobacter baumannii, Escherichia coli, Klebsiella pneumoniae, Pseudomonas aeruginosa and a general Gram-negative bacteria dataset achieving an AUROC of 0.857, 0.880, 0.756, 0.895 and 0.865 on the independent test, respectively. PmxPred outperformed the transfer learning method that trained on 10 million molecules.


<div align=center><img  src ="https://github.com/yanwu20/PmxPred/assets/49023946/ae556954-be44-4552-89c7-0036496b5ef4" alt="Framework of PmxPred"></div>



## Dependency
* python 3.8
* scikit_learn 0.22.2 post1
* imbalanced-learn 0.9.1
* rdkit 2022.3.5
* catboost 1.1
* torch_geometric 2.3.0

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

||Global score|Node score|final score|
| ---------- | :-----------:  | :-----------: | :-----------: | 
|id|score of global model|score of residue model|final score|

## Reference
  
