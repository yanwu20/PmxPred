# PmxPred

As the emergence of multidrug-resistant gram-negative bacteria raised a worldwide concern on human health, polymyxin has recently re-merged in the clinical practice due to its high antimicrobial activity of multidrug-resistant bacteria. However, the nephrotoxicity and neurotoxicity of polymyxin seriously hindered its practical use in clinic. Based on the quantitative structure-activity relationship (QSAR), analogue design is an efficient strategy for discovering biologically active compounds with fewer adverse effects. To speed up the polymyxin analogues discovery process and find the polymyxin analogues with high antimicrobial activity against gram-negative bacteria, here we developed PmxPred, a SMILES-based machine learning framework. Morgan fingerprint and physicochemical descriptors were used for the molecular representation and the ensemble machine learning model is utilized for the antimicrobial activity prediction. This framework was trained and evaluated on gram-negative bacteria dataset, including A. baumannii, E. coli, K. pneumoniae, P. aeruginosa and a general gram-negative bacteria dataset achieving an AUROC of 0.819, 0.756, 0.933, 0.849 and 0.924 respectively on the independent test. PmxPred outperformed the transfer learning method that trained on 10 million molecules. 

![Figure 1 Framework](https://user-images.githubusercontent.com/49023946/172044872-63d904f5-70c3-4899-855e-94cecc442a21.png)

## Dependency
* python 3.8
* xgboost 0.90
* scikit_learn 0.22.2 post1
* R 
* rpy2

## Dataset
* A. baumannii ATCC 19606_Binary.csv
* E. coli ATCC 25922_Binary.csv
* K. pneumoniae ATCC 13883_Binary.csv
* P. aeruginosa ATCC 27853_Binary.csv
* all_Binary.csv


## Usage


```
* Make the prediction 
```
python PmxPred.py --type predict --isolate <isolate name> --SMILES_file <SMILES file>  
```



* output results format

||Morgan figerprint|physicochemical descriptor|mean|pred|
| ---------- | :-----------:  | :-----------: | :-----------: | :-----------: |
|sequence name|score of Morgan fingerprint model|score of physicochemical descriptor model|final score|predict class|

## Reference
  
