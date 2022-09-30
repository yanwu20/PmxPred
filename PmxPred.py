import argparse
import warnings
import pandas as pd
from rdkit.Chem import AllChem
from rdkit import Chem
import pickle
import numpy as np


def FingerPrint(mols):
    fingerprints = []
    safe = []
    for mol_idx, mol in enumerate(mols):
        fingerprint = [x for x in AllChem.GetMorganFingerprintAsBitVect(mol, 3, 2048)]
        fingerprints.append(fingerprint)
        safe.append(mol_idx)
    fp_MorganFingerprint = pd.DataFrame(fingerprints).fillna(0)
    return fp_MorganFingerprint


def Smile2FP(FileName):
    df_input = pd.read_csv(FileName)
    mols = [Chem.MolFromSmiles(m) for i, m in df_input.iloc[:, 0].items()]
    fp_MorganFingerprint = FingerPrint(mols)
    fp_MorganFingerprint["label"] = df_input["Label"]
    return fp_MorganFingerprint


def Smile2CDK(FileName):
    r = robjects.r
    r(
        '''
        library("rcdk")
        f <- function(r){
         input_data<-read.csv(r)
         Smiles<-input_data[1]
         dc <- get.desc.categories()
         desc_list<-c("org.openscience.cdk.qsar.descriptors.molecular.BCUTDescriptor")
         for (num in 2:5){
             dn<- get.desc.names(dc[num])
             for (i in 1:length(dn)){
                     desc_list[length(desc_list)+1]<-dn[i]
             }
         }
         Smiles_v<-unlist(Smiles)
         m <- parse.smiles(Smiles_v)
         descs <- eval.desc(m, desc_list)
         return(descs)
         }

        '''
    )
    df_CDK = r['f'](FileName)

    return df_CDK


def prediction(df_CDK, df_MFP, isolate):
    model = pickle.load(open("./model/model", "rb"))
    CDK_model = model["CDK"]
    MFP_model = model["MFP"]
    model_RDK = CDK_model[isolate]
    model_FP = MFP_model[isolate]

    output_dic = dict()
    output_dic["RDK"] = model_RDK.predict_proba(df_CDK)[:, 1]
    output_dic["FP"] = model_FP.predict_proba(df_MFP)[:, 1]

    df_level1 = pd.DataFrame(output_dic)
    df_level1["score"] = 0.5 * df_level1["RDK"] + 0.5 * df_level1["FP"]
    df_level1["Label"] = np.rint(df_level1["score"])
    return df_level1


if __name__ == "__main__":
    # PmxPred

    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description='generate feature')
    parser.add_argument("--type", required=True,
                        choices=["benchmark", "predict"],
                        help="the encoding type")
    parser.add_argument("--isolate", required=True,
                        choices=["A. baumannii ATCC 19606", "E. coli ATCC 25922", "K. pneumoniae ATCC 13883",
                                 "P. aeruginosa ATCC 27853", "all"],
                        help="the name of isolate")
    parser.add_argument('--SMILES_file', type=str, help='input SMILES file')
    parser.add_argument('--output_name', type=str, help='output file name')
    args = parser.parse_args()

    if args.type == "benchmark":
        df_input = pd.read_csv("./polymyxin data/%s_Binary.csv" %args.isolates)
        print(prediction())
    else:
        df_MFP = Smile2FP(args.SMILES_file)
        df_CDK = Smile2CDK(args.SMILES_file)
        print(prediction(df_CDK=df_CDK, df_MFP=df_MFP, isolate=args.isolate))
