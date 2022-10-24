import argparse
import warnings
import pandas as pd
from rdkit.Chem import AllChem,Descriptors
from rdkit import Chem
import pickle
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from rdkit.ML.Descriptors import MoleculeDescriptors
from imblearn.pipeline import Pipeline as imbpipeline


def FingerPrint(mols):
    fingerprints = []
    safe = []
    for mol_idx, mol in enumerate(mols):
        fingerprint = [x for x in AllChem.GetMorganFingerprintAsBitVect(mol, 3, 2048)]
        fingerprints.append(fingerprint)
        safe.append(mol_idx)
    fp_MorganFingerprint = pd.DataFrame(fingerprints).fillna(0)
    return fp_MorganFingerprint


def Smile2FP(df):
    mols = [Chem.MolFromSmiles(m.values[0]) for i, m in df.iterrows()]
    fp_MorganFingerprint = FingerPrint(mols)
    # fp_MorganFingerprint["label"] =df["Label"]
    return fp_MorganFingerprint


def calc_chiral(mol):
    chiral_list = []
    for i, atom in enumerate(mol.GetAtoms()):
        chiral_list.append(str(atom.GetChiralTag()))
    return Counter(chiral_list)


def Smile2RDK(df):
    chiral_name = ["CHI_OTHER", "CHI_TETRAHEDRAL_CCW", "CHI_TETRAHEDRAL_CW", "CHI_UNSPECIFIED"]
    des_name = [x[0] for x in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(des_name)
    out_list = []
    mols = [Chem.MolFromSmiles(m.values[0]) for i,m in df.iterrows()]
    for mol in mols:
        if mol:
            descriptors = list(calculator.CalcDescriptors(mol))
            chiral_dict = calc_chiral(mol)
            for k in chiral_name:
                descriptors.append(chiral_dict[k])
            out_list.append(descriptors)
    des_name.extend(chiral_name)
    df_out = pd.DataFrame(out_list, columns=des_name)
    # df_out["label"] = df["Label"]
    return df_out


def prediction(X_RDK, X_MFP, isolate):
    model = pickle.load(open("./model", "rb"))
    RDK_model = model[isolate]["RDK"]
    MFP_model = model[isolate]["FP"]

    output_dic = dict()
    output_dic["RDK"] =RDK_model.predict_proba(X_RDK)[:,1]
    output_dic["FP"] = MFP_model.predict_proba(X_MFP)[:,1]
    df_level1 = pd.DataFrame(output_dic)
    df_level1["score"] = 0.5 * df_level1["RDK"] + 0.5 * df_level1["FP"]
    df_level1["Label"] = np.rint(df_level1["score"])
    return df_level1


if __name__ == "__main__":
    # PmxPred

    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description='Prediction')
    parser.add_argument("--type", required=True,
                        choices=["benchmark", "predict"],
                        help="the encoding type")
    parser.add_argument("--strain", required=True,
                        choices=["A. baumannii ATCC 19606", "E. coli ATCC 25922", "K. pneumoniae ATCC 13883",
                                 "P. aeruginosa ATCC 27853", "all"],
                        help="the name of strain")
    parser.add_argument('--SMILES_file', type=str, help='input SMILES file')
    parser.add_argument('--output_name', type=str, help='output file name')
    args = parser.parse_args()

    if args.type == "benchmark":
        df_input = pd.read_csv("./polymyxin data/%s_Binary.csv" % args.strain)
        X, X_test, y, y_test = train_test_split(df_input.iloc[:, :-1], df_input.iloc[:, -1], test_size=0.2,
                                                random_state=0, stratify=df_input.iloc[:, -1])
        df_MFP = Smile2FP(X_test)
        df_RDK = Smile2RDK(X_test)
        df_result = prediction(df_RDK, df_MFP, args.strain)
        df_result.to_csv(args.output_name)
        print(df_result)
    else:
        df_input = pd.read_csv(args.SMILES_file)
        df_MFP = Smile2FP(df_input)
        df_RDK = Smile2RDK(df_input)
        df_result = prediction(df_RDK, df_MFP, args.strain)
        df_result.to_csv(args.output_name)
        print(df_result)
