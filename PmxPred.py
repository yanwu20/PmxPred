import argparse
import warnings
import pandas as pd
import pickle
import numpy as np
import torch.nn.functional as F
from utils import *
from torch_geometric.loader import DataLoader



def file2test(df, feature_name):
    all_residue = []
    index_list = []
    for i, smile in df.iterrows():
        data = Smile2Graph_global_ali_new(smile[0], 0, feature_name)
        if data:
            all_residue.append(data)
            index_list.append(i)
        else:
            print("number %s is not polymyxin analogue" %i)
    return all_residue, index_list

def file2data_ali_new(df, feature_name):
    all_residue = []
    for i, (smile, y) in df.iterrows():
        data = Smile2Graph_global_ali_new(smile, y, feature_name)
        if data:
            all_residue.append(data)
    return all_residue

def test(test_loader, model):
    out_list = []
    model.eval()
    for data in test_loader:  # Iterate in batches over the training/test dataset.
        with torch.no_grad():
            data = data.cuda()
            x = F.normalize(data.x, p=1.0, dim=0)
            out = model(x, data.edge_index, data.batch)
            out_list.extend(out.tolist())
    node_score = [item for sublist in out_list for item in sublist]
    return node_score


if __name__ == "__main__":
    # PmxPred

    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description='Prediction')
    parser.add_argument("--type", required=True,
                        choices=["benchmark", "predict"],
                        help="the encoding type")
    parser.add_argument("--strain", required=True,
                        choices=["A. baumannii ATCC 19606", "E. coli ATCC 25922", "K. pneumoniae ATCC 13883",
                                 "P. aeruginosa ATCC 27853", "Gram negative"],
                        help="the name of strain")
    parser.add_argument("--combine",
                        choices=["LR", "Min"],default = "LR",
                        help="the combine type")
    parser.add_argument('--SMILES_file', type=str, help='input SMILES file')
    parser.add_argument('--output_name', type=str, help='output file name')
    args = parser.parse_args()

    batch_size = 4

    if args.type == "benchmark":
        df_input = pd.read_csv("./Data/%s_Binary_test.csv" % args.strain, index_col=0)
        test_set = file2data_ali_new(df_input, "Des")
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        test_label = [d.y.item() for d in test_set]
        index_list = df_input.index

    else:
        df_input = pd.read_csv(args.SMILES_file)
        # print(df_input.shape)
        test_set, index_list = file2test(df_input, "Des")
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # node-level model
    model = torch.load("./models/GCN_ali_%s_" % args.strain).cuda()
    node_score = test(test_loader, model)

    # global feature model
    with open("./cv_results/catBoost_%s_model" % args.strain, "rb") as f:
        global_cat_model = pickle.load(f)

    global_x_test = [d.global_feature.tolist() for d in test_set]
    global_score = global_cat_model.predict_proba(global_x_test)[:, 1]

    df_result = pd.DataFrame([global_score, node_score], columns=index_list, index=["Global socre","node score"]).T

    combine_type = "LR"
    if combine_type == "Min":
        score_list = [min(s1, s2) for s1, s2 in zip(global_score, node_score)]
        pred_list = [1 if s > 0.5 else 0 for s in score_list]
    elif combine_type == "Mean":
        score_list = [(s1+s2)/2 for s1, s2 in zip(global_score, node_score)]
        pred_list = [1 if s > 0.5 else 0 for s in score_list]

    elif combine_type == "LR":
        with open("./models/LR_%s" % args.strain, "rb") as f:
            clf = pickle.load(f)
        score_list = clf.predict_proba(df_result)[:, 1]
        pred_list = clf.predict(df_result)

    df_result["Final"] = score_list
    # df_result.to_csv(args.output_name)
    print(df_result)
