# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 19:56:10 2020

@author: Liz
"""
import random
import pandas as pd
import networkx as nx
from sklearn.metrics import roc_auc_score

#load train data
with open('train.txt') as f:
    node_list_1 = []
    node_list_2 = []
    for line in f:
        node_list = line.rstrip().split()
        len_node_list = len(node_list)
        node_list_1.extend([node_list[0]] * (len_node_list-1))
        node_list.pop(0)
        node_list_2.extend(node_list)
    f.close()
graph_df = pd.DataFrame({'node_1': node_list_1, 'node_2': node_list_2})

# create graph
G = nx.from_pandas_edgelist(graph_df, "node_1", "node_2", create_using=nx.Graph())

n = G.number_of_nodes()
m = G.number_of_edges()

edge_subset = random.sample(G.edges(), int(0.25 * G.number_of_edges()))

dev_set = pd.read_csv('dev.csv')
dev_edges = []
for i in range(len(dev_set)):
    dev_edges.append(tuple([str(dev_set.iloc[i,1]),str(dev_set.iloc[i,2])]))

dev_label = pd.read_csv('dev-labels.csv')
dev_y = list(dev_label['Expected'])

#jaccard coefficient
prediction_jaccard = list(nx.jaccard_coefficient(G))
j_score = [s for (u,v,s) in prediction_jaccard]

all_edges = []
for (u,v,s) in prediction_jaccard:
    all_edges.append((u,v))

pred = []
for i in range(len(dev_edges)):
    try:
        pred.append(j_score[all_edges.index(dev_edges[i])])
    except ValueError:
        try:
            pred.append(j_score[all_edges.index(tuple([dev_edges[i][1],dev_edges[i][0]]))])
        except ValueError:
            pred.append(-1)

pred_fill = [random.random() if x==-1 else x for x in pred]
j_roc_score = roc_auc_score(dev_y, pred)

#Adamic-Adar index
prediction_adamic = list(nx.adamic_adar_index(G))
aa_score = [s for (u,v,s) in prediction_adamic]

all_edges = []
for (u,v,s) in prediction_adamic:
    all_edges.append((u,v))
    
aa_pred = []
for i in range(len(dev_edges)):
    try:
        aa_pred.append(aa_score[all_edges.index(dev_edges[i])])
    except ValueError:
        try:
            aa_pred.append(aa_score[all_edges.index(tuple([dev_edges[i][1],dev_edges[i][0]]))])
        except ValueError:
            aa_pred.append(-1)

aa_pred_fill = [random.random() if x==-1 else x for x in aa_pred]
aa_roc_score = roc_auc_score(dev_y, aa_pred)

#Preferential Attachment
prediction_pref = list(nx.preferential_attachment(G))
pa_score = [s for (u,v,s) in prediction_pref]
all_edges = []
for (u,v,s) in prediction_pref:
    all_edges.append((u,v))
    
pa_pred = []
for i in range(len(dev_edges)):
    try:
        pa_pred.append(pa_score[all_edges.index(dev_edges[i])])
    except ValueError:
        try:
            pa_pred.append(pa_score[all_edges.index(tuple([dev_edges[i][1],dev_edges[i][0]]))])
        except ValueError:
            pa_pred.append(-1)

pa_pred_fill = [random.random() if x==-1 else x for x in pa_pred]
pa_roc_score = roc_auc_score(dev_y, pa_pred_fill)