
import networkx as nx
import numpy as np
from LPMethod import similarities
import os  
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
import math
from ds_module import ECMLP_function

def layerNum(file):                   # Returning the number of layers.
    path = file[:file.rfind('/')]
    all_files = os.listdir(path)
    type_dict = dict()

    for f in all_files:
        if os.path.isdir(f):
            type_dict.setdefault('folder', 0)
            type_dict['folder'] += 1
        else:
            ext = os.path.splitext(f)[1]
            type_dict.setdefault(ext, 0)
            type_dict[ext] += 1
    return type_dict['.edgelist']



def CSL(a, b):                      #Calculating the similarity between layers.
    
    nodes = []                                          
    for va in nx.nodes(a):
        if va not in nodes:
            nodes.append(va)
    for vb in nx.nodes(b):
        if vb not in nodes:
            nodes.append(vb)        
                    
    matrix1 = np.zeros((len(nodes), len(nodes)))
    for i in range(len(nodes)):
        for j in range(i, len(nodes)):
            if pair(nodes[i], nodes[j]) in a.edges():
                matrix1[i][j] = 1
                matrix1[j][i] = 1
    avector = matrix1.flatten().tolist()
    matrix2 = np.zeros((len(nodes), len(nodes)))
    for i in range(len(nodes)):
        for j in range(i, len(nodes)):
            if pair(nodes[i], nodes[j]) in b.edges():
                matrix2[i][j] = 1
                matrix2[j][i] = 1
    bvector = matrix2.flatten().tolist()

    return (1 - cosine(avector, bvector))
 

def PCC(a, b):                                               #Calculating the similarity between layers.
    
    nodes = []                                         
    for va in nx.nodes(a):
        if va not in nodes:
            nodes.append(va)
    for vb in nx.nodes(b):
        if vb not in nodes:
            nodes.append(vb)        
                    
    matrix1 = np.zeros((len(nodes), len(nodes)))
    for i in range(len(nodes)):
        for j in range(i, len(nodes)):
            if pair(nodes[i], nodes[j]) in a.edges():
                matrix1[i][j] = 1
                matrix1[j][i] = 1
    avector = matrix1.flatten().tolist()
    matrix2 = np.zeros((len(nodes), len(nodes)))
    for i in range(len(nodes)):
        for j in range(i, len(nodes)):
            if pair(nodes[i], nodes[j]) in b.edges():
                matrix2[i][j] = 1
                matrix2[j][i] = 1
    bvector = matrix2.flatten().tolist()

    return pearsonr(avector, bvector)[0]


def GOR(a, b):                                          #Calculating the similarity between layers.
    
    G1 = a
    a_edge_num = nx.number_of_edges(G1)
    a_edge_list = [pair(u, v) for u, v in nx.edges(G1)]
    G2 = b
    b_edge_num = nx.number_of_edges(G2)
    b_edge_list = [pair(u, v) for u, v in nx.edges(G2)]
    overlap = 0
    for (u, v) in a_edge_list:
        if (u, v) in b_edge_list:
            overlap += 1
    return 2 * overlap / (a_edge_num + b_edge_num)
    

def pair(x, y):                                                                          
    if (x < y):
        return (x, y)
    else:
        return (y, x)
 
  
        
def ds_sim_function(train_graph, method, alpha, rele, graph):                  # Returning the similarity scores of node pairs              
    
    ln = layerNum(graph)  
    
    nodes1 = []
    for i0 in range(ln):
        G0 = nx.read_edgelist(graph + str(i0 + 1) + '.edgelist', nodetype = int)                                         
        for va0 in nx.nodes(G0):
            if va0 not in nodes1:
                nodes1.append(va0)

    node_num = len(nodes1)                                                                                     
    non_edge_list1 = [pair(u1, v1) for u1, v1 in nx.non_edges(train_graph)]
    edge_list1 = [pair(u2, v2) for u2, v2 in nx.edges(train_graph)]
   
    level_sim_dic = {}
    level_rele_dic = {}
    
    for i in range(ln):
        if i + 1 == alpha:                                        #The target layer.               
            sim_dic_method = similarities(train_graph, method)
            for non_edge1 in non_edge_list1:
                sim_dic_method[non_edge1] = sim_dic_method[non_edge1] if non_edge1 in sim_dic_method.keys() else 0
            level_sim_dic[i+1] = sim_dic_method    
            level_rele_dic[i+1] = 1
        else:                                                    # The auxiliary layers.                                                                                       
            sim_dic_auxiliary = {}
            G = nx.read_edgelist(graph + str(i + 1) + '.edgelist', nodetype = int)
            
            s1 = similarities(G, method)
            s2 = similarities(G, method, True)
            s1.update(s2)
            
            for nodepair, sim1 in s1.items():
                s1[nodepair] = sim1 / node_num
               
            for non_edge2 in non_edge_list1:
                sim_dic_auxiliary[non_edge2] = 1 if non_edge2 in G.edges() else 0
                if non_edge2 in s1.keys():
                    sim_dic_auxiliary[non_edge2] += s1[non_edge2]
                    
            level_sim_dic[i+1] = sim_dic_auxiliary
            if rele == 'CSL':
                level_rele_dic[i+1] = CSL(train_graph, G)
            if rele == 'PCC':
                level_rele_dic[i+1] = PCC(train_graph, G)  
            if rele == 'GOR':
                level_rele_dic[i+1] = GOR(train_graph, G)    
                
            
    sim_dict = ECMLP_function(level_sim_dic, level_rele_dic, alpha, ln)        
    return sim_dict    
