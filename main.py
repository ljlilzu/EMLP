
import lp
import os
t = 20             # training times
p = 0.1            # proportion of the observed links as the testing set
suf = str(p * 10)


def layerNum(file):                                       #   Calculating the number of layers in a multiplex network.                                               
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


path = './DataSet/'

networks = [                                                    # All the multiplex networks used in the experiment.
            'Vickers/Vicker',  #0
            'CKM/CKM',  #1
            'Lazega/Lazega',  #2
            'Aarhus/Aarhus',  #3
            'Kapferer/Kapferer',  #4
            'Krackhardt/Krackhardt',  #5
            'Celegans/celegans',  #6
            'Padgett/Padgett', #7
            'TF/TF' #8            
           ]

results = [
             './results/Vicker_',    './results/CKM_',         './results/lazega_',    './results/Aarhus_',
             './results/Kapferer_', './results/Krackhardt_',  './results/Celegans_' , './results/Padgett_',  './results/TF_'   
          ]

net_ids = [0]                                      # Selecting the multiplex networks used in the experiment.

graph_file_list = []
result_file_list = []

for i in net_ids:
    graph_file_list.append(path + networks[i])
    result_file_list.append(results[i] + suf)

sim_methods = [                                                    # The basic similarity indexes.
    'CN',  # 0
    'PA',  # 1
    'RA',  # 2
    'Jaccard',  # 3
    'AA',  # 4
    'LP',  #5
    'RALP', #6
    'ECM'  # 7
]


sim_method = sim_methods[7] 

for i in range(len(graph_file_list)):
    graph_file = graph_file_list[i]
    result_file = result_file_list[i]     
    print(graph_file)                                           
    out_file = open(result_file + 'ds', 'w')                     # Opening the result file.
    out_file.write( 'Layer\tAUC\tRanking_Score\tTime (us)\tPrecision (10)\n')                                        
    for alpha in range(1,layerNum(graph_file)+1):
        out_file.write( str(alpha) + '\t' )
        rele = 'CSL'                                             # Setting the layer similarity measure        
        lp.LP(graph_file, out_file, sim_method, t, p, alpha, rele)
        out_file.flush()                        
        out_file.write('\n')           
    out_file.close()
            
