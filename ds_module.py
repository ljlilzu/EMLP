
import math


def intersect(B, C):

    if len(B) == 1 and len(C) == 1 and B == C:
        return B

    ss = set(B) & set(C)
    ll = list(ss)
    size = len(ll)
    if size == 0:
        return ''
    if size == 1:
        return ll[0]
   
    ll.sort()
    A = ''.join(ll)

    return A


def Dempster_combination(m1, m2):
   

   
    K = 0.0
    for key1, value1 in m1.items():
        if int(value1 * 10000) == 0:
            continue
        for key2, value2 in m2.items():
            if int(value2 * 10000) == 0:
                continue
            if intersect(key1, key2) == '':
               
                K += value1 * value2
            # end if
        # end for
    # end for
    # print(K)
    One_minus_K = 1 - K

   
    m = {}
    for key1, value1 in m1.items():
        if int(value1 * 1000000) == 0:
            continue
        for key2, value2 in m2.items():
            if int(value2 * 1000000) == 0:
                continue
            A = intersect(key1, key2)
            if A == '':
                continue
            if A in m.keys():
                m[A] += value1 * value2
            else:
                m[A] = value1 * value2
            # end if
        # end for
    # end for

    for key, value in m.items():
        m[key] = value / One_minus_K
    # end for

    return m
# end def

def Dempster_rule(m_list, n):
    
    if n == 1:
        return m_list[0]

    m = m_list[0]
    for i in range(1, n):
        m_i = m_list[i]
        m = Dempster_combination(m, m_i)
    # end for

    return m

def ECMLP_function(level_sim_dic, level_rele_dic, alpha, ln): 
    
    
    max_rele = max(level_rele_dic.values())  
    
    for key5, value5 in level_sim_dic.items():                                        # constructing the mass functions of node pairs.        
        max_value2 = max(value5.values()) 
        min_value2 = min(value5.values())
        D_value2 = max_value2 - min_value2
        discount = level_rele_dic[key5] / max_rele
        for key6,value6 in value5.items():            
            mass = {}
            mass['L'] = (value6 - min_value2) / (D_value2 + 1)
            mass['L'] *= discount
            mass['U'] = (max_value2 - value6) / (D_value2 + 1)
            mass['U'] *= discount
            mass['LU'] = 1 / ( D_value2 + 1 )
            mass['LU'] = 1 - discount + discount * mass['LU']
            value5[key6] = mass           
           
    dic11 = {}                                                                     
    for key7 in level_sim_dic[alpha].keys():
        mass_list = []
        for value8 in level_sim_dic.values():        
            dic22 = value8
            value9 = dic22[key7]
            mass_list.append(value9)
        dic11[key7] = mass_list 
    
    dic_fusion = {}                                                               
    sim_dic1 = {}
    for key10 in level_sim_dic[alpha].keys():    
        dic_fusion[key10] = Dempster_rule(dic11[key10], ln)                      # fusing mass function via the Dempster’s rule of combination, and the result is saved in dic_fusion.
        if 'L' not in dic_fusion[key10].keys():                                  #constructing the dictionary of similarity scores for node pairs.
            sim_dic1[key10] = 0.5 * dic_fusion[key10]['LU']
        else:
            sim_dic1[key10] = dic_fusion[key10]['L'] + 0.5 * dic_fusion[key10]['LU']
        
    return sim_dic1



