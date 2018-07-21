#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 08:45:03 2018

@author: raghebal-ghezi
"""


from web.evaluate import evaluate_on_all
import numpy as np


word_vectors = dict() #key(word):float32(300 dim word vector)
compositional_dict = dict() # dictionary of words extracted from wikitionary
syn_dict = dict() #synonyms
ant_dict = dict() #antonyms

def norm(v):
    return v / np.sqrt((v**2 + 1e-6).sum())
def dis(u,v):
    return np.dot(u,v)

print("Loading data...")
with open('../glove.6B/glove.6B.300d.txt','r',encoding="utf-8") as f:
    for line in f:
    	v = np.float32(line.split()[1:])
    	word_vectors[line.split()[0].strip()] = norm(v)
with open("./linguistic_constraints/compositional.txt",'r',encoding="utf-8") as file:
    for line in file:
        compositional_dict[line.split()[0].strip()]=line.split()[1].strip().split(",")
        
with open("./linguistic_constraints/ppdb_synonyms.txt",'r',encoding="utf-8") as file:
    for line in file:
        syn_dict[line.strip().split()[0]] = line.strip().split()[1:]

with open("./linguistic_constraints/ppdb_antonyms.txt",'r',encoding="utf-8") as file:
    for line in file:
        ant_dict[line.strip().split()[0]] = line.strip().split()[1:]
        
     
def enhance_euclid(word_vectors,syn_dict,ant_dict,compositional_dict):
    '''
    Enhances Word vector (GloVe) using synonyms, antonyms and etymological components.
    ==================================================================================
    NOTE: This represents the online update (partial derivative of the objective function w.r.t q_i, and after equating to zero, 
    we get the following update:
    q_i = \frac{\sum_{j : (i,j) \in SYN} \beta_{ij} q_j + \alpha_i
  \hat{q_i} +\sum_{r : (i,r) \in ANT} \gamma_{ir} q_r + \sum_{c : (i,c) \in COMP} \delta_{ic} \sum_{1:(c)}q_c } 
{\sum_{j : (i,j) \in SYN} \beta_{ij} + \sum_{r : (i,r) \in ANT} \gamma_{ir} + \sum_{c : (i,c) \in COMP} \delta_{ic} + \alpha_i}

    word_vectors: Distribuational Vector
    syn_dict: dictionary of key(word):values(list of synonyms) from WordNet
    ant_dict: dictionary of key(word):values(list of antonyms) from WordNet
    compositional_dict: dictionary of key(word):values(list of components) from Wiktionary.org
    alpha: distributional preservation weight. Ideally, it should be 1.
    beta: synonym vector weight. Should be 1/len(neighbors) ?
    gamma: antonym vector weight. Should be 1/len(neighbors) ?
    delta: compositional weight. set to 1/len(components)
    n_iter: Number of iteration, defaulted to 10
    
    Returns : Dictionary of enhanced Vectors
    '''
    Y = word_vectors.copy()
    Y_prev = word_vectors.copy()
    
    for i in range(1,11):
        for w in word_vectors.keys():
            try:
                
                word_vectors[w] = word_vectors[w] * 2 #2 is beta
                for j in set(syn_dict[w]):
                    word_vectors[w] += word_vectors[j]
                word_vectors[w] = word_vectors[w] / 2 # weighted sum
                
                word_vectors[w] = word_vectors[w] * 2
                for r in set(ant_dict[w]):
                    word_vectors[w] -= word_vectors[j]
                word_vectors[w] = word_vectors[w] / 2
                
                word_vectors[w] = word_vectors[w] * 2
                q_c = np.zeros(300)
    
                for c in set(compositional_dict[w]):
                    q_c += word_vectors[c]
                q_c = q_c / len(word_vectors[c])
                word_vectors[w] += q_c
                word_vectors[w] = word_vectors[w] / 2

            except KeyError:
                pass
            #print("before",w, Y_prev[w])
        Y[w] -= word_vectors[w]
            #print("after",w, Y[w])
    return Y

Y = enhance_euclid(word_vectors,syn_dict,ant_dict,compositional_dict)
results_retrofitted = evaluate_on_all(Y)
print(results_retrofitted)

'''
{'AP': {0: 0.63681592039800994},
 'BLESS': {0: 0.82000000000000006},
 'Battig': {0: 0.41693748805199771},
 'ESSLI_1a': {0: 0.77272727272727271},
 'ESSLI_2b': {0: 0.82500000000000007},
 'ESSLI_2c': {0: 0.64444444444444449},
 'Google': {0: 0.10284486287351617},
 'MEN': {0: 0.73746469698055173},
 'MSR': {0: 0.086999999999999994},
 'MTurk': {0: 0.63318199788472018},
 'RG65': {0: 0.76952497886121318},
 'RW': {0: 0.36701861264054064},
 'SemEval2012_2': {0: 0.16995896694889304},
 'SimLex999': {0: 0.37050035710869067},
 'TR9856': {0: 0.099725736554407501},
 'WS353': {0: 0.54355827297629189},
 'WS353R': {0: 0.47754204101544628},
 'WS353S': {0: 0.66245573499038346}}
 '''
