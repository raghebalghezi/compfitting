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

print("Loading data...")
with open('./linguistic_constraints/glove.6B.300d.txt','r',encoding="utf-8") as f:
    for line in f:
    	v = np.float32(line.split()[1:])
    	word_vectors[line.split()[0].strip()] = v/np.linalg.norm(v)

with open("./linguistic_constraints/compositional.txt",'r',encoding="utf-8") as file:
    for line in file:
        compositional_dict[line.split()[0].strip()]=line.split()[1].strip().split(",")
        
with open("./linguistic_constraints/wordnet-synonyms+.txt",'r',encoding="utf-8") as file:
    for line in file:
        syn_dict[line.strip().split()[0]] = line.strip().split()[1:]

with open("./linguistic_constraints/wordnet_antonyms.txt",'r',encoding="utf-8") as file:
    for line in file:
        ant_dict[line.strip().split()[0]] = line.strip().split()[1:]

def compoFitting(X, syn_dict,ant_dict,compositional_dict, n_iter=10):
    '''
    Enhances Word vector (GloVe) using synonyms, antonyms and etymological components.
    ==================================================================================
    NOTE: This represents the online update (partial derivative of the objective function w.r.t q_i, and after equating to zero, 
    we get the following update:
    q_i = \frac{\sum_{j : (i,j) \in SYN} \beta_{ij} q_j + \alpha_i
  \hat{q_i} +\sum_{r : (i,r) \in ANT} \gamma_{ir} q_r + \sum_{c : (i,c) \in COMP} \delta_{ic} \sum_{1:(c)}q_c } 
{\sum_{j : (i,j) \in SYN} \beta_{ij} + \sum_{r : (i,r) \in ANT} \gamma_{ir} + \sum_{c : (i,c) \in COMP} \delta_{ic} + \alpha_i}

    X: Distribuational Vector
    syn_dict: dictionary of key(word):values(list of synonyms) from WordNet
    ant_dict: dictionary of key(word):values(list of antonyms) from WordNet
    compositional_dict: dictionary of key(word):values(list of components) from Wiktionary.org
    alpha: distributional preservation weight. Ideally, it should be 1.
    beta: synonym vector weight. Should be 1/len(neighbors) ?
    gamma: antonym vector weight. Should be zero as we want to repel from antonyms ? 
    delta: compositional weight. set to 1, but the ideal value is unknown.
    n_iter: Number of iteration, defaulted to 10
    
    Returns : Dictionary of ComFitted Vectors
    '''
    
    Y = X.copy() #intialize w/ X vectors
    Y_prev = Y.copy()
    alpha = lambda x:1
    beta = lambda x:len(syn_dict[x]) if len(syn_dict[x])!=0 else 1
    gamma = lambda x:0.1
    delta = lambda x:1
    #normalize = lambda x:(x/np.linalg.norm(x)) + 1e-6

    for iteration in range(1, n_iter+1):
        
        for idx,word in enumerate(X):
            try:
                t1 = X[word] * alpha(word)
                t1 = t1 + np.array([Y_prev[j] * beta(word) for j in syn_dict[word]]).sum(axis=0)
                t2 = t1 + np.array([Y_prev[r] * gamma(word) for r in ant_dict[word]]).sum(axis=0)
                t3 = t2 + np.array([Y_prev[c] for c in compositional_dict[word]]).sum(axis=0)
                Y_prev[word] = t3/( beta(word) + alpha(word) + gamma(word) + delta(word))
                return Y_prev      
            except KeyError:
                pass


retrofitted_word_vector = compoFitting(word_vectors, syn_dict,ant_dict,compositional_dict, n_iter=10)
results_original = evaluate_on_all(word_vectors) #evaluate original vectors
results_retrofitted = evaluate_on_all(retrofitted_word_vector) # evaluate after retrofitting

print(results_original.to_dict())
print(results_retrofitted.to_dict())


#
