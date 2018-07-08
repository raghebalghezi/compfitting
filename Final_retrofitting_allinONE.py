#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 08:45:03 2018

@author: raghebal-ghezi
"""

from retrofit import *
from web.evaluate import evaluate_on_all
from counterfitting import *

original_word_Vector = extract('glove.6B.300d.txt') #dict of glove

#retrofitting
wordVecs = read_word_vecs('glove.6B.300d.txt') #Read all the word vectors and normalize them
lexicon = read_lexicon("linguistic_constraints/compositional.txt",wordVecs)
retrofitted_word_vector = retrofit(wordVecs, lexicon, 10)

#counterfitting
counterfitted_word_vector = counter_fit()

#retrofitting AND counterfitting
result_retro_counter = retrofit(counterfitted_word_vector, lexicon, 10)

#evaluation
results_original = evaluate_on_all(original_word_Vector) #evaluate original vectors
results_retrofitted = evaluate_on_all(retrofitted_word_vector) # evaluate after retrofitting
results_counterfitted = evaluate_on_all(counterfitted_word_vector) # evaluate after counterfitting
results_counterfitted_retrofitted = evaluate_on_all(counterfitted_word_vector) #evaluate after both
print(results_original)
print(results_retrofitted)
print(results_counterfitted)
print(results_counterfitted_retrofitted)

#
