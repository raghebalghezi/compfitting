#import ConfigParser
import numpy
import sys
import time
import random 
import math
import os
from copy import deepcopy
import json
from numpy.linalg import norm
from numpy import dot
from scipy.stats import spearmanr



def print_word_vectors(word_vectors, write_path):
	"""
	This function prints the collection of word vectors to file, in a plain textual format. 
	"""
	print ("Saving the counter-fitted word vectors to", write_path, "\n")
	with open(write_path, "wb") as f_write:
		for key in word_vectors:
			print >>f_write, key, " ".join(map(str, numpy.round(word_vectors[key], decimals=6))) 


def normalise_word_vectors(word_vectors, norm=1.0):
	"""
	This method normalises the collection of word vectors provided in the word_vectors dictionary.
	"""
	for word in word_vectors:
		word_vectors[word] /= math.sqrt((word_vectors[word]**2).sum() + 1e-6)
		word_vectors[word] = word_vectors[word] * norm
	return word_vectors

def extract(vector):
    #converts Glove Vector format into dictionary of np.float32 
    word_d = dict()
    with open(vector,encoding='utf-8') as f:
        for line in f:
            word_d[line.split()[0]] = numpy.float32(line.split()[1:])
    return word_d

def load_constraints(constraints_filepath, vocabulary):
	"""
	This methods reads a collection of constraints from the specified file, and returns a set with
	all constraints for which both of their constituent words are in the specified vocabulary.
	"""
	#constraints_filepath.strip()
	constraints = set()
	with open(constraints_filepath, "r+") as f:
		for line in f:
			word_pair = line.split()
			if word_pair[0] in vocabulary and word_pair[1] in vocabulary and word_pair[0] != word_pair[1]:
				constraints |= {(word_pair[0], word_pair[1])}
				constraints |= {(word_pair[1], word_pair[0])}

	print (constraints_filepath, "yielded", len(constraints), "constraints.")

	return constraints



def distance(v1, v2, normalised_vectors=True):
	"""
	Returns the cosine distance between two vectors. 
	If the vectors are normalised, there is no need for the denominator, which is always one. 
	"""
	if normalised_vectors:
		return 1 - dot(v1, v2)
	else:
		return 1 - dot(v1, v2) / ( norm(v1) * norm(v2) )


def compute_vsp_pairs(word_vectors, vocabulary, rho=0.2):
	"""
	This method returns a dictionary with all word pairs which are closer together than rho.
	Each pair maps to the original distance in the vector space. 

	In order to manage memory, this method computes dot-products of different subsets of word 
	vectors and then reconstructs the indices of the word vectors that are deemed to be similar.
	"""
	print ("Pre-computing word pairs relevant for Vector Space Preservation (VSP). Rho =", rho)
	
	vsp_pairs = {}

	threshold = 1 - rho 
	vocabulary = list(vocabulary)
	num_words = len(vocabulary)

	step_size = 1000 # Number of word vectors to consider at each iteration. 
	vector_size = 300#random.choice(word_vectors.values()).shape[0]

	# ranges of word vector indices to consider:
	list_of_ranges = []

	left_range_limit = 0
	while left_range_limit < num_words:
		curr_range = (left_range_limit, min(num_words, left_range_limit + step_size))
		list_of_ranges.append(curr_range)
		left_range_limit += step_size

	range_count = len(list_of_ranges)

	# now compute similarities between words in each word range:
	for left_range in range(range_count):
		for right_range in range(left_range, range_count):

			# offsets of the current word ranges:
			left_translation = list_of_ranges[left_range][0]
			right_translation = list_of_ranges[right_range][0]

			# copy the word vectors of the current word ranges:
			vectors_left = numpy.zeros((step_size, vector_size), dtype="float32")
			vectors_right = numpy.zeros((step_size, vector_size), dtype="float32")

			# two iterations as the two ranges need not be same length (implicit zero-padding):
			full_left_range = range(list_of_ranges[left_range][0], list_of_ranges[left_range][1])		
			full_right_range = range(list_of_ranges[right_range][0], list_of_ranges[right_range][1])
			
			for iter_idx in full_left_range:
				vectors_left[iter_idx - left_translation, :] = word_vectors[vocabulary[iter_idx]]

			for iter_idx in full_right_range:
				vectors_right[iter_idx - right_translation, :] = word_vectors[vocabulary[iter_idx]]

			# now compute the correlations between the two sets of word vectors: 
			dot_product = vectors_left.dot(vectors_right.T)

			# find the indices of those word pairs whose dot product is above the threshold:
			indices = numpy.where(dot_product >= threshold)

			num_pairs = indices[0].shape[0]
			left_indices = indices[0]
			right_indices = indices[1]
			
			for iter_idx in range(0, num_pairs):
				
				left_word = vocabulary[left_translation + left_indices[iter_idx]]
				right_word = vocabulary[right_translation + right_indices[iter_idx]]

				if left_word != right_word:
					# reconstruct the cosine distance and add word pair (both permutations):
					score = 1 - dot_product[left_indices[iter_idx], right_indices[iter_idx]]
					vsp_pairs[(left_word, right_word)] = score
					vsp_pairs[(right_word, left_word)] = score
		
	# print "There are", len(vsp_pairs), "VSP relations to enforce for rho =", rho, "\n"
	return vsp_pairs


def vector_partial_gradient(u, v, normalised_vectors=True):
	"""
	This function returns the gradient of cosine distance: \frac{ \partial dist(u,v)}{ \partial u}
	If they are both of norm 1 (we do full batch and we renormalise at every step), we can save some time.
	"""

	if normalised_vectors:
		gradient = u * dot(u,v)  - v 
	else:		
		norm_u = norm(u)
		norm_v = norm(v)
		nominator = u * dot(u,v) - v * numpy.power(norm_u, 2)
		denominator = norm_v * numpy.power(norm_u, 3)
		gradient = nominator / denominator

	return gradient


def one_step_SGD(word_vectors, synonym_pairs, antonym_pairs, vsp_pairs):
	"""
	This method performs a step of SGD to optimise the counterfitting cost function.
	"""
	new_word_vectors = deepcopy(word_vectors)

	gradient_updates = {}
	update_count = {}
	oa_updates = {}
	vsp_updates = {}

	# AR term:
	for (word_i, word_j) in antonym_pairs:

		current_distance = distance(new_word_vectors[word_i], new_word_vectors[word_j])

		if current_distance < 1.0: #delta
	
			gradient = vector_partial_gradient( new_word_vectors[word_i], new_word_vectors[word_j])
			gradient = gradient * 0.1 

			if word_i in gradient_updates:
				gradient_updates[word_i] += gradient
				update_count[word_i] += 1
			else:
				gradient_updates[word_i] = gradient
				update_count[word_i] = 1

	# SA term:
	for (word_i, word_j) in synonym_pairs:

		current_distance = distance(new_word_vectors[word_i], new_word_vectors[word_j])

		if current_distance > 0.0: #current_experiment.gamma: 
		
			gradient = vector_partial_gradient(new_word_vectors[word_j], new_word_vectors[word_i])
			gradient = gradient * 0.1 #current_experiment.hyper_k2 

			if word_j in gradient_updates:
				gradient_updates[word_j] -= gradient
				update_count[word_j] += 1
			else:
				gradient_updates[word_j] = -gradient
				update_count[word_j] = 1
	
	# VSP term:			
	for (word_i, word_j) in vsp_pairs:

		original_distance = vsp_pairs[(word_i, word_j)]
		new_distance = distance(new_word_vectors[word_i], new_word_vectors[word_j])
		
		if original_distance <= new_distance: 

			gradient = vector_partial_gradient(new_word_vectors[word_i], new_word_vectors[word_j]) 
			gradient = gradient * 0.1 #current_experiment.hyper_k3 

			if word_i in gradient_updates:
				gradient_updates[word_i] -= gradient
				update_count[word_i] += 1
			else:
				gradient_updates[word_i] = -gradient
				update_count[word_i] = 1

	for word in gradient_updates:
		# we've found that scaling the update term for each word helps with convergence speed. 
		update_term = gradient_updates[word] / (update_count[word]) 
		new_word_vectors[word] += update_term 
		
	return normalise_word_vectors(new_word_vectors)


def counter_fit():
	"""
	This method repeatedly applies SGD steps to counter-fit word vectors to linguistic constraints. 
	"""
	word_vectors = normalise_word_vectors(extract("glove.6B/glove.6B.300d.txt"))
	vocabulary = word_vectors.keys()
	antonyms = load_constraints("linguistic_constraints/ppdb_antonyms.txt", vocabulary)
	synonyms = load_constraints("linguistic_constraints/ppdb_synonyms.txt", vocabulary)
	
	current_iteration = 0
	vsp_pairs = {}

	if 0.1 > 0.0: # if we need to compute the VSP terms.
 		vsp_pairs = compute_vsp_pairs(word_vectors, vocabulary, rho=0.2)
	
	# Post-processing: remove synonym pairs which are deemed to be both synonyms and antonyms:
	for antonym_pair in antonyms:
		if antonym_pair in synonyms:
			synonyms.remove(antonym_pair)
		if antonym_pair in vsp_pairs:
			del vsp_pairs[antonym_pair]

	max_iter = 20
	print ("\nAntonym pairs:", len(antonyms), "Synonym pairs:", len(synonyms), "VSP pairs:", len(vsp_pairs))
	print ("Running the optimisation procedure for", max_iter, "SGD steps...")

	while current_iteration < max_iter:
		current_iteration += 1
		word_vectors = one_step_SGD(word_vectors, synonyms, antonyms, vsp_pairs)

	return word_vectors


