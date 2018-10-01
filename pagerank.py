#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 16:33:06 2018

@author: zifeiyu
"""

import numpy as np
from numpy import linalg as LA
import scipy as sci
from scipy import sparse
import sympy as sp
import random


class NotSquareMatrixError(Exception):
    pass
class Index_Error(Exception):
    pass

def random_hyperlink_matrix(m):
    """
    input integer `m`, return row-normalized random square matrix of size `m`
    (since the intelligent surfer may be more likely to jump to content-filled pages)
    """
    # create a square matrix of size `m`
    random_matrix = np.random.rand(m, m)
    # normalizing rows of `random_matrix`
    row_sums = random_matrix.sum(axis=1)
    new_matrix = random_matrix / row_sums[:, np.newaxis]
    return new_matrix

def random_lst(num):
    """
    input integer`num`
    return a list of length `num` with random integer element between 0 and `num`-1
    """
    end = num - 1
    lst = []
    for j in range(num):
        lst.append(random.randint(0,end))
    return lst

def same_index(I,J):
    """
    This function is used to exam if `I` and `J` indicate same element more than once
    input two lists `I` and `J`
    return a boolian value
    """
    I_index = list()
    for i in range(len(I)):
        I_index.append((I[i-1],J[i-1]))
    if len(I_index) == len(set(I_index)):
        return True
    else:
        return False
           


def hyperlink_matrix(I,J):
    """
    input lists `I` and `J` to specify the nonzero element
    (the xth elements i in `I`and j in `J`means there is a hyperlink from i to j)
    return the row normalized hyperlink matrix
    (all outlinks from a page are given equal weight in terms of the random surfer's hyperlinking probabilities)
    """
    if len(I) != len(J):
        raise Index_Error("The nonzero elements' indices are not one-to-one correspond")
    else:
        l = len(I)
        # elements equal to 1 if there is a hyperlink
        V = [1] * l
        B = sparse.coo_matrix((V,(I,J)),shape=(l,l)).todense()
        print(B)
        """
        # remove zero rows 
        B = np.delete(B,np.where(~B.any(axis=1))[0], axis=0)
        # remove zero columns by removing zero rows in the transpose of `B`
        transpose_B = np.delete(B.transpose(),np.where(~B.transpose().any(axis=1))[0], axis=0)
        B = transpose_B.transpose()
        print(B)
        print(np.shape(B))
        """
        row_sum = np.sum(B,axis=1).tolist()
        # remove the list symbol in every element of `row_sum`
        row_sum_lst = [x[0] for x in row_sum]
        # create a diagonal matrix with entries 1/cj(the links on jth page) if cj!=0
        # and 0 otherwise
        row_sum_inverse = [1/x if x!=0 else 0 for x in row_sum_lst]
        row_sum_inverse_array = np.array(row_sum_inverse)
        Diag_matrix = np.diag(row_sum_inverse_array)
        
    return np.matmul(Diag_matrix, B)

    
def matrix_size(x):
    """
    input a square matrix `x`
    return its size
    """
    n = np.shape(x)
    # raise an error if not a square matrix
    if n[0] != n[1]:
        raise NotSquareMatrixError("The matrix is not a square atrix")
    else:
        return n[0]
        


def zero_rows(x):
    """
    input a matrix `x`
    return the indices of its zero-rows
    """
    return list(np.where(~x.any(axis=1))[0])




def adjust_matrix(x):
    """
    add dangling node vector to original matrix
    The random surfer can hyperlink to any page at random, after entering a dangling node)
    """
    n = matrix_size(x)
    # entries in `dangling_node` is 1 if the page is a dangling node, and 0 otherwise
    dangling_node = np.zeros((n,1))
    for i in zero_rows(x):
        dangling_node[i][0] = 1
    # generate 1xn row vector with all entries as 1/n
    row_vector = 1/n * np.ones((1,n),dtype=int)
    # adjust the matrix to be H+a(1/n*e)
    S = x + np.matmul(dangling_node,row_vector)
    return S


def Google_matrix(x,alpha):
    """
    input a matrix `x`, and parameter `alpha` between 0 and 1
    return the corresponding Google Matrix
    """
    n = matrix_size(x)
    # teleportation matrix `E` is uniform (the surfer is equally likely to jump
    # to any page when teleporting
    E = 1/n * np.ones((n,n),dtype=int)
    # G = alpha*S+(1-alpha)*1/n*e*eT
    G = alpha * adjust_matrix(x) + (1-alpha) * E    
    return G

def PageRank_vector(x,alpha,e):
    """
    input a row normalized hyperlink matrix `x` and tolerance of convergence `e`
    retun the corresponding PageRank vector
    """
    n = matrix_size(x)
    PR_vector = 1/n * np.ones((n,1),dtype=int)
    # transpose the PageRank vector
    PR_transpose = PR_vector.transpose()
    # difference is originally between `PR_transpose` and zero-vector
    difference = PR_transpose
    counter = 0
    # stop the iteration until the power method has converged to a tolerance of 10^(-10)
    while LA.norm(difference) >= e:
        New_PR_transpose = np.matmul(PR_transpose,Google_matrix(x,alpha))
        difference = New_PR_transpose - PR_transpose
        PR_transpose = New_PR_transpose
        counter += 1
    print(counter)
    return PR_transpose


def PageRank_vector_alpha(x,k):
    """
    input the a row normalized hyperlink matrix `x` and the number of iterations `k`
    return the corresponding PageRank vector with symbolic parameter alpha
    """
    n = matrix_size(x)
    # create the google matrix with symbolic parameter alpha reserved
    E = 1/n * np.ones((n,n),dtype=int)
    # create symbolic parameter alpha as `a`
    a = sp.symbols('a')
    G = a * adjust_matrix(x) + (1-a) * E 
    PR_transpose = 1/n * np.ones((1,n),dtype=int)
    # calculate the matrix multiplication with object arrays for `k` times
    for i in range(k):
        PR_transpose = np.dot(np.array(PR_transpose,object),np.array(G,object))
    
    # expand the multiplication of polynomials
    return np.array([sp.expand(x) for x in PR_transpose[0]])


def PageRank_vector_derivative(x,k,i):
    """
    input a PageRank vector with symbolic parameter `a`(after `k`th iteration)
    return its `i`th order derivative of every component  
    """
    y = PageRank_vector_alpha(x,k)
    a = sp.symbols('a')    
    return [sp.diff(x,a,i) for x in y]
    
    



def rank(x,alpha,e):
    """
    input the PageRank vector, return the rank for each entries
    """
    array = PageRank_vector(x,alpha,e)
    order = array.argsort()
    ranks = order.argsort()
    return ranks


def main():
    """
    H = random_hyperlink_matrix(100)
    print(H)
    
    print(matrix_size(H))
    print(zero_rows(H))
    #print(row_vector(5))
    print(adjust_matrix(H))
    #print(Google_matrix(H,0.99))
    print(PageRank_vector(H,0.9999,10 ** -10))
    print(PageRank_vector(H,0.9,10 ** -10))
    print(PageRank_vector(H,0.85,10 ** -10))
    print(rank(H,0.85,10 ** -10))
    """
    """
    I = [2,4,1,3,4,1,3]
    J = [1,1,2,2,2,3,4]
    F = hyperlink_matrix(I,J)
    print(F)
    print(PageRank_vector(F,0.9999,10 ** -10))
    print(PageRank_vector(F,0.9,10 ** -10))
    print(PageRank_vector(F,0.85,10 ** -10))
   
    print(rank(F,0.9999,10 ** -10))
    print(PageRank_vector_alpha(F,1))
    print(PageRank_vector_derivative(F,1,1))
    """
    I = random_lst(1000)
    J = random_lst(1000)
    while same_index(I,J) is False:
       I = random_lst(1000)
       J = random_lst(1000)
    F = hyperlink_matrix(I,J)
    print(PageRank_vector(F,0.85,10 ** -10))
    
    
    """
    i = [1,2,2,3,4,4,4,5,6]
    j = [1,1,3,5,2,3,4,6,5]
    M = hyperlink_matrix(i,j)
    print(PageRank_vector_alpha(M,2))
    print(PageRank_vector_derivative(M,2,1))
    """
    
main()