#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 16:33:06 2018

@author: yifan yu
"""

import numpy as np
from numpy import linalg as LA
import scipy as sci
from scipy import sparse
import sympy as sp
import random
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import math

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
    (the xth elements i in `I`and j in `J` means there is a hyperlink from i to j)
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
        # delete the first row and first column(all zeros)of the hyperlink matrix
        B = np.delete(B,0,0)
        B = np.delete(B,0,1)
        # delete the last few zero rows and zero columns of the hyperlink matrix
        B = np.delete(B,np.s_[max(I+J):l],0)
        B = np.delete(B,np.s_[max(I+J):l],1)
        print(B)
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


def normalized_hyperlink(H):
    """
    input the original hyperlink matrix of wikipedia webpages
    return the row normalized form
    """
    row_sums = H.sum(axis=1)
    for i in range(len(row_sums)):
        if row_sums[i] != 0:
            H[i:] = H[i:]/row_sums[i]
    return H

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

def PageRank_vector_iteration(x,alpha,k):
    """
    input a row normalized hyperlink matrix `x` and number of itearation `k`
    retun the corresponding PageRank vector
    """
    n = matrix_size(x)
    PR_vector = 1/n * np.ones((n,1),dtype=int)
    # transpose the PageRank vector
    PR_transpose = PR_vector.transpose()
    for i in range(k):
        New_PR_transpose = np.matmul(PR_transpose,Google_matrix(x,alpha))
        PR_transpose = New_PR_transpose
    return PR_transpose



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




def power_pagerank(H,alpha,iteration):
    """
    input hyperlink matrix `H`, teleportation parameter `alpha` and iteration times
    return the error list of PageRank vector calculated by the power method.
    pi^(k+1)T = alpha pi^{(k)T} + (alpha pi^{(k)T}a +(1-alpha))1/n e^T
    """
    n = matrix_size(H)
    PR_vector = 1/n * np.ones((n,1),dtype=int)
    # transpose the PageRank vector
    PR_transpose = PR_vector.transpose()
    dangling_node = np.zeros((n,1))
    for i in zero_rows(H):
        dangling_node[i][0] = 1  
    v = 1/n * np.ones((1,n),dtype=int)
    error_lst = list()
    for i in range(iteration):
        part = alpha * np.matmul(PR_transpose,dangling_node)+(1-alpha)
        New_PR_transpose = alpha * np.matmul(PR_transpose,H)+ part*v
        difference = New_PR_transpose - PR_transpose
        error = LA.norm(difference)
        error_lst.append(error)
        PR_transpose = New_PR_transpose
    return error_lst

def Jacobi(A,b,iteration):
    """
    input matrix `A`, vector `b` and iteration times
    return the list of error of jacobi's method in each iteration
    """
    n = matrix_size(A)
    x = 1/n * np.ones((n,1),dtype=int)
    x_new = np.zeros((n,1))
    error_lst = list()
    for k in range(iteration):
        
        for i in range(n):
            sigma = 0
            for j in range(n):
                if j != i:
                    sigma = sigma + float(A[i][j]) * float(x[j][0])
            x_new[i] = (b[i][0] - sigma)/A[i][i]
        
        
        difference = x_new - x
        error = LA.norm(difference)
        error_lst.append(error)
        x = np.copy(x_new)
       
        
    return error_lst


def Jacobi_PageRank(H,alpha,iteration):
    """
    input hyperlink matrix `H`, teleportation parameter `alpha` and iteration times
    return the error list of PageRank vector calculated by Jacobi Mathod
    (I-alpha*S)pi^{T} = (1-alpha) * 1/n * e^T
    """
    n = matrix_size(H)
    I = np.identity(n)
    S = adjust_matrix(H)
    A = I - alpha * S.transpose()
    b = (1-alpha)/n * np.ones((n,1),dtype=int)
    return Jacobi(A,b,iteration)


def Gauss_Seidel(A,b,iteration):
    """
    input matrix `A`, vector `b` and iteration times
    return the list of error of Gauss-Seidel's method in each iteration
    """
    n = matrix_size(A)
    x = 1/n * np.ones((n,1),dtype=int)
    x_new = np.zeros((n,1))
    error_lst = list()
    for k in range(iteration):
        for i in range(n):
            sigma = 0
            for j in range(0,i):
                sigma = sigma + float(A[i][j]) * float(x_new[j][0])
            for j in range(i+1,n):
                sigma = sigma + float(A[i][j]) * float(x[j][0])
            x_new[i] = (b[i][0] - sigma)/A[i][i]
        
        
        difference = x_new - x
        error = LA.norm(difference)
        error_lst.append(error)
        x = np.copy(x_new)
       
    return error_lst




def Gauss_Seidel_PageRank(H,alpha,iteration):
    """
    input hyperlink matrix `H`, teleportation parameter `alpha` and iteration times
    return the error list of PageRank vector calculated by Gauss-Seidel Mathod
    (I-alpha*S)pi^{T} = (1-alpha) * 1/n * e^T
    """
    n = matrix_size(H)
    I = np.identity(n)
    S = adjust_matrix(H)
    A = I - alpha * S.transpose()
    b = (1-alpha)/n * np.ones((n,1),dtype=int)
    return Gauss_Seidel(A,b,iteration)


def SOR(A,b,omega,iteration):
    """
    input matrix `A`, vector `b`, relaxation parameter `omega` and iteration times
    return the list of error of Successive over-relaxation (SOR) method in each iteration
    """
    n = matrix_size(A)
    x = 1/n * np.ones((n,1),dtype=int)
    x_new = np.zeros((n,1))
    error_lst = list()
    for k in range(iteration):
        for i in range(n):
            sigma = 0
            for j in range(0,i):
                sigma = sigma + float(A[i][j]) * float(x_new[j][0])
            for j in range(i+1,n):
                sigma = sigma + float(A[i][j]) * float(x[j][0])
            x_new[i] = (1-omega)* x[i] + omega * (b[i][0] - sigma)/A[i][i]
        
        difference = x_new - x
        error = LA.norm(difference)
        error_lst.append(error)
        x = np.copy(x_new)
       
    return error_lst



def SOR_PageRank(H,alpha,omega,iteration):
    """
    input hyperlink matrix `H`, teleportation parameter `alpha`
    relaxation parameter `omega` and iteration times
    return the error list of PageRank vector calculated by SOR Mathod
    (I-alpha*S)pi^{T} = (1-alpha) * 1/n * e^T
    """
    n = matrix_size(H)
    I = np.identity(n)
    S = adjust_matrix(H)
    A = I - alpha * S.transpose()
    b = (1-alpha)/n * np.ones((n,1),dtype=int)
    return SOR(A,b,omega,iteration)

def SOR_2(A,b,omega,e):
    """
    input matrix `A`, vector `b`, 
    relaxation parameter `omega` and error `e`
    return iteration times for Successive over-relaxation (SOR) method
    converges within required error
    """
    n = matrix_size(A)
    x = 1/n * np.ones((n,1),dtype=int)
    x_new = np.zeros((n,1))
    error = 1
    count = 0
    while error >= e:
        for i in range(n):
            sigma = 0
            for j in range(0,i):
                sigma = sigma + float(A[i][j]) * float(x_new[j][0])
            for j in range(i+1,n):
                sigma = sigma + float(A[i][j]) * float(x[j][0])
            x_new[i] = (1-omega)* x[i] + omega * (b[i][0] - sigma)/A[i][i]
        
        difference = x_new - x
        error = LA.norm(difference)
        count += 1
        x = np.copy(x_new)
       
    return count

def SOR_2_PageRank(H,alpha,omega,e):
    """
    input hyperlink matrix `H`, teleportation parameter `alpha`
    relaxation parameter `omega` and error `e`
    return the iteration times of PageRank vector calculated by SOR Mathod
    (I-alpha*S)pi^{T} = (1-alpha) * 1/n * e^T
    """
    n = matrix_size(H)
    I = np.identity(n)
    S = adjust_matrix(H)
    A = I - alpha * S.transpose()
    b = (1-alpha)/n * np.ones((n,1),dtype=int)
    return SOR_2(A,b,omega,e)





def error_plot(H,alpha,iteration):
    """
    input Hyperlink matrix `H`,teleportation parameter `alpha` and iteration times
    return the plot of error of Power Method, Jacobi Method and Gauss-Seidel with respect to iteration times
    """
    error_lst_1 = power_pagerank(H,alpha,iteration)
    error_lst_2 = Gauss_Seidel_PageRank(H,alpha,iteration)
    error_lst_3 = Jacobi_PageRank(H,alpha,iteration)
    print(error_lst_1, error_lst_2,error_lst_3)
    plt.axis([0, iteration, 0, 10**(-17)])
    t = np.arange(0,iteration,1)
    plt.xlabel('Number of iterations')
    plt.ylabel('Error between consecutive iterations')
    plt.title('Comparison of iterative methods')
    power = plt.plot(t,error_lst_1,'r--',label= 'Power Method')
    Gauss = plt.plot(t,error_lst_2,'k',label = 'Jacobi Method')
    Jacobi = plt.plot(t,error_lst_3,'g^',label = 'Gauss-Seidel Method')
    plt.legend()
    plt.show()
    return


def GMRES_PageRank(H,alpha,tol):
    """
    input hyperlink matrix `H`, teleportation parameter `alpha` and iteration times
    return the error list of PageRank vector calculated by GMRES Mathod
    (I-alpha*S)pi^{T} = (1-alpha) * 1/n * e^T
    """
    n = matrix_size(H)
    I = np.identity(n)
    S = adjust_matrix(H)
    A = I - alpha * S.transpose()
    b = (1-alpha)/n * np.ones((n,1),dtype=int)
    x0 = 1/n * np.ones((n,1))
    return spla.gmres(A,b,x0,tol)    



def BICGSTAB_PageRank(H,alpha,tol):
    """
    input hyperlink matrix `H`, teleportation parameter `alpha` and iteration times
    return the error list of PageRank vector calculated by BICGSTAB Mathod
    (I-alpha*S)pi^{T} = (1-alpha) * 1/n * e^T
    """
    n = matrix_size(H)
    I = np.identity(n)
    S = adjust_matrix(H)
    A = I - alpha * S.transpose()
    b = (1-alpha)/n * np.ones((n,1),dtype=int)
    x0 = 1/n * np.ones((n,1))
    return spla.bicgstab(A,b,x0,tol)  



def main():
    H = mmread("/Users/zifeiyu/desktop/mat.mtx").toarray()
    normalized_H = normalized_hyperlink(H)
    #print(np.shape(normalized_H))
    adjust_H = adjust_matrix(normalized_H)
    G = Google_matrix(normalized_H,alpha = 0.85)
    plt.axis([1, 2700, 1, 2700])
    t = np.arange(1,2658,1)
    for i in range(20):
        pi = rank(normalized_H,alpha=(1+i)/20,e=10**(-10)).tolist()[0]
        convergence_times = plt.plot(t,pi,'r--',label= 'number of iterations')
    plt.show()
    
main()