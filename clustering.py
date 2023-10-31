# imports
import math
import sys
import numpy as np
import copy

MAX_NUMBER = sys.maxsize

# inputs
# input_array = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
# input_array = [[100], [2], [3], [4], [1560], [1500], [1550], [8], [9], [102]]
input_array = [[0, 0], [1, 1], [100, 100], [101, 101], [105, 101], [1500, 1400], [1550, 1401], [1500, 1402], [1540, 1200], [1, 2]]


# vars
D_original_matrix = []
R_k = []


# functions


def dist(i, j):
    result = math.dist(i, j)
    return result


def d_original(input_array):
    for i in input_array:
        col = []
        for j in input_array:
            col.append(dist(i, j))
        D_original_matrix.append(col)


def get_R_k(k, N):
    d_original_copy = copy.deepcopy(D_original_matrix)
    for i in range(N):
        r_k_i = []
        d_original_copy[i][i] = MAX_NUMBER
        for j in range(0, k):
            np_array = np.array(d_original_copy[i])
            min_index = np.argmin(np_array)
            r_k_i.append(min_index)
            d_original_copy[i][min_index] = MAX_NUMBER
        R_k.append(r_k_i)


def initial_label(N):
    L_array = []
    for i in range(N):
        L_array.append(i)
    return L_array


def initial_k(n):
    k_n = []
    for i in range(n):
        k_n.append(i)
    return k_n


def update_s_and_k(s_n, k_n, chosen_element_index):
    k_n.remove(chosen_element_index)
    s_n.append(chosen_element_index)
    return s_n, k_n


def goal_function_1(D_matrix, m):
    avg_array = []
    for i in range(m):
        sum_i = 0
        for j in range(m):
            sum_i += D_matrix[i][j]
        avg_i = sum_i/m
        avg_array.append(avg_i)
    avg_np_array = np.array(avg_array)
    min_index = np.argmin(avg_np_array)
    return min_index


def get_minvalue_index(inputlist):
    min_value = min(inputlist)
    min_index = inputlist.index(min_value)
    return min_index


def get_maxvalue_index(inputlist):
    max_value = max(inputlist)
    max_index = inputlist.index(max_value)
    return max_index


def goal_function_2(D_matrix, s_n, k_n):
    res2 = []
    res2_index = []
    for k_item in k_n:
        res1 = []
        res1_index = []
        for s_item in s_n:
            res1.append(D_matrix[k_item][s_item])
            res1_index.append(k_item)
        res2.append(min(res1))
        res2_index.append(res1_index[get_minvalue_index(res1)])
    i_n = res2_index[get_maxvalue_index(res2)]
    return i_n


def get_D_current(c_previous, k):
    D_current_matrix = []
    coefficient = 1/math.pow(k+1, 2)
    for i in range(c_previous):
        row_i = []
        for j in range(c_previous):
            sum_i_j = 0
            for a in R_k[i]:
                for b in R_k[j]:
                    sum_i_j += D_original_matrix[a][b]
            for b in R_k[j]:
                sum_i_j += D_original_matrix[i][b]
            for a in R_k[i]:
                sum_i_j += D_original_matrix[a][j]
            d_i_j = coefficient * sum_i_j
            row_i.append(d_i_j)
        D_current_matrix.append(row_i)
    return D_current_matrix


def update_labels(L_array, s_n, k_n, D_matrix):
    for i in k_n:
        row_i = []
        row_i_index = []
        for j in s_n:
            row_i.append(D_matrix[i][j])
            row_i_index.append(j)
        min_index = row_i_index[get_minvalue_index(row_i)]
        for m in range(len(L_array)):
            if L_array[m] == i:
                L_array[m] = min_index
    for i in range(len(L_array)):
        L_array[i] = s_n.index(L_array[i])
    print(L_array)
    return L_array


def p_i(L_array, label):
    cluster_elements = []
    temp_p_i = []
    # cluster elements
    for i in range(0, len(L_array)):
        if L_array[i] == label:
            temp_p_i.append(i)
            cluster_elements.append(i)

    # neighbors
    for cluster_element in cluster_elements:
        for r_k in R_k[cluster_element]:
            temp_p_i.append(r_k)

    # prunning
    P_i = np.unique(temp_p_i)
    return P_i


def update_D_current(c_current, L_array, s_n):
    updated_D_current_matrix = []
    for i in range(c_current):
        P_i = p_i(L_array, i)
        col = []
        for j in range(c_current):
            P_j = p_i(L_array, j)
            sum_of_D = 0
            for a in P_i:
                for b in P_j:
                    sum_of_D += D_original_matrix[a][b]
            col.append((1/(len(P_i) * len(P_j)))*sum_of_D)
        updated_D_current_matrix.append(col)
    return updated_D_current_matrix


# main functions:


def choose_keys_process(D_matrix, c, s_n, k_n):
    I1 = goal_function_1(D_matrix, len(D_matrix))
    s_n, k_n = update_s_and_k(s_n, k_n, I1)
    n = len(s_n)
    while n != c:
        I_n_plus_1 = goal_function_2(D_matrix, s_n, k_n)
        s_n, k_n = update_s_and_k(s_n, k_n, I_n_plus_1)
        n = len(s_n)
    return s_n


def clustering(inputs, c_target):
    N = len(inputs)
    d_original(inputs)
    k = 2
    get_R_k(k, N)
    g = 2
    L_array = initial_label(N)
    k_n = initial_k(N)
    s_n = []
    c_previous = N
    c_current = math.floor(N / g)
    D_current_matrix = get_D_current(c_previous, k)
    iteration = 1
    while c_current > c_target:
        print("i = ", iteration)
        s_n = choose_keys_process(D_current_matrix, c_current, s_n, k_n)
        update_labels(L_array, s_n, k_n, D_current_matrix)
        D_current_matrix = update_D_current(c_current, L_array, s_n)
        iteration += 1
        k_n = initial_k(c_current)
        s_n = []
        c_previous = c_current
        c_current = math.floor(c_previous / g)
    s_final = choose_keys_process(D_current_matrix, c_target, s_n, k_n)
    update_labels(L_array, s_final, k_n, D_current_matrix)
    print("clusters : ", L_array)


# test

clustering(input_array, 3)
