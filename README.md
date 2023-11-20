# Detecting the topic of comments on Reddit
• This project implemented for "Fundamentals of Computational Intelligence" course.

• The main job of this project is improving "Bag of Words" algorithm with inspiration from "Word2vec" algorithm to detect the topic of comments with higher accuracy.

## Part I: Clustering implementation
The main process of the algorithm is implemented in a function called clustering, and its output contains a list of cluster numbers as follows:

```py
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
```

The following fixed values are considered for testing:

```py
k = 2
c_target = 3
g = 2
```
Below are two examples of inputs and outputs:

### Test 1:
Input:
```py
input_array = [[100], [2], [3], [4], [1560], [1500], [1550], [8], [9], [102]]
```
OutPut:
```py
clusters: [0, 2, 2, 2, 1, 1, 1, 2, 2, 0]
```
### Test 2:
Input:
```py
input_array = [[0, 0], [1, 1], [100, 100], [101, 101], [105, 101], [1500, 1400], [1550, 1401], [1500, 1402], [1540, 1200], [1, 2]]
```
OutPut:
```py
clusters: [1, 1, 2, 2, 2, 0, 0, 0, 0, 1]
```

## Part II: Implementation of "Bag of Words" algorithm
To implement this algorithm, we have first cleaned the data. </br>
In this way, first we removed symbols such as dots, commas, etc. </br>
Then lowercase the comments and then remove the word stops. </br>
Finally, we replace the roots of the words with the compounds made from them, and thus the comments are cleared. </br>

In the next step, we get the **"tf_idf"** values for the train and test dataset comments that have been cleared. Then, to represent the data, we give these values to the SVM so that learning is done on the train data.
 

Finally, we obtain the **"accuracy"** value for the test data, which will be as follows:

```
simple Bag Of Words accuracy on the test data: 0.80
```

## Part III: improving "Bag of Words" algorithm using "word2vec" algorithm
One of the problems of "Bag of Words" algorithm is that it considers each word separately and does not consider the relationship between words that have the same meaning. </br>
In order to improve the Words Of Bag algorithm, we have designed a module that extracts a model from the pre-trained word2vec model ( <a href="https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g">  GoogleNews-vectors-negative300.bin </a> )  which is an array of vectors where each vector represents a synonym word. </br>
The word is from the collection of Dataset words. In this way, we will have a dictionary at the output, which stores one word and up to 5 synonyms in each vector.
</br>
</br>
In the next step, in the module related to the implementation of Bag of Words, after we have cleared the dataset project according to the "Part II"; We use the dictionary created in the previous step and for each word that is in the comments, we replace its synonym that is known as the head of that word collection in the dictionary.
</br>
Now we will have a set of new comments for which we will obtain tf_idf values according to the "Part II" of the project.
</br>
Then, to represent the data, we give these values to the **SVM** so that learning is done on the train data. And finally, we get the accuracy value for the test data, which will be as follows:

```
improved Bag Of Words accuracy on the test data: 0.83
```

## Result 

It is clear that we have about 3% improvement in accuracy compared to the previous state.
</br>
This method makes the improved Words Of Bag algorithm consider the relationship of synonymous words in a way, and in this way we will have normalized tf_idf values and a more accurate representation of train and test data. </br>
Also, accordingly, the accuracy also increases for two datasets, test and train.