# Assignment 3 for ODU CS 733 - Natural Language Processing

## Author: Alexji Gaston

The first cell imports the dataset as a csv file and then converts it into a list of strings. I decided to get rid of the second column, which contained the urls for each tedt talk. The reason was since we're focusing on the transcripts, it was easire for simplicity.

The second cell uses the regular expression(re) and nltk library for preprocessing. I used re to get rid of special characters and punctuations, while using nltk to tokenize the text and eliminate stop words since the library already had a ready made list to use.

The third cell is my sanity check to ensure that everything is working out the way I want. This cell is entirely optional.

The fourth cell is the word2vec class. The reason why it's a class instead of a function is because when I was trying to incorporate everything in the textbook, it seemed that there were a lot of components so I decided to make it a class instead with functions inside it. I used the numpy library in order to do matrix and vector operations. The cell also uses the collection library to count the number of words.

The fifth cell is where I create the co-occurence matrix for the PMI and PPMI. It also has the functions to make a PMI matrix and a PPMI matrix. I used the collections library to get the defaultdict, the numpy library to make a 2d matrix, and the math library to get the log of the joint probability * the marginal probability. The numpy library also helped when trying to make sure that the pmi and ppmi matrix are the correct types.

The sixth cell is where the term frequency inverse document frequency is computed. There three functions are used to create a tf-idf of the entire corpus: term_freq, inverse_doc_freq, and compute_document_frequency. These three functions are needed for the function tfidf, which uses these functions for variable to create a tf-idf matrix needed for cosine similarity later. I used the colelctions library to get the Counter function and the math library to get the log function.

The seventh cell is where I finally train my three models using the corpus. This cell takes the longest and it can only be run once. If it is ran multiple times, then the kernel crashes out due to the fact that it neededs to compute all three models one after the other.
 
The eight cell is where I created my word pairs which are needed for the cosine similiarity and the heat matrix.

The ninth cell is where I create three functions to compute the cosine similarity of all three models: word2vec, ppmi, and tfidf. I used the numpy library to create the dot product using numpy dot function and the linear algebra norm function to get the vector norms. This is necessary because cosine similarity is the dot product of two vectors / (the norm of vector 1 * norm of vector 2)

The tenth cell is where I compare all model's cosine similarities. I use the word pair created in the eight cell and the functions in the ninth cell to create a dictionary that holds the results of every similarity function.

The cells 11-16 are heatmaps for the three models using the seaborn library and the matplot library, specifically pyplot. This is necessary to create the heatmaps.

### References
https://seaborn.pydata.org/generated/seaborn.heatmap.html
