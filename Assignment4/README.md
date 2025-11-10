# Assignment 4 for ODU CS 733 - Natural Language Processing

## Author: Alexji Gaston

Cell 1 is where I import my python libraries for the rest of the jupyter notebook. 

Cell 2 is where I read in my datasets. I used the pandas library to read in the ham-spam csv file and the os library to read in all the files inside the wikitext-2 dataset folder.

There are a few cells that do not need to be run. All of these cells have a comment of SANITY CHECK at the top. These do not need to be run.

I made two different preprocessing functions for the two datasets. Def preprocessing is reused code from assignment 3, but altered because the wiki-text dataset is a bit different from the dataset in assignment 3. Here I made sure to actually convert the numbers followed by words into words, so 1000th would become one-thousandth and so on.
For preprocessing dataframe, I looked at teh csv file and didn't find numbers followed by words so I wasn't as rigorous as the preprocessing function above. I preprocessed like the assignment said and applied it only yo the text column of the dataframe since that's the only we need to preprocess for the word2vec and neural network.

Then, I focused on the first task of the assignment and applied word2vec to the ham_spam dataset. I looked at how to apply the pretrained word2vec model to the dataset with two of my references below.

I then fed the dataset into a simple neural network. I tried to make it into a class, but then I thought it might get too complciated so I made it into a couple of functions. That made it too hard to follow so I went back to making it a class.

Then I worked on the wiki-text dataset. This one was a bit harder to make since I didn't know how to really make a one-hot vector, and when I tried to the kernel kept dying on me.

I can now get it to work but it takes almost 2 hours to run.




References for this assignment:
https://radimrehurek.com/gensim/models/word2vec.html
https://www.geeksforgeeks.org/nlp/word2vec-with-gensim/
https://www.geeksforgeeks.org/deep-learning/batch-size-in-neural-network/