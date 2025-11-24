# Assignment 5 for ODU CS 733 - Natural Language Processing

## Author: Alexji Gaston

For this project I used the glob and numpy library. Glob to read in the extracted_brown files and numpy for the math used in the HMM and CRF model.

For the first cell, I read in the files with the glob library and cretae a list of the files

For the second cell, I create a list of a list of tuples. The tuple contains a string, the word, and a set of strings, the part-of-speech tags. The reason why I made it into a set of strings is because some words have multiple tags and I need some way to hold both of those tags.

The third cell is a sanity check, to see if the structure was properly created in cell 2.

The fourth cell is a hidden markov model class. I set it without parameters because everything is calculated during the training of the HMM. This class has a train model that creates the states(part-of-speech tags) and obervsations(words). Then it calculates the initial probability, transition probability, and the emission probability matrices. 

The fifth cell is where I create the hidden markov model and then train it with the words and tags in the files.

The 6th cell is where I created a few test sentences to see if the viterbi algorithm works properly. I made a few short sentences and then two long ones to se if the Hidden Markov Model can accurately predict the tags for the words.

The seventh cell is a feature extraction function for my linear chain conditional random fields.

The eight cell is where I split my dataset into a training and test set. 

The ninth cell is to check the length of the split sets.

The 10th cell is the implementation of a linear chain conditional random fields for part of speech tagging.

The 11th cell is where I train the model and evaluate its accuracy.

The 12th cell reads in the GMB dataset as a pandas dataframe. I read it in with no header so I could create my column heads necessary for concatenating the words in every sentence.

The 13th cell is is a getsentence class,it creates a list of sentences where element in the list is the concatenated sentences.

The 14th cell is where I create the list of sentences from the GMB dataset.

The 15th cell is where I imported the nltk's gazetteer. This is for the NER CRFs.

The 16th cell is my modified word feature function that looks at the context words and the checks to see if the word exists in the gazetter.

The 17th cell is where I have two functions. The sent2features function creates the features for each word, and the sent2label gets the label for each word in the sentence.

The 18th cell is where I split my sentences into a training set and a test set.

The 19th cell is my modified CRF for NER.

The 20th cell is where I run the model's training with my sentences and evaluate the model.


### References
#### For the Hidden Markov Model
https://en.wikipedia.org/wiki/Viterbi_algorithm
https://medium.com/data-science/hidden-markov-models-explained-with-a-real-life-example-and-python-code-2df2a7956d65
https://goelarna.medium.com/part-of-speech-tagging-and-hidden-markov-models-5d1b548ece00
https://www.youtube.com/watch?v=FqGzSM11iKY

#### For the linear-chain Conditional random field
https://www.youtube.com/watch?v=3w0EhxiebUA&t=408s
https://youtu.be/B1nl8fLgKMk?si=nV8SeCWVPCkQB3bL
https://youtu.be/7L0MKKfqe98?si=k4jyeJ-Mv_4Eiz0T
https://youtu.be/2KTeXhfsc-k?si=4a98zJp9SRGme69q
https://www.geeksforgeeks.org/nlp/conditional-random-fields-crfs-for-pos-tagging-in-nlp/
https://www.cs.mcgill.ca/~jcheung/teaching/fall-2017/comp550/lectures/lecture9.pdf
https://eli5.readthedocs.io/en/latest/_notebooks/debug-sklearn-crfsuite.html
