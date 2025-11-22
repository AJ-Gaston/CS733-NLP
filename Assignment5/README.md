# Assignment 5 for ODU CS 733 - Natural Language Processing

## Author: Alexji Gaston

For this project I used the glob and numpy library. Glob to read in the extracted_brown files and numpy for the math used in the HMM and CRF model.

For the first cell, I read in the files with the glob library and cretae a list of the files

For the second cell, I create a list of a list of tuples. The tuple contains a string, the word, and a set of strings, the part-of-speech tags. The reason why I made it into a set of strings is because some words have multiple tags and I need some way to hold both of those tags.

The thid cell is a sanity check, to see if the structure was properly created in cell 2.

The fourth cell is a hidden markov model class. I set it without parameters because everything is calculated during the training of the HMM. This class has a train model that creates the states(part-of-speech tags) and obervsations(words). Then it calculates the initial probability, transition probability, and the emission probability matrices. 

The fifth cell is where I create the hidden markov model and then train it with the words and tags in the files.

The 6th cell is where I created a few test sentences to see if the viterbi algorithm works properly. I made a few short sentences and then two long ones to se if the Hidden Markov Model can accurately predict the tags for the words.



### References