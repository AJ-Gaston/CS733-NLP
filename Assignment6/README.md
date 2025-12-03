# Assignment 5 for ODU CS 733 - Natural Language Processing

Author: Alexji Gaston

For the entire assignment I decided to use PyTorch since I'm more familiar with that library than the others.

## Task 1: Implementing a transformer encoder
For this task, I started with reading in the csv files one by one using Pandas library. Then, I used Pandas' concatenate function to make one Pandas dataframe from the training.csv and the test.csv.

I applied the data preprocessing told in the assignment. 

## Task 2: Setting up a transformer decoder
### References
#### PyTorch library documentation
[PyTorch Transformer](https://docs.pytorch.org/docs/stable/generated/torch.nn.Transformer.html)
[PyTorch Embedding](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html)

[Pytorch Module](https://github.com/FrancescoSaverioZuppichini/Pytorch-how-and-when-to-use-Module-Sequential-ModuleList-and-ModuleDict)
https://www.youtube.com/watch?v=U0s0f995w14
https://www.youtube.com/watch?v=C9QSpl5nmrY
https://www.geeksforgeeks.org/nlp/how-do-self-attention-masks-work/
