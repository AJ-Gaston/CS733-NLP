# Assignment 5 for ODU CS 733 - Natural Language Processing

Author: Alexji Gaston

For the entire assignment I decided to use PyTorch since I'm more familiar with that library than the others.

## Task 1: Implementing a transformer encoder
Cells 1-12, except cell 5, are necessary to run the the encoder model.

Cells 1-4 and cell 6 are for the data preprocessing and preparing for PyTorch. Cell 5 is a snaity check to see if the columns were created correctly.
Cells 7-11 are where I create the Transformer Encoder, use PyTorch's dataloader to get my train loader and test loader, trian the model to see if it's learning, and evaluate the model (precision, accuracy, recall, f1-score).
Cell 12 is where I plot the model and create the model's confusion matrix.

## Task 2: Setting up a transformer decoder
For this part, I watched quite a few youtube videos to help me.
### References
#### PyTorch library documentation
[PyTorch Transformer](https://docs.pytorch.org/docs/stable/generated/torch.nn.Transformer.html)
[PyTorch Embedding](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
[PyTorch Modules](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)
[Pytorch Modules and when to use certain things](https://github.com/FrancescoSaverioZuppichini/Pytorch-how-and-when-to-use-Module-Sequential-ModuleList-and-ModuleDict)
[PyTorch Adam](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html)

https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
https://stackoverflow.com/questions/50307707/how-do-i-convert-a-pandas-dataframe-to-a-pytorch-tensor
https://www.geeksforgeeks.org/deep-learning/converting-a-pandas-dataframe-to-a-pytorch-tensor/
https://www.youtube.com/watch?v=U0s0f995w14
https://www.youtube.com/watch?v=C9QSpl5nmrY
https://www.geeksforgeeks.org/nlp/how-do-self-attention-masks-work/
