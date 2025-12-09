#import the libraries
import pandas as pd
import re
import torch
import transformers
from transformers import AutoTokenizer

def preprocess(text: str):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    

def main():
    reviews = pd.read_csv('./Dataset/Filtered_GoogleMaps_reviews.csv') #read in dataset with pandas
    #split each restaurant into their own dataframe
    grouped_dict = dict(tuple(reviews.groupby('title')))
    
    #Get 63 dataframes
    all_titles = list(grouped_dict.keys())
    
        
        
    
    
    return


if __name__ == '__main__':
    main()