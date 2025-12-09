#import the libraries
import pandas as pd
import re
import torch
import transformers
from transformers import AutoTokenizer

def preprocess(text: str):
    """
    Tokenizes the reviews text
    Args:
        text (str): reviews for each restaurant

    Returns:
        : _description_
    """
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    return tokenizer(text,truncation=True)

def main():
    reviews = pd.read_csv('./Dataset/Filtered_GoogleMaps_reviews.csv') #read in dataset with pandas
    #split each restaurant into their own dataframe
    grouped_dict = dict(tuple(reviews.groupby('title')))
    
    #Get 63 dataframes titles
    all_titles = list(grouped_dict.keys())
    # Able to have all 63 DataFrames accessible via grouped_dict[title]
    
    #Get the top 10 reviews for each restaurant by publishedAtDate
    top10_per_restaurant = []
    for title, df in grouped_dict.items():
        top10 = df.sort_values('publishedAtDate',ascending=False).head(10)
        top10_per_restaurant.append(top10)
        
    #preprocess text with tokenizer
    
    
    return


if __name__ == '__main__':
    main()