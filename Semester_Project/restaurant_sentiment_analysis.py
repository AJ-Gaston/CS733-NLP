import torch
import torch.nn as nn
import transformers

class RestaurantSentimentAnalysisModel(nn.model):
    def train_model(model, trainloader):
        """
        Train the model based on the review and star associated with review
        """
        return
    def evaluate_model(y_true,y_pred):
        """
        This evaluates the model's accuracy, precision, recall, and f1-score
        Also used to compare the top 10 reviews to the restaurant's overall score
        """
        return