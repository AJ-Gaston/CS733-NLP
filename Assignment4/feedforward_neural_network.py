class feedforward_nn:
    def __init__(self, hidden_size = 256, dropout = 0.2, learning_rate = 0.001, epoch=30):
        #Basic variabels that are initialized but can be altered if need be
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epoch
        
        #Training variables