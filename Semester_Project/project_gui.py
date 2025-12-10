from tkinter import *
from tkinter import ttk, messagebox
import semester_project

class RestaurantSentimentGUI:
    """
    Class that creates the GUI for restaurant sentiment
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Restaurant Sentiment Analyzer")
        self.root.geometry("800x600")
        
        #Initialize the data structures
        self.df = None
        self.model = None
        self.restaurant_data = None
        self.results = None
        
        #Review window options
        self.review_windows = [1,3,5,7,10]
        
        #Configure the styles
        self.setup_styles()
        
        #Build the GUI
        self.setup_gui()
        
        #Load data in the background
        self.load_data()
        
    def setup_styles(self):
        """
        Configure the colors and fonts for the GUI
        """
    
    def setup_gui(self):
        """
        Build the GUI interface
        """
        
    def load_data(self):
        """
        Load data and train the model in the background
        """

def main():
    root = Tk()
    app = RestaurantSentimentGUI(root)
    root.mainloop()
    

if __name__ == "__main__":
    main()