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
        self.colors = {
            'positive': '#4CAF50',
            'negative': '#F44336',
            'neutral': '#2196F3',
            'background': '#f0f0f0',
            'card': '#ffffff',
            'text': '#333333'
        }
        
        self.fonts = {
            'title': ('Arial', 16, 'bold'),
            'heading': ('Arial', 12, 'bold'),
            'normal': ('Arial', 10),
            'metric': ('Arial', 14, 'bold')
        }
        
    def setup_gui(self):
        """
        Build the GUI interface
        """
        # Main container with padding
        main_frame = Frame(self.root, bg=self.colors['background'], padx=20, pady=20)
        main_frame.pack(fill=BOTH, expand=True)
        
        # Title of the GUI
        title_label = Label(main_frame, text="Restaurant Sentiment Analyzer", 
                          font=self.fonts['title'], bg=self.colors['background'])
        title_label.pack(pady=(0, 20))
        
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