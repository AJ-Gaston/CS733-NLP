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
        
        # Control Panel Frame
        control_frame = Frame(main_frame, bg=self.colors['card'], relief=RAISED, bd=2)
        control_frame.pack(fill=X, pady=(0, 20))
        
        # Restaurant Selection (inside the control frame)
        Label(control_frame, text="Select Restaurant:", font=self.fonts['heading'], 
              bg=self.colors['card']).grid(row=0, column=0, padx=10, pady=10, sticky=W)
        #Make a string variable for restaurants
        self.restaurant_var = StringVar()
        #Make a drop down menu of each restaurant (read only)
        self.restaurant_dropdown = ttk.Combobox(control_frame, textvariable=self.restaurant_var, 
                                                width=40, state="readonly")
        self.restaurant_dropdown.grid(row=0, column=1, padx=10, pady=10)
        #Bind the dropdown menu
        self.restaurant_dropdown.bind('<<ComboboxSelected>>', self.on_restaurant_selected)
        
        # Review Window Selection (inside the control frame)
        Label(control_frame, text="Review Window:", font=self.fonts['heading'], 
              bg=self.colors['card']).grid(row=0, column=2, padx=10, pady=10, sticky=W)
        #Make it an int instead of string
        self.window_var = IntVar()
        #Make another dropdown menu (this time for the review windows)
        self.window_dropdown = ttk.Combobox(control_frame, textvariable=self.window_var, 
                                            width=15, state="readonly")
        self.window_dropdown.grid(row=0, column=3, padx=10, pady=10)
        self.window_dropdown.bind('<<ComboboxSelected>>', self.on_window_selected)

        #Analyze button (use with Button())
        self.analyze_btn = Button(control_frame, text="Analyze", command=self.analyze_restaurant,
                                 font=self.fonts['heading'], bg=self.colors['neutral'], 
                                 fg='white', state=DISABLED)
        self.analyze_btn.grid(row=0, column=4, padx=10, pady=10)
        
        # Display Results Area (using anothe frame)
        results_frame = Frame(main_frame, bg=self.colors['background'])
        results_frame.pack(fill=BOTH, expand=True)
        
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