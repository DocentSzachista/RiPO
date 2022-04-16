from tkinter import Label, Tk, Button,  filedialog, messagebox
from . import video_handler as handler
from  os import getcwd
class App:
    """
        class contains base user interface to choose and launch a recording  

        NOTE: Need to consider if it is needed to add in the future more options.
        If no, then we can delete one 'choose file' button and left only Play and quit buttons 
    """
    _BUTTON_WIDTH = 20
    _BUTTON_PAD = 10
    def __init__(self, window_name: str, window_size : str ) -> None:
        
        self.file_path = ""

        # Window config section
        self.window = Tk()
        self.window.title(window_name)
        self.window.geometry(window_size) # setup size of app
        self.window.resizable(0, 0) # disable resize option 
        
        #components config section
        self.label = Label(self.window, text="Welcome to menu", font=("Courier", 12))
        self.play_button = Button(self.window, text="Play", width=self._BUTTON_WIDTH, command=self.play)
        self.options_button = Button(self.window, text="Choose file", width=self._BUTTON_WIDTH, command=self.file_browser)
        self.quit_button = Button(self.window, text="Quit", width=self._BUTTON_WIDTH, command=self.quit)
        
        # place window's elements in window 
        self.label.pack(pady=10)                
        self.play_button.pack(pady=self._BUTTON_PAD)
        self.options_button.pack(pady=self._BUTTON_PAD)
        self.quit_button.pack(pady=self._BUTTON_PAD)
            
    def start (self) -> None: 
        """
            Start destkop app
        """
        self.window.mainloop()

    def play(self) -> None:
        """
            Function to run a video, or launch camera 
        """
        if self.file_path == "":
            messagebox.showinfo("Info", "You must choose file first")
            self.file_browser()
            return
        handler.video_handler(self.file_path)

    # for now commented out, if more options than file choice appears then it will be added 
    # def open_options(self) -> None:
    #     new_window = Toplevel()
    #     new_window.title("Options")
    #     new_window.geometry("400x400")

    #     button = Button(new_window,text="Quit", command=new_window.destroy, width= self._BUTTON_WIDTH)


    #     button.pack()

    def file_browser(self) -> None:
        """
            launches FileDialog at project current working directory, 
            where he can choose recording. For now lets allow only mp4.  
        """
        self.file_path = filedialog.askopenfilename(initialdir = getcwd(), 
                                                title = "Select recording",
                                                filetypes = [("mp4 files", "*.mp4")])

    def quit(self) -> None:
        """
            Quit from program
        """
        self.window.quit()

   
    
   



