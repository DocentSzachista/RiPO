import cv2
from .  import model_creation 
def video_handler( recording_name) -> None:
    """
        Handles reading video frame by frame by using openCV lib
        
        -------
        Params: 
        recording_name:
        -  str : if you want to read audio file provide filename as a parameter
        -  int : if you want to use your camera pass 0 as parameter

    """
    our_model = model_creation.RecognizerModel(load_model=True)
    capture = cv2.VideoCapture(recording_name) # initialize capture by creating VideoCapture Object
    if ( capture.isOpened() == False ): # check if we have managed to open a camera
        print("Error, couldn't open recording, pls check your file.")
    while ( capture.isOpened() ) :
        ret, frame = capture.read() # read frame
        
        if ( ret == True ):
            # TODO: here apply changes on frame     
            # transfer to grayscale 
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            make_image_changes(gray, our_model)
            cv2.imshow('Frame', frame) # show frame ov the image
            if cv2.waitKey(25) & 0xFF == ord('q'): # if we want to close frame window
	            break
        else:
            break
    capture.release() # release used resources 
    cv2.destroyAllWindows()

def make_image_changes(image: list, model) -> any:
    """
        Function that will apply changes on image 
        TODO: write function that will perform that changes
        
        -------
        Params:
        - image list : our list in shape of numpy array
        
        -------
        Returns numpy array with marked changes (At least I hope so)
    """
    model.predict(image)
    pass