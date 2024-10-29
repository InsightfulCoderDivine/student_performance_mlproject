import sys
from src.logger import logging

def error_msg_detail(error, error_detail: sys):
    """Custome error message."""
    _,_,exc_tb = error_detail.exc_info()
    
    # Get file name from exc_tb
    file_name = exc_tb.tb_frame.f_code.co_filename
    
    # Print error message
    error_message = "Error occured in python script name [{0}], line number [{1}], error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message
    
class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        # Inherit the __init__ of the Exception class
        super().__init__(error_message)
        
        # Initialize error message
        self.error_message = error_msg_detail(error=error_message, error_detail=error_detail)
        
    def __str__(self):
        return self.error_message
    
# if __name__ == "__main__":
#     try:
#         a = 1/0
#     except Exception as e:
#         logging.info("Divide by zero errorr.")
#         raise CustomException(e, sys)  