class DataException(Exception):
    """Base class for exceptions related to data processing."""
    pass

class FileNotFoundError(DataException):
    """Exception raised when a required file is not found."""
    def __init__(self, filepath, message="File not found"):
        self.filepath = filepath
        self.message = f"{message}: {filepath}"
        super().__init__(self.message)

class InvalidDataFormatError(DataException):
    """Exception raised when the format of the data is invalid."""
    def __init__(self, message="Invalid data format"):
        self.message = message
        super().__init__(self.message)