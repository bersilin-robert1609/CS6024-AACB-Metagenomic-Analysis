import os

def exists(folder, file):
    """
    Check if a file exists in a folder
    """
    return os.path.exists(os.path.join(folder, file))