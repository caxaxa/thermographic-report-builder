import os 
import shutil

def dxf_file_path(current_dir):
    dxf_dir = os.path.join(current_dir, "Inputs", "dxf")
    dxf_file_path, new_path = get_single_file_in_directory(current_dir, dxf_dir)

    if dxf_file_path:
        print(f"Found .dxf file at: {dxf_file_path}")
    else:
        print(f"No .dxf files found in {dxf_dir}")
    return new_path

def get_single_file_in_directory(current_dir, directory, extension='.dxf'):
    """
    Returns the first file in the given directory with the specified extension.
    
    Parameters:
    - directory (str): The directory to search in.
    - extension (str, optional): The file extension to look for. Defaults to '.dxf'.
    
    Returns:
    - str: The absolute path of the found file, or None if no file is found.
    """
    for filename in os.listdir(directory):
        if filename.endswith(extension):
            old_path = os.path.join(directory, filename)
            output_path = os.path.join(current_dir, "Output")
            shutil.copy(old_path, output_path)
            new_path = os.path.join(output_path, filename)
            return old_path, new_path
    return None, None