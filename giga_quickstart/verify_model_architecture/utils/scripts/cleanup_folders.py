
import os, shutil, glob
from pathlib import Path

def delete_before_start(output_dir):
    """"
    Deletes files and folders that are created during the execution to ensure a clean state before each run.
    This includes: files for comparing the python and cpp output, the output folders for the cpp and python output 
    and any existing bridge .so files, that were used to use the c code to be tested in python
    Args:        None
    """

    if output_dir.exists():
        shutil.rmtree(output_dir)