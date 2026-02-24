import subprocess
import ctypes
import os
import time

"""
This module provides a compilation and execution wrapper 
for GIGA inference. It bridges Python logic with the shared libraries.

Main Workflow:
1. Compiles a C++ bridge source into a Shared Object (.so) using g++.
2. Links against local GIGA and GIGA_cpu system libraries.
3. Loads the resulting binary via ctypes.
4. Executes 'run_inference' with path-based string arguments.

At the moment, it focues only on the cpu part, but it can be easily adapted to support other backends by changing the linked libraries and include paths.

"""


def run_giga_inference(input_txt, output_txt, model_name, giga_repo, output_dir, giga_python_bridge):
    lib_name = f"libgiga_bridge_{model_name}.so"
    lib_path = os.path.abspath(os.path.join(output_dir, lib_name))
    cmd = (
        f"g++ -O3 -shared -fPIC {giga_python_bridge} "
        f"-I . -I /usr/local/include/giga -I /usr/local/include "
        f"-I {giga_repo}/giga "
        f"-I {giga_repo}/giga_soft/cpu "
        f"-lGIGA_cpu -lGIGA -o {lib_path}"
    )

    try:
        subprocess.run(cmd.split(), check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Compilation failed for model '{model_name}':\n{e.stderr}")
        return False

    time.sleep(0.2)

    if not os.path.exists(lib_path):
        print(f"Library not found after compilation: {lib_path}")
        return False

    try:
        giga_lib = ctypes.CDLL(lib_path)
        giga_lib.run_inference.argtypes = [ctypes.c_char_p, ctypes.c_char_p]

        giga_lib.run_inference(str(input_txt).encode(), str(output_txt).encode())
        return True

    except OSError as e:
        print(f"Failed to load shared library '{lib_path}': {e}")
    except AttributeError:
        print(f"'run_inference' not found in '{lib_name}' check the exported symbols.")
    except Exception as e:
        print(f"Inference error for model '{model_name}': {e}")

    return False
