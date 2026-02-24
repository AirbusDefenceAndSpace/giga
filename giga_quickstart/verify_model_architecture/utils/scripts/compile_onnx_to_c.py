import subprocess
import os
from pathlib import Path

def compile_onnx_to_c(giga_repo, model_path, output_dir):
    """
    Compiles an ONNX model into C source code using the GIGA framework.
    
    This function sets the PYTHONPATH to include the GIGA source, ensures the 
    output directory exists, and runs the 'onnx_to_giga.py' tool with 
    Float32 precision settings.
    
    Args:
        giga_repo (str): Base path to the GIGA repository.
        model_path (str): Path to the .onnx file to be compiled.
        output_dir (str): Destination folder for the generated C code.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    python_src = Path(giga_repo) / "giga/python/src"
    export_tool = python_src / "GIGA_export/onnx_to_giga.py"
    
    env = os.environ.copy()
    env["PYTHONPATH"] = str(python_src) + ":" + env.get("PYTHONPATH", "")
    
    cmd = [
        "python3", str(export_tool),
        "-i", str(model_path),
        "-o", str(output_dir),
        "--input_type", "GIGA_Float32",
        "--output_type", "GIGA_Float32",
        "--intermediate_type", "GIGA_Float32",
        "--kernel_type", "GIGA_Float32"
    ]
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"ONNX to C compilation failed: {e}")