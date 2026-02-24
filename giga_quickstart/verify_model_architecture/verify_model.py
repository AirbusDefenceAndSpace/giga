###### Import Libraries ######
import os, subprocess, ctypes ,cv2, json, numpy as np, onnxruntime as ort, csv, onnx, time, _ctypes,sys,shutil, glob, argparse
from pathlib import Path

##### import helper scripts ######
from utils.scripts.cleanup_folders import delete_before_start
from utils.scripts.compile_onnx_to_c import compile_onnx_to_c
from utils.scripts.python_inference import run_python_inference
from utils.scripts.python_inference import create_input_tensor_from_image
from utils.scripts.giga_inference import run_giga_inference
from utils.scripts.compare_outputs import compare_outputs


###### Set Variables ######
np.random.seed(42) #same random seed for reproducibility

###### import GIGA_utils ######
#currently only pt models are supported for clean up, onnx will come in the future?
#from convert_utils_onnx import replace_leaky_relu_onnx

def main(MODEL_PATH, GIGAREPO_PATH):
    """
    verify_model.py

    Main entry point for verifying ONNX model conversions against the GIGA inference engine.

    Workflow:
        1. Cleans up the output directory from previous runs.
        2. Converts the ONNX model to GIGA C code.
        3. Creates an input tensor from a test image.
        4. Runs inference with the original ONNX model via Python/ONNXRuntime.
        5. Runs inference with the converted GIGA model via a compiled C++ bridge.
        6. Compares both outputs and reports if the conversion was successful.

    Usage:
        python verify_model.py --model_path <path_to_model.onnx> --gigarepo_path <path_to_giga_repo>

    Arguments:
        --model_path      Path to the ONNX model to be tested.
        --gigarepo_path   Path to the root of the GIGA repository.

    Example:
        python verify_model.py \\
            --model_path ../models/Model_Simple3x3Conv.onnx \\
            --gigarepo_path /home/patri/Projekte/giga
    """
    
    INPUT_TENSOR_PATH = GIGAREPO_PATH / "giga_quickstart/verify_model_architecture/tested_outputs/input_tensor.txt"
    OUTPUT_PYTHON_PATH = GIGAREPO_PATH / "giga_quickstart/verify_model_architecture/tested_outputs/output_python.txt"
    OUTPUT_GIGA_PATH = GIGAREPO_PATH / "giga_quickstart/verify_model_architecture/tested_outputs/output_giga.txt"
    GIGA_LIB_PATH =  GIGAREPO_PATH / "giga_quickstart/verify_model_architecture/tested_outputs/"
    GIGA_BRIDGE_PATH = GIGAREPO_PATH / "giga_quickstart/verify_model_architecture/utils/scripts/giga_python_bridge.cpp"
    OUTPUT_PATH_CONVERTED = GIGAREPO_PATH / "giga_quickstart/verify_model_architecture/tested_outputs/output"
    IMAGE_PATH_TO_TEST = GIGAREPO_PATH / "giga_quickstart/verify_model_architecture/utils/images_to_test/orion.jpg"
    FOLDER_TO_CLEAN = GIGAREPO_PATH / "giga_quickstart/verify_model_architecture/tested_outputs"


    ###### Clean up before each start ######
    delete_before_start(FOLDER_TO_CLEAN)

    ###### Compile ONNX to C code ######
    compile_onnx_to_c(GIGAREPO_PATH, MODEL_PATH,OUTPUT_PATH_CONVERTED)

    ###### Create input tensor from image ######
    create_input_tensor_from_image(INPUT_TENSOR_PATH, IMAGE_PATH_TO_TEST)

    ###### Python inference for verification ######
    run_python_inference(MODEL_PATH, INPUT_TENSOR_PATH, OUTPUT_PYTHON_PATH)

    ###### GIGA inference and comparison with python output ######

    model_name = MODEL_PATH.stem
    run_giga_inference(INPUT_TENSOR_PATH,OUTPUT_GIGA_PATH, model_name, giga_repo=GIGAREPO_PATH, output_dir=GIGA_LIB_PATH, giga_python_bridge=GIGA_BRIDGE_PATH)

    ###### Compare the outputs from python and giga ######
    print("Results for the model {}:".format(model_name))
    compare_outputs(OUTPUT_PYTHON_PATH, OUTPUT_GIGA_PATH)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        required=True, 
        type=str, 
        help="Path to th model that should be tested. Default is ../models/Model_Simple3x3Conv.onnx"
    )
    parser.add_argument(
        "--gigarepo_path",
        required=True,
        type=str, 
        help="Path to the giga repo. From the this file its ../... two folders up"
    )
    
    args = parser.parse_args()
    
    ARG_MODEL_PATH = Path(args.model_path)
    ARG_MODEL_PATH = ARG_MODEL_PATH.resolve()

    ARG_GIGAREPO_PATH = Path(args.gigarepo_path)
    ARG_GIGAREPO_PATH = ARG_GIGAREPO_PATH.resolve()

    #Check if the model path exists
    if ARG_MODEL_PATH.exists() and ARG_GIGAREPO_PATH.exists():
        print(f"Using model from path: {ARG_MODEL_PATH}")
        main(ARG_MODEL_PATH, ARG_GIGAREPO_PATH)
    else:
        print(f"Model path {ARG_MODEL_PATH} or GIGA repo path {ARG_GIGAREPO_PATH} does not exist. Please provide valid paths.")


