import numpy as np
import onnxruntime as ort
import os
from pathlib import Path
import cv2



def run_python_inference(modelpath_onnx, input_tensor_path, output_path):
    """    Executes inference on a given ONNX model using ONNX Runtime, with input data loaded from a specified text file.
    Args:
        modelpath_onnx (str): The file path to the ONNX model to be used for inference.
        input_tensor_path (str): The file path to the text file containing the input tensor data.
        output_path (str): The file path where the output tensor will be saved as a text file.
    """
    
    # # Configuration
    script_dir = Path(__file__).parent
    base_dir = script_dir.parents[1]

    # 1. Load the model and data
    session = ort.InferenceSession(modelpath_onnx)
    data = np.loadtxt(base_dir / input_tensor_path, dtype=np.float32)

    # 2. Reshape data to match model input
    input_name = session.get_inputs()[0].name
    data = data.reshape(session.get_inputs()[0].shape)

    # 3. Run inference
    outputs = session.run(None, {input_name: data})

    # Save
    np.savetxt(base_dir / output_path, outputs[0].flatten(), fmt='%.8f')

def create_input_tensor_from_image(input_tensor_path, testing_image_path):
    """    Converts an image file into a flattened text-based tensor for model testing.
    Args:
        input_tensor_path (str): The path where the output tensor text file will be saved.
        testing_image_path (str): The path to the input image file to be converted.
    """
    print(os.getcwd())
    script_dir = Path(__file__).parent
    base_dir = script_dir.parents[1]

    img = cv2.imread( str(base_dir / testing_image_path))
    tensor = img.transpose(2, 0, 1)
    np.savetxt(base_dir / input_tensor_path, tensor.flatten(), fmt='%.8f') 