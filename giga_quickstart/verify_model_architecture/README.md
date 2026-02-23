# Model Architecture Verification

Tool to verify the conversion of ONNX models into GIGA-compatible C code. 
The verification process ensures that the models architecture is correctly translated and that 
the inference results from the GIGA engine match the original ONNX model.

## Overview

The verification workflow performs a side-by-side comparison between:
1.  **Standard Python Inference:** Running the original `.onnx` model using `onnxruntime`.
2.  **GIGA Inference:** Running the converted C code using the GIGA inference engine via a C++ bridge.

## Directory Structure

*   `verify_model.py`: The main execution script that orchestrates the entire verification process.
*   `utils/`: Contains helper scripts for:
    *   `cleanup_folders.py`: Resetting the environment before each run.
    *   `compile_onnx_to_c.py`: Converting ONNX models to GIGA C code.
    *   `python_inference.py`: Running standard ONNX inference.
    *   `giga_inference.py`: Running inference on the converted GIGA model.
    *   `compare_outputs.py`: Analyzing and reporting differences between the two outputs.
    *   `images_to_test/`: Sample images used to generate input tensors.
*   `tested_outputs/`: Directory where intermediate files, generated C code, and final inference results are stored.

## Usage

Run the verification script by providing the path to the ONNX model and the root of the GIGA repository:

```bash
python verify_model.py --model_path <path_to_model.onnx> --gigarepo_path <path_to_giga_repo>
```

### Example

```bash
python verify_model.py \
    --model_path ../models/Model_Simple3x3Conv.onnx \
    --gigarepo_path /path/to/the/giga/repo
```

## How It Works

1.  **Cleanup:** Deletes previous outputs in the `tested_outputs/` folder to ensure a clean state.
2.  **Conversion:** The ONNX model is parsed and converted into optimized C code (`main_graph.c` and `main_graph.h`) located in `tested_outputs/output/`.
3.  **Input Preparation:** A test image is converted into a raw input tensor (`input_tensor.txt`).
4.  **Reference Inference:** The original model processes the input tensor via Python, saving results to `output_python.txt`.
5.  **GIGA Inference:** The converted C code is compiled and executed via a Python-C++ bridge, saving results to `output_giga.txt`.
6.  **Comparison:** The script compares `output_python.txt` and `output_giga.txt`. It checks for numerical parity and reports whether the conversion was successful.
