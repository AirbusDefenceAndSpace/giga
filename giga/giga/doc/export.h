/*!
 * (C) 2025 Airbus copyright all rights reserved
 */
/*!
 * \page net_export Exporting neural networks
 *
 * GIGA provides onnx_to_giga.py and nnef_to_giga.py scripts to convert an ONNX or a NNEF network to GIGA C code.
 *
 * \dot Typical workflow when using PyTorch
 * digraph workflow_pytorch
 * {
 *  rankdir = LR;
 *  node[shape=box, style="rounded,filled", fillcolor="#3060ff"];
 *  GIGA [URL="\ref GIGA"];
 *  NNEF [URL="https://www.khronos.org/nnef"];
 *  ONNX [URL="https://onnx.ai"];
 *  PyTorch [URL="https://pytorch.org"];
 *
 *  PyTorch -> ONNX -> NNEF -> GIGA;
 * }
 * \enddot
 *
 * \dot Typical workflow when using TensorFlow
 * digraph workflow_tensorflow
 * {
 *  rankdir = LR;
 *  node[shape=box, style="rounded,filled", fillcolor="#3060ff"];
 *  GIGA [URL="\ref GIGA"];
 *  NNEF [URL="https://www.khronos.org/nnef"];
 *  TensorFlow [URL="https://www.tensorflow.org"];
 *
 *  TensorFlow -> NNEF -> GIGA;
 * }
 * \enddot
 *
 * It is recommended to cleanup your networks before exporting them. For instance you likely want to do the following:
 * - merge BatchNormalization layers with Conv2D/Dense/... layers
 * - replace automatic padding parameters with actual integer values (PyTorch doesn't support exporting values like 'same', 'valid' or 'extended')
 * - replace unsupported activation functions with a supported equivalent (ie. replace LeakyReLU with ReLU)
 *
 * The generated code embed both structure (as C code) and weights (as constant arrays). The network is converted layer by layer to GIGA. Some layers
 * may be implemented with more than one GIGA function. For instance linear upsampling is implemented using nearest neighbor upsampling followed by a
 * depth-wise convolution with an averaging filter. Some operations may be implemented implicitly such as tensor concatenation which makes them free.
 */
