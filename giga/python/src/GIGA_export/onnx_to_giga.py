#!/usr/bin/env python3
"""
(C) 2024 Airbus copyright all rights reserved

author Lucas Marti (lucas.marti@airbus.com)
17/04/2023

Translation from ONNX files to C code for calling the Giga API.
"""

import torch
import os
from pathlib import Path
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser(prog='onnx_to_giga',
                            description='Translation from ONNX files to C code for calling the Giga API.')

    parser.add_argument('-i', '--input', required=True,
                        help="Name of the ONNX model to convert")
    parser.add_argument('-o', '--output', required=True,
                        help="Directory containing the C files after conversion")
    parser.add_argument('-s', '--memory_zone_size',
                        help="If a single memory zone is used, the size of that memory zone")
    parser.add_argument('--input_type',
                        help="Giga_type of the input", default="GIGA_UFixed8")
    parser.add_argument('--input_fp_shift',
                        help="Number of precision bits for input when using fixed point formats", default=4)
    parser.add_argument('--output_type',
                        help="Giga_type of the output", default="GIGA_SFixed16")
    parser.add_argument('--output_fp_shift',
                        help="Number of precision bits for output when using fixed point formats", default=4)
    parser.add_argument('--intermediate_type',
                        help="Giga_type of the input and output values in the middle of the network", default="GIGA_SFixed16")
    parser.add_argument('--intermediate_fp_shift',
                        help="Number of precision bits for intermediate tensors when using fixed point formats", default=4)
    parser.add_argument('--kernel_type',
                        help="Giga_type of the kernel and bias values in the middle of the network", default="GIGA_SFixed16")
    parser.add_argument('-a', '--allocator',
                        help="Specify the memory allocator to use. Valid values are: 'sequential', 'greedy'", default="sequential")

    args = parser.parse_args()
    print("Creating directories")
    os.makedirs(args.output, exist_ok=True)
    torch.manual_seed(0)
    input = Path(args.input)
    output = Path(args.output)

    print("Converting to NNEF")
    os.system(f"python3 -m nnef_tools.convert --keep-io-names --tensor-mapping {output.with_suffix('.nnef')/'tensor_mapping.json'} \
    --input-format onnx --output-format nnef --input-model {args.input} --output-model {output.with_suffix('.nnef')}")

    opts = ''

    if args.memory_zone_size:
        opts += f'-s {args.memory_zone_size} '
    if args.input_type:
        opts += f'--input_type {args.input_type} '
    if args.input_fp_shift:
        opts += f'--input_fp_shift {args.input_fp_shift} '
    if args.output_type:
        opts += f'--output_type {args.output_type} '
    if args.output_fp_shift:
        opts += f'--output_fp_shift {args.output_fp_shift} '
    if args.intermediate_type:
        opts += f'--intermediate_type {args.intermediate_type} '
    if args.intermediate_fp_shift:
        opts += f'--intermediate_fp_shift {args.intermediate_fp_shift} '
    if args.kernel_type:
        opts += f'--kernel_type {args.kernel_type} '
    if args.allocator:
        opts += f'--allocator {args.allocator} '

    print("Converting to GIGA")
    os.system(f"nnef_to_giga.py {opts} --input {output.with_suffix('.nnef')} --output {output}")

