#!/usr/bin/env python3
"""\
(C) 2024 Airbus copyright all rights reserved

author Lucas Marti (lucas.marti@airbus.com)
17/04/2023

author Roland Brochard (roland.brochard@airbus.com)
28/03/2025

Translation from NNEF files to C code for calling the GIGA API.
"""

import os

import nnef
from pathlib import Path
from argparse import ArgumentParser
import numpy as np
from typing import List
from dataclasses import dataclass

#%%
class GIGA_sequential_memory_allocator:
    def __init__(self,
                 memory_size: int = 0):
        self.memory_size = memory_size
        self.next = 0
        self.total_used = 0
        
    def allocate_tensor(self,
                        tensor_name: str,
                        shape : List[int],
                        element_type: str = 'GIGA_Float32'):
        if element_type == 'GIGA_Float32':
            tensor_size = 4
        elif element_type == 'GIGA_Float16' or element_type == 'GIGA_SFixed16' or element_type == 'GIGA_UFixed16':
            tensor_size = 2
        elif element_type == 'GIGA_SFixed8' or element_type == 'GIGA_UFixed8':
            tensor_size = 1
        else:
            tensor_size = 1
        for s in shape:
            tensor_size *= int(s)
        tensor_offset = self.next
        allocated_size = (tensor_size + 7) // 8 * 8 # Round to 8 bytes
        self.next += allocated_size
        self.total_used += allocated_size
        
        return 0, tensor_offset;    # Memory zone ID, offset in bytes
    
    def release_tensor(self,
                       tensor_name: str):
        pass;
        
    def memory_used(self):
        return self.total_used;

    def memory_needed(self):
        return self.total_used;

class GIGA_greedy_memory_allocator:
    def __init__(self,
                 memory_size: int = 0):
        self.memory_size = memory_size
        self.free_blocks = {0: memory_size}
        self.total_needed = 0
        self.tensors = {}
        
    def allocate_tensor(self,
                        tensor_name: str,
                        shape : List[int],
                        element_type: str = 'GIGA_Float32'):
        if element_type == 'GIGA_Float32':
            tensor_size = 4
        elif element_type == 'GIGA_Float16' or element_type == 'GIGA_SFixed16' or element_type == 'GIGA_UFixed16':
            tensor_size = 2
        elif element_type == 'GIGA_SFixed8' or element_type == 'GIGA_UFixed8':
            tensor_size = 1
        else:
            tensor_size = 1
        for s in shape:
            tensor_size *= int(s)

        allocated_size = (tensor_size + 7) // 8 * 8
        
        tensor_offset = None
        best_match_size = self.memory_size + 1
        for block in self.free_blocks:
            block_size = self.free_blocks[block];
            if block_size >= allocated_size and block_size < best_match_size:
                best_match_size = block_size;
                tensor_offset = block;
        
        if tensor_offset is None:
            raise RuntimeError("Not enough memory to allocate tensor!");
        
        self.tensors[tensor_name] = (tensor_offset, allocated_size)
       
        if best_match_size == allocated_size:       # Exact match
            self.free_blocks.pop(tensor_offset)     # Remove the free block
        else:                                       # If free block is larger than requested size
            self.free_blocks.pop(tensor_offset)     # Remove the beginning of the block
            self.free_blocks[tensor_offset + allocated_size] = best_match_size - allocated_size
        
        self.total_needed = max(self.total_needed, tensor_offset + allocated_size);

        return 0, tensor_offset;    # Memory zone ID, offset in bytes
    
    def release_tensor(self,
                       tensor_name: str):
        if tensor_name not in self.tensors:
            raise RuntimeError(f"Unknown tensor '{tensor_name}'")
        offset, size = self.tensors.pop(tensor_name)
        
        connected_block_before = None
        connected_block_after = None
        # Detect continuous blocks before and after the released block to perform a merge if needed
        for block in self.free_blocks:
            block_size = self.free_blocks[block];
            if block + block_size == offset:
                connected_block_before = block;
                if connected_block_after is not None:
                    break;
            elif block == offset + size:
                connected_block_after = block;
                if connected_block_before is not None:
                    break;
        
        if connected_block_before is None and connected_block_after is None:        # Simple case without merge
            self.free_blocks[offset] = size;
        elif connected_block_before is not None and connected_block_after is None:  # Merge with previous block only
            size += self.free_blocks[connected_block_before];
            self.free_blocks[connected_block_before] = size;
        elif connected_block_before is None and connected_block_after is not None:  # Merge with next block only
            size += self.free_blocks[connected_block_after];
            self.free_blocks.pop(connected_block_after);
            self.free_blocks[offset] = size;
        else:                                                                       # Merge both sides
            size += self.free_blocks[connected_block_after];
            size += self.free_blocks[connected_block_before];
            self.free_blocks.pop(connected_block_after);
            self.free_blocks[connected_block_before] = size;
            
        pass;
        
    def memory_used(self):
        total_free = 0
        for k in self.free_blocks:
            total_free += self.free_blocks[k];
        return self.memory_size - total_free;

    def memory_needed(self):
        return self.total_needed;

#%%
@dataclass
class TensorInfo:
    name: str
    prefix: str
    giga_type: str
    c_type: str
    is_kernel: bool
    declared: bool
    nb_dims: int

#%%
class GIGA_Code_Generator:
    """

    """

    def __init__(self, nnef_directory: Path,
                 verbose_code: bool = False,
                 memory_zone_size: int = 0,
                 input_type: str = "GIGA_UFixed8",
                 input_fp_shift: int = 4,
                 output_type: str = "GIGA_SFixed16",
                 output_fp_shift: int = 4,
                 intermediate_type: str = "GIGA_SFixed16",
                 intermediate_fp_shift: int = 4,
                 kernel_type: str = "GIGA_SFixed16",
                 memory_allocator: str = "sequential") -> None:
        """
 
        :param: nnef_directory: The location of the exported NNEF files
        """
        self.dir_path = nnef_directory
        self.graph = nnef.parse_file((self.dir_path / "graph.nnef").__str__())
        self.verbose_code = verbose_code
        if memory_allocator == "sequential":
            self.allocator = GIGA_sequential_memory_allocator(memory_zone_size)
        elif memory_allocator == "greedy":
            self.allocator = GIGA_greedy_memory_allocator(memory_zone_size)
        else:
            self.allocator = GIGA_sequential_memory_allocator(memory_zone_size)

        nnef.infer_shapes(self.graph)

        self.network_name = self.graph.name

        # Creation of each function and structure necessary
        self.header_string = (f'#include "{self.network_name}.h"\n'
                              f'#include <stdio.h>\n')
        self.parameters_structure_string = f"typedef struct {self.network_name}_tensors{{\n"
        self.op_structure_string = f"typedef struct {self.network_name}_ops{{\n"
        self.io_structure_string = f"typedef struct {self.network_name}_io{{\n"

        self.header_file = "#include <giga/giga.h>\n#include <string.h>\n#include <giga/utils.h>"

        self.initialize_string = f"int initialize_{self.network_name}(uint32_t *device_id)"

        self.allocate_tensors_string = \
            f"int allocate_{self.network_name}_tensors({self.network_name}_tensors *tensors, " \
            f"{self.network_name}_io *io, uint32_t device_id)"

        self.fill_string = f"int fill_{self.network_name}_tensors({self.network_name}_tensors *tensors)"

        self.set_operations_string = f"int set_{self.network_name}_ops({self.network_name}_ops *ops_params, " \
                                     f"{self.network_name}_tensors *tensors)"

        self.process_string = f"int process_{self.network_name}_tensors({self.network_name}_tensors *tensors, const {self.network_name}_ops *ops_params, {self.network_name}_io *io)"

        # Function prototypes in the header file need a ;
        self.prototypes_string = self.initialize_string + ";\n\n" \
                                 + self.allocate_tensors_string + ";\n\n" \
                                 + self.fill_string + ";\n\n" \
                                 + self.set_operations_string + ";\n\n" \
                                 + self.process_string + ";\n\n"

        # Add missing opening bracket for function definition.
        self.allocate_tensors_string += "{\n    GIGA_error error;\n"
        if self.verbose_code:
            self.allocate_tensors_string += '    printf("Allocating\\n");\n'
        self.fill_string += "{\n    GIGA_error error;\n"
        if self.verbose_code:
            self.fill_string += '    printf("Filling\\n");\n'
        self.set_operations_string += "{\n    GIGA_error error;\n"
        if self.verbose_code:
            self.set_operations_string += '    printf("Defining operations\\n");\n'
        self.process_string += "{\n    GIGA_error error;\n"
        if self.verbose_code:
            self.process_string += '    printf("Processing\\n");\n'

        self.initialize_string += ('{\n'
                                     '    GIGA_error error;\n'
                                     '    *device_id = giga_get_default_device_id(&error);\n'
                                     '\n'
                                     '    if(error != GIGA_Success)\n'
                                     '        return error;\n'
                                     '\n'
                                     '    error = giga_initialize_device(*device_id);\n'
                                     '    if(error != GIGA_Success)\n'
                                     '        return error;\n')

        self.avg_pool_declared = False
        self.op_index = 0

        self.declared_tensors = set()  # Names mapped to offsets in their memory zones

        self.giga_input_type = input_type
        self.input_type = GIGA_Code_Generator.get_C_type(self.giga_input_type)
        self.input_fp_shift = input_fp_shift
        self.giga_output_type = output_type
        self.output_type = GIGA_Code_Generator.get_C_type(self.giga_output_type)
        self.output_fp_shift = output_fp_shift
        self.giga_intermediate_type = intermediate_type
        self.intermediate_type = GIGA_Code_Generator.get_C_type(self.giga_intermediate_type)
        self.intermediate_fp_shift = intermediate_fp_shift

        self.giga_kernel_type = kernel_type
        self.kernel_type = GIGA_Code_Generator.get_C_type(self.giga_kernel_type)

        self.kernels = set()
        self.biases = set()
        self.implicit_tensors = set()    # Tensors which views/slices of other tensors and are therefore not allocated
        # Go through the list of operations
        self.process_list = [None for _ in self.graph.operations]

        # Detect the number of times tensors are read/used
        # We need this information to detect when to "free" tensors in order to recycle their memory
        remaining_tensor_uses = {};
        def mark_tensor_use(tensor_name):
            if tensor_name in remaining_tensor_uses:
                remaining_tensor_uses[tensor_name] += 1;
            else:
                remaining_tensor_uses[tensor_name] = 1;
            # In case it's an output, make sure it is never deleted
            # Inputs can be recycled (we could make this optional in case we want to reuse input tensors)
            if tensor_name in self.graph.outputs:
                remaining_tensor_uses[tensor_name] += 1;
        
        for index, operation in enumerate(self.graph.operations):  # First declare the memory layout sensitive tensors
            if 'input' in operation.inputs:
                mark_tensor_use(operation.inputs['input']);
            if 'filter' in operation.inputs:
                self.kernels.add(operation.inputs['filter']);
            if 'bias' in operation.inputs:
                self.biases.add(operation.inputs['bias']);
        
        for index, operation in enumerate(self.graph.operations):  # First declare the memory layout sensitive tensors
            if operation.name == "concat":
                self.declare_concat(operation, index)
                
        def free_inputs(operation):
            if 'input' in operation.inputs:
                tensor_name = operation.inputs['input'];
                if tensor_name in remaining_tensor_uses:
                    remaining_tensor_uses[tensor_name] -= 1;
                    if remaining_tensor_uses[tensor_name] == 0:
                        if tensor_name not in self.implicit_tensors:
                            self.allocator.release_tensor(tensor_name)

        # Go through the list of variables (and allocate them before work tensors)
        for index, operation in enumerate(self.graph.operations):
            if operation.name == 'variable':
                self.declare_fill(self.graph.tensors[operation.attribs['label']])

        for index, operation in enumerate(self.graph.operations):
            if operation.name == 'conv':
                ret, op = self.look_for_relu_after(operation)
                # integrates the ReLU into the convolutions
                if ret:
                    operation.outputs['output'] = op.outputs['y']
                    self.declare_conv(operation, with_relu=True, index=index)
                else:
                    self.declare_conv(operation, with_relu=False, index=index)
                    
            # TODO: support dense layers

            # We assume 'relu' as already processed as part of a previous layer (either conv or dense)
            if operation.name == 'relu':
                continue

            if operation.name == "avg_pool":
                self.declare_avg_pool(operation, index)

            if operation.name == "multilinear_upsample":
                self.declare_nearest_upsample(operation, index)  # TODO Transform this into a proper multilinear

            if operation.name == "nearest_upsample":
                self.declare_nearest_upsample(operation, index)

            if operation.name == "batch_normalization":
                self.declare_batch_normalization(operation, index)

            self.op_index += 1
            
            free_inputs(operation)

        self.parameters_structure_string += f"}} {self.network_name}_tensors;\n"
        self.op_structure_string += f"}} {self.network_name}_ops;\n"
        self.io_structure_string += f"}} {self.network_name}_io;\n"
        self.initialize_string += "    return 0;\n}\n"
        self.allocate_tensors_string += "    return 0;\n}\n"
        self.fill_string += "    return 0;\n}\n"
        self.set_operations_string += "    return 0;\n}\n"

        self.process_string += "\n".join(filter(lambda x: x is not None, self.process_list))
        self.process_string += "    return 0;\n}\n"

    def get_pretty_graph_string(self) -> str:
        return nnef.format_graph(self.graph.name, self.graph.inputs, self.graph.outputs, self.graph.operations,
                                 self.graph.tensors)

    @staticmethod
    def get_C_type(GIGA_type: str) -> str:
        return {"GIGA_Float16": "half",
                "GIGA_Float32": "float",
                "GIGA_SFixed4": "int4_t",
                "GIGA_SFixed8": "int8_t",
                "GIGA_SFixed16": "int16_t",
                "GIGA_UFixed4": "uint4_t",
                "GIGA_UFixed8": "uint8_t",
                "GIGA_UFixed16": "uint16_t",
                }.get(GIGA_type)

    @staticmethod
    def get_GIGA_type(C_type: str) -> str:
        return {"half": "GIGA_Float16",
                "float": "GIGA_Float32",
                "int4_t": "GIGA_SFixed4",
                "int8_t": "GIGA_SFixed8",
                "int16_t": "GIGA_SFixed16",
                "uint4_t": "GIGA_UFixed4",
                "uint8_t": "GIGA_UFixed8",
                "uint16_t": "GIGA_UFixed16",
                }.get(C_type)

    @staticmethod
    def is_fixed(GIGA_type: str) -> bool:
        return {"GIGA_Float16": False,
                "GIGA_Float32": False,
                "GIGA_SFixed4": True,
                "GIGA_SFixed8": True,
                "GIGA_SFixed16": True,
                "GIGA_UFixed4": True,
                "GIGA_UFixed8": True,
                "GIGA_UFixed16": True,
                }.get(GIGA_type)

    @staticmethod
    def get_type_size_in_bits(GIGA_type: str) -> bool:
        return {"GIGA_Float16": 16,
                "GIGA_Float32": 32,
                "GIGA_SFixed4": 4,
                "GIGA_SFixed8": 8,
                "GIGA_SFixed16": 16,
                "GIGA_UFixed4": 4,
                "GIGA_UFixed8": 8,
                "GIGA_UFixed16": 16,
                }.get(GIGA_type)

    @staticmethod
    def get_nb_bytes_from_c_type(c_type: str) -> int:
        return {
            "float": 4,
            "half": 2,
            "uint8_t": 1,
            "uint16_t": 2,
            "int8_t": 1,
            "int16_t": 2
        }.get(c_type)

    def declare_avg_pool(self, avg_operation: nnef.Operation, index) -> None:
        """
        Declares average pooling operation using the convolution operation and a specially made tensor. Each channel is
        convoluted by this tensor using views.
        :param: avg_operation: The operation.
        :param: index: the index of the operation in the graph.
        :return: None
        """
        operation_name = f"op_{self.op_index}"
        input_name = avg_operation.inputs['input']
        output_name = avg_operation.outputs['output']

        prefix_i = "tensors"
        if input_name in self.graph.inputs or input_name in self.graph.outputs:
            prefix_i = "io"

        prefix_o = "tensors"
        if output_name in self.graph.inputs or output_name in self.graph.outputs:
            prefix_o = "io"

        if input_name not in self.declared_tensors:
            self.declare_tensor(self.graph.tensors[input_name])
        if output_name not in self.declared_tensors:
            self.declare_tensor(self.graph.tensors[output_name])

        nb_dims = len(self.graph.tensors[input_name].shape)
        nb_chans = 1
        if nb_dims > 2:
            nb_chans = self.graph.tensors[input_name].shape[nb_dims - 3]

            # Declare views for each channels
            for chan in range(nb_chans):
                self.declare_slice(self.graph.tensors[input_name], nb_dims - 3, chan)
                self.declare_slice(self.graph.tensors[output_name], nb_dims - 3, chan)

        kernel_name = self.declare_avg_pool_tensor()

        self.op_structure_string += f"    GIGA_conv2d_t {operation_name}_params;\n"
        self.set_operations_string += ('\n'
                                       f'    ops_params->{operation_name}_params = (GIGA_conv2d_t){{\n'
                                        '        .padding = { { 1, 1 }, { 1, 1 } },\n'
                                        '        .stride = { 2, 2 },\n'
                                        '        .dilation = { 1, 1 },\n'
                                        '        .b_ReLU = false,\n'
                                       f'        .kernel = &tensors->{kernel_name},\n'
                                        '        .bias = NULL,\n'
                                        '        };\n')

        self.process_list[index] = ""
        if self.verbose_code:
            self.process_list[index] += f'    printf("{operation_name}\\n");\n'
        if nb_dims > 2:
            self.process_list[index] += "    /* Avg pooling */\n"
            for chan in range(nb_chans):
                self.process_list[index] += ('\n'
                                 f'    if((error = giga_conv2d(&ops_params->{operation_name}_params, &tensors->{input_name}_{chan}, &tensors->{output_name}_{chan})) != GIGA_Success)\n'
                                 '        return error;\n')
        else:
            self.process_list[index] += ('\n'
                             f'    if((error = giga_conv2d(&ops_params->{operation_name}_params, &{prefix_i}->{input_name}, &{prefix_o}->{output_name})) != GIGA_Success)\n'
                             '        return error;')
            
    def set_tensor_params(self, tensor_type, fp_shift, shape) -> str:
        nb_dims = len(shape)
        ret = (f'(GIGA_tensor_t){{\n'
               f'        .device_id = device_id,\n'
               f'        .nb_dims = {nb_dims},\n'
               f'        .type = {tensor_type},\n'
               f'        .dims = {{')

        for dim in range(nb_dims):
            ret += f"{shape[dim]}, "

        for dim in range(nb_dims, 4):
            ret += "0, "

        ret += '},\n'
        ret += f'        .fp_shift = {fp_shift},\n'
        ret += '        .data = NULL,\n'
        ret += '        };\n'
        return ret
    
    def get_tensor_info(self,
                        tensor: str):
        tensor_name = tensor.name;
        tinfo = TensorInfo(
                name = tensor_name,
                prefix = "tensors",
                giga_type = self.giga_intermediate_type,
                c_type = self.intermediate_type,
                is_kernel = False,
                declared = tensor_name in self.declared_tensors,
                nb_dims = len(tensor.shape)
                )

        if (tinfo.name in self.kernels) or (tinfo.name in self.biases):
            tinfo.giga_type = self.giga_kernel_type
            tinfo.c_type = self.kernel_type
            tinfo.is_kernel = True
        else:
            tinfo.giga_type = self.giga_intermediate_type
            tinfo.c_type = self.intermediate_type
        if tinfo.name in self.graph.inputs or tinfo.name in self.graph.outputs:
            tinfo.prefix = "io"
            if tinfo.name in self.graph.inputs:
                tinfo.giga_type = self.giga_input_type
                tinfo.c_type = self.input_type
            else:
                tinfo.giga_type = self.giga_output_type
                tinfo.c_type = self.output_type
        
        return tinfo;

    def declare_slice(self, tensor: str, dimension: int, slice_number: int) -> None:

        og_tensor = self.get_tensor_info(tensor)
        
        tensor_name = f"{og_tensor.name}_{slice_number}"
        if tensor_name in self.declared_tensors:
            return
        
        fp_shift = 0
        if GIGA_Code_Generator.is_fixed(og_tensor.giga_type):
            fp_shift = self.get_fp_shift(og_tensor.name)
        self.parameters_structure_string += f"    GIGA_tensor_t {tensor_name};\n"
        shape = [tensor.shape[i] if i != dimension else 1 for i in range(og_tensor.nb_dims)]
        self.allocate_tensors_string += f'    tensors->{tensor_name} = ' + self.set_tensor_params(og_tensor.giga_type, fp_shift, shape)
        
        self.declared_tensors.add(tensor_name)

        self.allocate_tensors_string += ('\n'
                                         f'    GIGA_view_t view_params_{tensor_name};\n'
                                         f'    memset(&view_params_{tensor_name}, 0, sizeof(GIGA_view_t));\n'
                                         f'    view_params_{tensor_name}.offset[{dimension}] = {slice_number};\n'
                                         f'    error = giga_view( &view_params_{tensor_name}, &{og_tensor.prefix}->{og_tensor.name}, &tensors->{tensor_name});\n'
                                          '    if(error != GIGA_Success)\n'
                                          '        return error;\n'
                                          '\n')

    def declare_nearest_upsample(self, nearest_upsample_operation: nnef.Operation, index) -> None:
        """

        :param nearest_upsample_operation:
        :param index:
        :return:
        """

        if nearest_upsample_operation.attribs['factor'][0] != 2 or nearest_upsample_operation.attribs['factor'][1] != 2:
            print("Wrong upsampling factor")
            exit(-1)

        operation_name = f"op_{self.op_index}"
        input_name = nearest_upsample_operation.inputs['input']
        output_name = nearest_upsample_operation.outputs['output']

        if input_name not in self.declared_tensors:
            self.declare_tensor(self.graph.tensors[input_name])
        if output_name not in self.declared_tensors:
            self.declare_tensor(self.graph.tensors[output_name])

        prefix_i = "tensors"
        if input_name in self.graph.inputs or input_name in self.graph.outputs:
            prefix_i = "io"

        prefix_o = "tensors"
        if output_name in self.graph.inputs or output_name in self.graph.outputs:
            prefix_o = "io"

        self.op_structure_string += f"    GIGA_upsample_t {operation_name}_params;\n"
        self.set_operations_string += f"\n    ops_params->{operation_name}_params.factor = 2;\n"

        self.process_list[index] = ""
        if self.verbose_code:
            self.process_list[index] += f'    printf("{operation_name}\\n");\n'
        self.process_list[index] += ('    /* Nearest upsampling */\n'
                                     f'    if((error = giga_upsample(&ops_params->{operation_name}_params, &{prefix_i}->{input_name}, &{prefix_o}->{output_name})) != GIGA_Success)\n'
                                      '        return error;\n')

    def declare_batch_normalization(self, batch_norm_operation: nnef.Operation, index):
        input_name = batch_norm_operation.inputs['input']
        if input_name not in self.declared_tensors:
            self.declare_tensor(self.graph.tensors[input_name])

        output_name = batch_norm_operation.outputs['output']
        if output_name not in self.declared_tensors:
            self.declare_tensor(self.graph.tensors[output_name])

        mean_name = batch_norm_operation.inputs['mean']
        mean_values, mean_type, mean_dimensions = self.get_data_values((self.dir_path / mean_name).with_suffix(".dat"))

        variance_name = batch_norm_operation.inputs['variance']
        variance_values, variance_type, variance_dimensions = self.get_data_values(
            (self.dir_path / variance_name).with_suffix(".dat"))

        offset_name = batch_norm_operation.inputs['offset']
        offset_values, offset_type, offset_dimensions = self.get_data_values(
            (self.dir_path / offset_name).with_suffix(".dat"))

        scale_name = batch_norm_operation.inputs['scale']
        scale_values, scale_type, scale_dimensions = self.get_data_values((self.dir_path / scale_name).with_suffix(".dat"))

        epsilon = batch_norm_operation.attribs['epsilon']

        denom = np.sqrt(variance_values + epsilon)
        a = scale_values / denom
        b = offset_values - (a * mean_values)

        kernels, biases = self.declare_batch_norm_tensors(a, b, batch_norm_operation)

        nb_dims = len(self.graph.tensors[input_name].shape)
        nb_chans = 1
        if nb_dims > 2:
            nb_chans = self.graph.tensors[input_name].shape[nb_dims - 3]

        for chan in range(nb_chans):
            self.declare_slice(self.graph.tensors[input_name], nb_dims - 3, chan)
            self.declare_slice(self.graph.tensors[output_name], nb_dims - 3, chan)

        nb_operations = len(kernels)
        operation_name = f"op_{self.op_index}"

        self.process_list[index] = "    /* Batch normalization */\n"
        for i in range(nb_operations):
            self.op_structure_string += f"    GIGA_conv2d_t {operation_name}_{i}_params;\n"
            self.set_operations_string += ('\n'
                                           f'    ops_params->{operation_name}_{i}_params = (GIGA_conv2d_t){{\n'
                                            '        .padding = { { 1, 1 }, { 1, 1 } },\n'
                                            '        .stride = { 1, 1 },\n'
                                            '        .dilation = { 1, 1 },\n'
                                            '        .b_ReLU = false,\n'
                                           f'        .kernel = &tensors->{kernels[i]},\n'
                                            '        .bias = &tensors->{biases[i]}\n'
                                            '        };\n')

            if self.verbose_code:
                self.process_list[index] += f'    printf("{operation_name}_{i}\\n");\n'
            self.process_list[index] += ('\n'
                             f'    if((error = giga_conv2d(&ops_params->{operation_name}_{i}_params, &tensors->{input_name}_{i}, &tensors->{output_name}_{i})) != GIGA_Success)\n'
                             '        return error;\n')

    def declare_batch_norm_tensors(self, a: np.ndarray, b: np.ndarray, batch_norm_operation: nnef.Operation) \
            -> (List, List):

        kernels = []
        biases = []

        operation_name = f"op_{self.op_index}"
        for i in range(a.shape[0]):
            # Allocation of the kernel
            tensor_name = f"{operation_name}_kernel_{i}"
            kernels.append(tensor_name)

            self.declare_tensor({'name':tensor_name, 'shape': [1,1,3,3]})

            self.fill_string += ('\n'
                                 f'    {self.intermediate_type} data_{tensor_name}[] = {{0, 0, 0, 0, {a[i]:.10f}, 0, 0, 0, 0}};\n'
                                 f'    if ((error = giga_copy_to_tensor(data_{tensor_name}, {self.giga_intermediate_type}, 0, &tensors->{tensor_name})) != GIGA_Success)\n'
                                  '        return error;\n')

            # Allocation of the bias
            tensor_name = f"{operation_name}_bias_{i}"
            biases.append(tensor_name)

            self.declare_tensor({'name':tensor_name, 'shape': [1]})

            self.fill_string += ('\n'
                                 f'    {self.intermediate_type} data_{tensor_name}[] = {{{b[i]}}};\n'
                                 f'    if ((error = giga_copy_to_tensor(data_{tensor_name}, {self.giga_intermediate_type}, 0, &tensors->{tensor_name})) != GIGA_Success)\n'
                                  '        return error;\n')

        return kernels, biases

    def declare_concat(self, concat_operation: nnef.Operation, index) -> None:
        """
        Concatenation is for now only possible using views. In the future it will be improved depending on the actual graph
        :param: concat_operation
        :param: index
        :return: None
        """

        operation_name = f"op_{self.op_index}"
        input_name1 = concat_operation.inputs['values'][0]
        input_name2 = concat_operation.inputs['values'][1]
        output_name = concat_operation.outputs['value']

        self.declare_tensor(self.graph.tensors[output_name])

        self.declare_tensor(self.graph.tensors[input_name1], is_view=True)
        self.declare_tensor(self.graph.tensors[input_name2], is_view=True)
        
        input2 = self.graph.tensors[input_name2]
        
        prefix_i1 = "tensors"
        if input_name1 in self.graph.inputs or input_name1 in self.graph.outputs:
            prefix_i1 = "io"

        prefix_i2 = "tensors"
        if input_name2 in self.graph.inputs or input_name2 in self.graph.outputs:
            prefix_i2 = "io"

        prefix_o = "tensors"
        if output_name in self.graph.inputs or output_name in self.graph.outputs:
            prefix_o = "io"

        self.allocate_tensors_string += ('\n'
                                         f'    GIGA_view_t view_params_{input_name1};\n'
                                         f'    memset(&view_params_{input_name1}, 0, sizeof(GIGA_view_t));\n'
                                         f'    if((error = giga_view(&view_params_{input_name1}, &{prefix_o}->{output_name}, &{prefix_i1}->{input_name1})) != GIGA_Success)\n'
                                         '        return error;\n'
                                         '\n')

        self.allocate_tensors_string += (f'    GIGA_view_t view_params_{input_name2};\n'
                                         f'    memset(&view_params_{input_name2}, 0, sizeof(GIGA_view_t));\n'
                                         f'    view_params_{input_name2}.offset[1] = {input2.shape[1]};\n'
                                         f'    if((error = giga_view(&view_params_{input_name2}, &{prefix_o}->{output_name}, &{prefix_i2}->{input_name2})) != GIGA_Success)\n'
                                         '        return error;\n'
                                         '\n')

    def look_for_relu_after(self, operation: nnef.Operation) -> (bool, nnef.Operation):
        output_name = operation.outputs['output']
        for op in self.graph.operations:
            if op.name == 'relu':
                if op.inputs['x'] == output_name:
                    return True, op

        return False, None

    def declare_avg_pool_tensor(self) -> str:
        """
        :return: The name of the declared tensor
        """
        tensor_name = "avg_pool_kernel"
        if self.avg_pool_declared:
            return tensor_name

        self.declare_tensor({'name':tensor_name, 'shape': [1,1,3,3]})

        # Fill the upper half of the tensor
        self.fill_string += ('\n'
                             f'    {self.intermediate_type} data_{tensor_name}[] = {{0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.0, 0.25, 0.25}};\n'
                             f'    if ((error = giga_copy_to_tensor(data_{tensor_name}, {self.giga_intermediate_type}, 0, &tensors->{tensor_name})) != GIGA_Succes)\n'
                              '        return error;\n')

        self.avg_pool_declared = True
        self.kernels.add(tensor_name)
        return tensor_name

    def declare_tensor(self, tensor, is_view: bool = False) -> int:
        """

        :param tensor:
        :param is_view: True if the tensor is a view of another tensor
        :return: Total tensor size
        """
        tinfo = self.get_tensor_info(tensor)
        if tinfo.declared:
            return -1;
        
        if tinfo.prefix == "io":
            self.io_structure_string += f"    GIGA_tensor_t {tinfo.name};\n"
        else:
            self.parameters_structure_string += f"    GIGA_tensor_t {tinfo.name};\n"

        fp_shift = 0
        if GIGA_Code_Generator.is_fixed(tinfo.giga_type):
            fp_shift = self.get_fp_shift(tensor)
        
        shape = tensor.shape

        total_size = self.__class__.get_nb_bytes_from_c_type(tinfo.c_type)
        if tinfo.name not in self.kernels:
            for dim in range(tinfo.nb_dims):
                total_size *= tensor.shape[dim]
        else:  # Kernels always have (Co, Ci, 3, 3) dimensions
            shape = [tensor.shape[0], tensor.shape[1],3,3]

        self.allocate_tensors_string += f'\n    {tinfo.prefix}->{tinfo.name} = ' + self.set_tensor_params(tinfo.giga_type, fp_shift, shape)

        if not is_view:
            memory_zone_id, offset = self.allocator.allocate_tensor(tinfo.name, tensor.shape, tinfo.giga_type)
            self.allocate_tensors_string += ('\n'
                                             f'    GIGA_allocate_t {tinfo.name}_allocate = {{\n'
                                             f'        .memory_zone_id = {memory_zone_id},\n'
                                             f'        .offset = {offset},\n'
                                             f'        }};\n'
                                             f'    if ((error = giga_allocate_tensor( & {tinfo.prefix}->{tinfo.name}, &{tinfo.name}_allocate)) != GIGA_Success)\n'
                                             '        return error;\n\n')
        else:
            self.implicit_tensors.add(tinfo.name)
        self.declared_tensors.add(tinfo.name)

        return total_size

    @staticmethod
    def get_scalar_type(quant_algorithm_vendor: int, quant_algorithm: int,
                        quant_algorithm_parameters: int, nb_bits: int) -> (str, np.dtype):
        """

        :param quant_algorithm_vendor:
        :param quant_algorithm:
        :param quant_algorithm_parameters:
        :param nb_bytes:
        :return:
        """
        if quant_algorithm_vendor != 0:
            print("Tensor quantization algorithm vendor is unknown")
            exit(-1)

        if quant_algorithm == 0:
            if nb_bits == 16:
                return "half", np.float16
            elif nb_bits == 32:
                return "float", np.float32
            else:
                print("Unsupported tensor scalar format")
                exit(-1)

        elif quant_algorithm == 1:
            if nb_bits == 16:
                return "int16_t", np.int16_t
            elif nb_bits == 8:
                return "uint8_t", np.uint8_t
            else:
                print("Unsupported tensor scalar format")
                exit(-1)

        else:  # TODO To be defined during quantization
            print("Quantized values not supported yet")

    def get_data_values(self, path: Path) -> (np.ndarray, str, List):
        """

        :param path:
        :return: array of values
        """
        with open(path, 'rb') as data_file:
            magic_number = data_file.read(2)
            version_major = int.from_bytes(data_file.read(1), byteorder="little")
            version_minor = int.from_bytes(data_file.read(1), byteorder="little")
            data_length = int.from_bytes(data_file.read(4), byteorder="little")  # In bytes
            tensor_rank = int.from_bytes(data_file.read(4), byteorder="little")

            dimensions = []
            for dim in range(tensor_rank):
                dimensions.append(int.from_bytes(data_file.read(4), byteorder="little"))

            data_file.read((8 - tensor_rank) * 4)  # read the remaining bytes

            bits_per_item = int.from_bytes(data_file.read(4), byteorder="little")
            quant_algorithm_vendor = int.from_bytes(data_file.read(2), byteorder="little")
            quant_algorithm = int.from_bytes(data_file.read(2), byteorder="little")
            quant_algorithm_parameters = int.from_bytes(data_file.read(32), byteorder="little")

            scalar_text, scalar_np_type = self.get_scalar_type(quant_algorithm_vendor, quant_algorithm, quant_algorithm_parameters, bits_per_item)

            data_file.seek(128, 0)  # Go to the beginning of actual data

            nb_values = int(data_length / (bits_per_item / 8))

            output = np.frombuffer(data_file.read(data_length * int(bits_per_item / 8)), dtype=scalar_np_type)

            return output, scalar_text, dimensions

    def declare_fill(self, tensor: nnef.Tensor) -> None:
        """

        :param tensor:
        :return:
        """
        tensor_name = tensor.name

        self.declare_tensor(tensor)

        values, scalar_text, dimensions = self.get_data_values(self.dir_path / (tensor.name + ".dat"))

        if tensor_name in self.kernels:
            shaped_values = values.reshape(dimensions)
            values = np.zeros((dimensions[0], dimensions[1], 3, 3))
            if dimensions[2:4] == [1, 1]:
                values[:, :, 1:2, 1:2] = shaped_values

            elif dimensions[2:4] == [2, 1]:
                values[:, :, 0:2, 1:2] = shaped_values

            elif dimensions[2:4] == [1, 2]:
                values[:, :, 1/2, 0:2] = shaped_values

            elif dimensions[2:4] == [2, 2]:
                values[:, :, 0:2, 0:2] = shaped_values

            else:
                values = shaped_values

            values = values.flatten()

        output = f"\n    {scalar_text} data_{tensor_name}[] = {{"
        nb_elements_in_row = 0;
        for i in range(len(values)):
            nb_elements_in_row += 1;
            if i > 0:
                output += ","
                if nb_elements_in_row >= 16:
                    output += "\n"
                    nb_elements_in_row = 0;
                
            output += f"{values[i]:.7f}f"
                
        output += "    };\n"
                
        giga_type = self.get_GIGA_type(scalar_text)

        output += (f"    if ((error = giga_copy_to_tensor(data_{tensor_name}, {giga_type}, 0, &tensors->{tensor_name})) != GIGA_Success)\n"
                    "        return error;\n")

        self.fill_string += output

    def declare_conv(self, conv_operation: nnef.Operation, with_relu: bool, index) -> None:
        """

        :param: conv_operation:
        :param: with_relu:
        :param: index:
        :return:
        """

        operation_name = f"op_{self.op_index}"
        kernel_name = conv_operation.inputs['filter']
        kernel_tensor = self.graph.tensors[kernel_name]
        kernel_shape = kernel_tensor.shape
        self.kernels.add(kernel_name)
        bias_name = conv_operation.inputs['bias']
        self.biases.add(bias_name)
        stride_ud = conv_operation.attribs['stride'][0]
        stride_lr = conv_operation.attribs['stride'][1]
        padding = conv_operation.attribs['padding']
        padding[0] = list(padding[0])
        padding[1] = list(padding[1])
        with_relu = str(with_relu).lower()
        input_name = conv_operation.inputs['input']
        output_name = conv_operation.outputs['output']

        # Padding surgery for smaller kernels
        if kernel_shape[2] in [1, 2]:
            padding[0][1] += 1
        if kernel_shape[3] in [1, 2]:
            padding[1][1] += 1

        if kernel_shape[2] == 1:
            padding[0][0] += 1
        if kernel_shape[3] == 1:
            padding[1][0] += 1

        if padding[0][0] > 2 or padding[0][1] > 2 or padding[1][0] > 2 or padding[1][1] > 2:
            print("Warning : Padding is higher than 2 !")

        if input_name not in self.declared_tensors:
            self.declare_tensor(self.graph.tensors[input_name])
        if output_name not in self.declared_tensors:
            self.declare_tensor(self.graph.tensors[output_name])

        self.op_structure_string += f"    GIGA_conv2d_t {operation_name}_params;\n"
        self.set_operations_string += ( '\n'
                                       f'    ops_params->{operation_name}_params = (GIGA_conv2d_t){{\n'
                                       f'        .padding = {{ {{ {padding[0][0]}, {padding[0][1]} }}, {{ {padding[1][0]}, {padding[1][1]} }} }},\n'
                                       f'        .stride = {{ {stride_ud}, {stride_lr} }},\n'
                                        '        .dilation = { 1, 1 },\n'
                                       f'        .b_ReLU = {with_relu},\n'
                                       f'        .kernel = &tensors->{kernel_name},\n'
                                       f'        .bias = &tensors->{bias_name}\n'
                                        '        };\n')

        prefix_i = "tensors"
        if input_name in self.graph.inputs or input_name in self.graph.outputs:
            prefix_i = "io"

        prefix_o = "tensors"
        if output_name in self.graph.inputs or output_name in self.graph.outputs:
            prefix_o = "io"

        self.process_list[index] = '    /* Convolution */\n'
        if self.verbose_code:
            self.process_list[index] += f'    printf("{operation_name}\\n");'
        self.process_list[index] += (f'    if((error = giga_conv2d(&ops_params->{operation_name}_params, &{prefix_i}->{input_name}, &{prefix_o}->{output_name})) != GIGA_Success)\n'
                                      '        return error;\n')

    def get_full_c_file_string(self):
        return self.header_string + "\n" \
               + self.initialize_string + "\n" \
               + self.allocate_tensors_string + "\n" \
               + self.fill_string + "\n" \
               + self.set_operations_string + "\n" \
               + self.process_string + "\n"

    def get_full_h_file_string(self):

        include_gard_var = self.network_name.upper() + "_H"
        ret_string = f"#ifndef {include_gard_var}\n" + \
                    f"#define {include_gard_var}\n" + \
                    self.header_file + "\n" \
                    + self.parameters_structure_string + "\n" \
                    + self.op_structure_string + "\n" \
                    + self.io_structure_string + "\n" \
                    + self.prototypes_string + "\n" \
                    + f"#endif //{include_gard_var}"
        return ret_string

    def get_fp_shift(self, tensor):
        tinfo = self.get_tensor_info(tensor)
        
        if not self.is_fixed(tinfo.giga_type):
            return 0;
        
        if not tinfo.is_kernel:
            if tinfo.name in self.graph.inputs:
                return self.input_fp_shift;
            if tinfo.name in self.graph.outputs:
                return self.output_fp_shift;
            return self.intermediate_fp_shift;

        values, scalar_text, dimensions = self.get_data_values(self.dir_path / (tensor.name + ".dat"))
        values = values.flatten()
        
        bits = self.get_type_size_in_bits(tinfo.giga_type)
        data_range = int(np.ceil(np.log2(np.max(abs(values)))))
        if np.min(values) < 0:
            data_range += 1
        return bits - data_range


if __name__ == "__main__":
    parser = ArgumentParser(prog='nnef_to_giga',
                            description='Translation from NNEF files to C code for calling the Giga API.')

    parser.add_argument('-i', '--input', required=True,
                        help="Directory containing the graph.nnef file and associated .dat weight files")
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
    
    memory_zone_size = 0
    if args.memory_zone_size is not None:
        if type(args.memory_zone_size) == str:
            if args.memory_zone_size[-1] >= '0' and args.memory_zone_size[-1] <= '9':
                memory_zone_size = int(args.memory_zone_size)
            else:
                memory_zone_size = int(args.memory_zone_size[:-1])
            if args.memory_zone_size[-1] == 'G':
                memory_zone_size *= 1024 * 1024 * 1024
            elif args.memory_zone_size[-1] == 'M':
                memory_zone_size *= 1024 * 1024
            elif args.memory_zone_size[-1] == 'K':
                memory_zone_size *= 1024

    generator = GIGA_Code_Generator(Path(args.input), verbose_code=False,
                                    memory_zone_size=memory_zone_size,
                                    input_type=args.input_type,
                                    input_fp_shift=args.input_fp_shift,
                                    output_type=args.output_type,
                                    output_fp_shift=args.output_fp_shift,
                                    intermediate_type=args.intermediate_type,
                                    intermediate_fp_shift=args.intermediate_fp_shift,
                                    kernel_type=args.kernel_type,
                                    memory_allocator=args.allocator)

    os.makedirs(args.output, exist_ok=True)

    with open(args.output + "/" + generator.network_name + ".c", 'w') as output_file:
        output_file.write(generator.get_full_c_file_string())

    with open(args.output + "/" + generator.network_name + ".h", 'w') as output_file:
        output_file.write(generator.get_full_h_file_string())
    
    def format_RAM_size(mem_size):
        if mem_size > 1024**3:
            mem_size = f'{(mem_size / 1024**3):.3f}GB'
        elif mem_size > 1024**2:
            mem_size = f'{(mem_size / 1024**2):.3f}MB'
        elif mem_size > 1024**1:
            mem_size = f'{(mem_size / 1024**1):.3f}KB'
        else:
            mem_size = mem_size + "B"
        return mem_size
        
    mem_used = format_RAM_size(generator.allocator.memory_used())
    mem_needed = format_RAM_size(generator.allocator.memory_needed())
    print(f'{mem_used} RAM used')
    print(f'{mem_needed} RAM needed')
