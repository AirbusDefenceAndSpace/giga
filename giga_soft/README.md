# Generic Interface for Generic Accelerators {#mainpage}

The GIGA API is a C language interface for porting neural networks to generic accelerators such as FPGAs or multi-core processors. Its main purpose is to facilitate the access to
accelerated 3x3 2d convolution as it is the main building block of a large majority of embeddable neural networks for image processing and computer vision. 2d convolution is also
at the heart of simple linear filters such as Sobel edge detection, etc...

This API is strictly for inference. No training can be done with it.

## Features

### Tensors

The basic data manipulation unit of the API is the tensors. Tensors are used to store both data and variables necessary for the processing such as kernels for the convolution.
Data can be written and read from the tensors using the mapping feature of the API (see [Memory Management](#memory-management))

### Data types

The API is able to work with different datatypes :
 - IEEE 32-bit floating point
 - IEEE 16-bit floating point
 - 8 bit signed fixed point with variable fractional bytes
 - 16 bit signed fixed point with variable fractional bytes
 - 8 bit unsigned fixed point with variable fractional bytes 
 - 16 bit unsigned fixed point with variable fractional bytes
 
A tensor can only be of one data type. Operations mixing fixed and floating points aren't authorized in the API.

#### Fixed point representation

Fixed point numbers are actually integers with a fixed number of fractional bytes. The number represented is as such :

$$\ x_{float} = x_{int} 2^{-fp_shift}

A tensor with fixed point data has the same number of fractional bytes defined in the fp_shift member of the tensor structure. This representation includes integers when the number
of fractional bytes is set to zero.

### Processing

Processing by the API is simply a sequence of function calls, each of them parametrized by relevant parameters.

### Supported Operations

The GIGA API, being primarily a convolution acceleration API does not support all typical operations associated. Here's a list of natively supported operations.

#### 2d Convolution 

2d convolution is supported with the following parameters :

- Kernel size : 3x3 only.
- Stride : 1 or 2.
- Padding : 0, 1 or 2 with zeros. Assymetric padding is possible.

By changing parameters, it's possible to mimic other kernel sizes. For example, a 2x2 kernel with 1 padding can be done with a 3x3 kernel filled with zeros on the last row and column
as well as changing the left and bottom padding to 2. A ReLU activation function can be applied at the end of the convolution. The kernel must use a signed data type.

### Dense layers

Dense layers (also known as linear layers) are supported. The input and output tensors must be one or two dimensional with two dimensional tensors having the batch dimension as the
first dimension. As with convolution, a ReLU activation function can be applied to the result of the dense layer. The kernel must use a signed data type.

### Concatenation

Concatenation is supported through the use of views. In that context, a tensor can only be concatenated once. Concatenation requiring copy is not supported natively.

### Upsampling

Nearest Neighbour upsampling is supported. Only upsampling by a factor of 2 is supported.

### Softmax

Softmaxing is supported in two cases. One related to image classification is a 1d softmax. The other case is related to segmentation and is a 2d softmax along the channels dimension.
This operation is not available for fixed point data types.

### Reshaping

### Addition

Two tensors can be added as long as they have exactly the same number of dimensions and the same shape.

### Other operations

Many common operations can be implemented using the previous basic operations. The creative use of convolution allows for the implementation of many accelerated operations :

 - **Batch Normalization** is a very common operation in convolutional neural networks. It can be implemented in the GIGA API using a combination of convolutions and views.

 - **2d Average Pooling** can be implemented using convolution (possibly with a smaller kernel trick) and views.

 - **Bilinear Upsampling** can be very closely approximated using nearest neighbours upsampling and a 2d Convolution with an appropriate kernel.


### Memory management {#memory-management}

Since the API is designed for embedded application, memory management is a crucial topic. The philosophy of the API is to put the user in charge of the memory layout. Each implementation
is however responsible for giving guidelines and constraints as to where each of the tensors should be allocated depending on their usage. Accelerators may have multiple types of memory
depending on their use. It is the responsibility of the API backend implementation to provide specifications of the available types of memory and the size of each of them. Each backend
implementation is also responsible for specifying the type of memory each tensor should be allocated in depending on the operations performed on them.
Tensors can share memory when the user deems it appropriate in order to save on memory imprint. Views implicitely also declare tensors sharing the same memory as other tensors. It is
strongly encouraged to create the memory layout of all tensors before starting processing. The implementation of a Neural Network will therefore often be dependent on the backend
implementation when it comes to allocation.

Memory with always be owned by the API backend. In order to access the tensors' memory to write and read from it, the API provides a mapping feature.

### Asynchronous processing

The API offers the possibility to have all the processing done in an asynchronous manner. A callback and wait-for-completion mechanism allows resynchronisation after processing.
This is particularly useful for very multi-threaded environements.


## Baseline CPU implementation

Along with the API definition. A baseline CPU implementation of the entire API in a synchronous mode is [available](#cpu_impl_doc)

## Optimized CPU backend

This CPU backend can be built either as a reference implementation (optimizations disabled, no multithreading) or as an optimized CPU backend. The optimized build relies on OpenMP for
multithreading. If also focuses on common use cases, in particular, it does not build all combinations of input types but is restricted to same input types with exceptions when it makes
sense.
