# NNEF to GIGA

This scripts allows the user to generate a C file that's a direct translation in the GIGA API of a NNEF file.

The implemented operations are :
Convolution
Nearest neighbours upsampling
2x2 Average pooling
2d Batch normalization
Concatenation along the channels axis


## Usage

`NNEF_to_GIGA [-h] -i INPUT -o OUTPUT`

`-h, --help` show this help message and exit

`-i INPUT, --input` INPUT Directory containing the graph.nnef file and associated .dat weight files

`-o OUTPUT, --output` OUTPUT Directory containing the C files after conversion

## Output

The script outputs a C header file containing :
- The declaration of a structure containing all the tensors necessary to the network such as filters and biases
- The declaration of a structure containing all the tensors necessary to the network such as filters and biases
- The declaration of a structure containing all the operation parameters
- The prototypes of the functions in the following C file.

It also outputs a C file containing : 
- A function that initializes the accelerator
- A function that allocates the weight and intermediate tensors
- A function that fills them with the weights
- A function that sets the right parameters for the operations
- A function that sets the right parameters for the input and output tensors
- A function that takes parameters tensors and the network inputs and outputs the results of the processing
