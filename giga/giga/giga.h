/*!
 * (C) 2025 Airbus copyright all rights reserved
 *
 */
#ifndef GIGA_H_
#define GIGA_H_

/*!
 * \file
 * \author Roland Brochard (roland.brochard@airbus.com)
 * \author Lucas Marti (lucas.marti@airbus.com)
 * \date 3/04/2023
 *
 * \name GIGA = Generic Interface Generic Accelerator
 * \version 1.0.0
 *
 */

#ifdef _WIN32
#ifdef GIGA_API_EXPORT
#define GIGA_API __declspec(dllexport)
#else
#define GIGA_API __declspec(dllimport)
#endif
#else
#define GIGA_API
#endif


#ifdef __cplusplus
extern "C"
{
#endif

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

/*!
 * \defgroup GIGA GIGA API interface definition
 * @{
 */

/*!
 * \brief Enum containing the errors the API can return.
 *
 * This enum contains all the errors that the API can return when calling one of its functions.
 *
 */
GIGA_API typedef enum GIGA_error
{
    GIGA_Success                            = 0x0000, //!< No error
    GIGA_Unknown_Error                      = 0x0001, //!< Unknown Error
    GIGA_Incorrect_Parameter                = 0x0002, //!< A parameter is not valid in the context of the function
    GIGA_Out_Of_Host_Memory                 = 0x0003, //!< The host is out of memory
    GIGA_Out_Of_Device_Memory               = 0x0004, //!< The device (accelerator) is out of memory
    GIGA_Inconsistent_Tensor_Sizes          = 0x0005, //!< The tensors' sizes are not compatible for this operation
    GIGA_Inconsistent_Number_Of_Dimensions  = 0x0006, //!< The tensors' (including the possible parameters) number of dimension are not compatible for this operation
    GIGA_Unimplemented_Type                 = 0x0007, //!< This type is not a valid data type for a tensor.
    GIGA_Unknown_tensor                     = 0x0008, //!< This tensor has not been declared to the API
    GIGA_Inconsistent_Tensor_Types          = 0x0009, //!< The data types for the tensors are not compatible
    GIGA_Bad_Alloc                          = 0x000A, //!< Bad allocation
    GIGA_Device_Not_Initialized             = 0x000B, //!< The requested device is not initialized
    GIGA_Bad_Memory_Alignment               = 0x000C, //!< The request is not compatible with the memory alignment
    GIGA_Not_Implemented                    = 0x000D, //!< The request service or configuration is not implemented
    GIGA_Device_Error                       = 0x000E, //!< Error in the device implementation
    GIGA_Inconsistent_Device                = 0x000F, //!< The given device identifier is/are inconsistent
    GIGA_Process_Mapped_Tensor              = 0x0010, //!< Cannot process a mapped tensor
    GIGA_Memory_Alignement_Error            = 0x0011, //!< Memory is not aligned as expected by the backend
    GIGA_Memory_Layout_Error                = 0x0012  //!< Memory isn't laid out in accordance to backend specification

} GIGA_error;


/*! \brief Get the string from the GIGA error code
 *
 * @param err GIGA error code
 * @return A constant string describing the given GIGA error code
 */
GIGA_API const char *giga_str_error(GIGA_error err);


/* Device initialization */

/*! \brief Get the id of the default device
 *
 *  \param[out] err A pointer to an error variable for output
 *  \return The id of the device
 */
GIGA_API uint32_t giga_get_default_device_id(GIGA_error *err);

/*! \brief List the available devices
 *
 *  \param[out]    device_ids A pointer to a table that will contain the device IDs
 *  \param[in,out] nb_devices A pointer to a variable that contains the maximum
 *                            number of device ids storable in device_ids and after
 *                            the call, the number of devices ids stored  in device_ids
 *  \return Error
 */
GIGA_API GIGA_error giga_list_devices(uint32_t *device_ids, uint32_t *nb_devices);

/*! \brief Initialize the selected device
 *
 *  \param[in] device_id
 *  \return Error
 */
GIGA_API GIGA_error giga_initialize_device(uint32_t device_id);

/*! \brief Data types possible for tensors' data.
 */
GIGA_API typedef enum GIGA_data_type
{
    GIGA_Float16    = 0x00, //!< IEEE 16-bit floating point
    GIGA_Float32    = 0x01, //!< IEEE 32-bit floating point

    GIGA_SFixed4 = 0x02,    //!< 4-bit signed fixed point, fractional bits definied in tensor metadata
    GIGA_SFixed8 = 0x03,    //!< 8-bit signed fixed point, fractional bits definied in tensor metadata
    GIGA_SFixed16 = 0x04,   //!< 16-bit signed fixed point, fractional bits definied in tensor metadata

    GIGA_UFixed4 = 0x05,    //!< 4-bit unsigned fixed point, fractional bits definied in tensor metadata
    GIGA_UFixed8 = 0x06,    //!< 8-bit unsigned fixed point, fractional bits definied in tensor metadata
    GIGA_UFixed16 = 0x07,   //!< 16-bit unsigned fixed point, fractional bits definied in tensor metadata

} GIGA_data_type;



#define giga_type2str_case(type) \
case type : \
return  #type ;\

/*! \brief Get the string from the GIGA type
 *
 * @param type GIGA type code
 * @return The name of the given GIGA type
 */
GIGA_API inline const char * giga_data_type_str(GIGA_data_type data_type)
{
    switch(data_type)
    {
    giga_type2str_case(GIGA_Float16)
    giga_type2str_case(GIGA_Float32)
    giga_type2str_case(GIGA_SFixed4)
    giga_type2str_case(GIGA_SFixed8)
    giga_type2str_case(GIGA_SFixed16)
    giga_type2str_case(GIGA_UFixed4)
    giga_type2str_case(GIGA_UFixed8)
    giga_type2str_case(GIGA_UFixed16)
    default:
        return "Undefined Type";
    }
}

/*! \brief Tensor
 */
GIGA_API typedef struct GIGA_tensor_t
{
    uint32_t device_id;     //!< Device on which the tensor is destined to be stored.
    uint32_t nb_dims;       //!< Number of dimensions of the tensor (between 1 and 4)
    GIGA_data_type type;    //!< Data type of the tensor.
    uint32_t dims[4];       //!< Value of the dimensions of the tensors, valid up to nb_dims. Typically (B, C, H, W)
    uint32_t strides[4];    //!< Value of the strides (Number of bytes between each dimension slice) of the tensors, valid up to nb_dims. Typically (B, C, H, W)
    uint8_t fp_shift;       //!< Bit shift for fixed point representation (fractional bytes)

    void *data; //!< A pointer for the API to store internal data.
} GIGA_tensor_t;

/* Memory management functions */

/*! \brief Flag values for mapping and unmapping the tensors to either read or write values from it.
 */
GIGA_API typedef enum GIGA_memory_flag
{
    GIGA_Memory_Discard = 0x0, //!< During mapping
    GIGA_Memory_Sync    = 0x1,
} GIGA_memory_flag;

/*! \brief Parameters from allocating a new \link GIGA_tensor_t \endlink
 */
GIGA_API typedef struct GIGA_allocate_t
{
    uint32_t memory_zone_id;    //!< The id of the memory zone in which the tensor must be allocated
    uint32_t offset;            //!< The offset from the start of the memory zone
} GIGA_allocate_t;

/*! \brief Allocates a new \link GIGA_tensor_t \endlink
 *
 * This function allocates a tensor described by the tensor parameter. The number of dimensions, the device id, the data type and the dimensions must be specified.
 * The strides are filled by the API. Overlapping tensors are allowed, as they can be used to implement implicit concatenation.
 * \param tensor A pointer to the tensor to be allocated
 * \param[in] params A pointer to the allocation parameters
 *
 * \return Error
 */
#define giga_allocate_tensor(tensor, params) giga_allocate_tensor_(tensor,params,__FILE__,__LINE__)
GIGA_API GIGA_error giga_allocate_tensor_(GIGA_tensor_t *tensor, const GIGA_allocate_t *params, const char *file, int line);

/*! \brief Maps a tensor for writing or reading its data.
 *
 * This function asks for a pointer to the actual data of the tensor in order to write it or read it. Depending on the flags, the API knows if the memory has changed.
 * Mapping/Unmapping operations act as synchronisation points. The GIGA operations declared aboved are guaranteed to have taken place when the function returns
 *
 * \param tensor A pointer to the tensor to be mapped
 * \param[out] ptr A pointer to a pointer that will contain the address of the pointer to the data.
 * \param[in] flags \link GIGA_memory_flag \endlink flags to indicate the purpose of the mapping.
 *
 * \return Error
 */
#define giga_map_tensor(tensor, ptr, flags) giga_map_tensor_(tensor, ptr, flags, __FILE__ , __LINE__)
GIGA_API GIGA_error giga_map_tensor_(GIGA_tensor_t *tensor, void **ptr, GIGA_memory_flag flags, const char *file, int line);

/*! \brief Unmaps a tensor.
 *
 * This function indicates to the API that the tensor doesn't need to be mapped anymore. Mapping/Unmapping operations act as synchronisation points.
 * The GIGA operations declared above are guaranteed to have taken place when the function returns.
 *
 * \param tensor A pointer to the tensor to be unmapped
 * \param[out] ptr A pointer to the data.
 * \param[in] flags \link GIGA_memory_flag \endlink flags to indicate what to do with the memory
 *
 * \return Error
 */
#define giga_unmap_tensor(tensor, ptr, flags) giga_unmap_tensor_(tensor, ptr, flags, __FILE__ , __LINE__)
GIGA_API GIGA_error giga_unmap_tensor_(GIGA_tensor_t *tensor, void *ptr, GIGA_memory_flag flags, const char *file, int line);

/*! \brief Releases the memory of a tensor.
 *
 * This function indicates to the API that the memory used by the tensor will no longer be used by the API client.
 *
 * \param tensor A pointer to the tensor to released.
 *
 * \return Error
 */
#define giga_release_tensor(tensor) giga_release_tensor_(tensor,__FILE__,__LINE__)
GIGA_API GIGA_error giga_release_tensor_(GIGA_tensor_t *tensor, const char *file, int line);

/* Operations */

/*! \brief Parameters for the 3x3 2-d convolution of two \link GIGA_tensor_t \endlink.
 */
GIGA_API typedef struct GIGA_conv2d_t
{
    int32_t padding[2][2];          //!< The padding on each side of the tensor in the H, W dimensions (0, 1 or 2)
    uint32_t stride[2];             //!< The convolution stride in dimensions H, W (1 or 2)
    uint32_t dilation[2];           //!< The dilation in H, W (only 1 is allowed)
    bool b_ReLU;                    //!< If true, a ReLU is applied to the output of the convolution
    const GIGA_tensor_t *kernel;    //!< A pointer to a tensor acting as the kernel. Should be of dimensions (Co, Ci, H=3, W=3).
    const GIGA_tensor_t *bias;      //!< A pointer to a tensor acting as the bias. Should be of dimensions (Co) or (1, Co). If NULL no bias is applied
} GIGA_conv2d_t;

/*! \brief Performs the 3x3 2-d convolution of two \link GIGA_tensor_t \endlink.
 *
 * This function performs the convolution of the tensor using the parameters. The output tensor's dimensions must be consistent with the input dimensions, padding and stride.
 * The tensors must have 2, 3 or 4 dimensions and have the same number of dimensions. The number of input channels of the kernel must be the same as the number of channels in
 * the input tensor. The number of output channels of the kernel must be the same as the number of channels in the output tensor.
 * The value of the batch dimension must be the same between the input and the output.
 * The tensors must be stored in the same device.
 *
 * Tensor format is NCHW (Batch, Channel, Height, Width)
 *
 * \param[in] params A pointer to the 2-D convolution parameters
 * \param[in] in The input tensor
 * \param[out] out The output tensor
 *
 * \return Error
 */
#define giga_conv2d(params,in,out) giga_conv2d_(params,in,out,__FILE__,__LINE__)
GIGA_API GIGA_error giga_conv2d_(const GIGA_conv2d_t *params, const GIGA_tensor_t *in, GIGA_tensor_t *out, const char *file, int line);

/*! \brief Parameters for a dense layer of type A*X + B
 */
GIGA_API typedef struct GIGA_dense_t
{
    bool b_ReLU; //!< If true, a ReLU is applied to the output of the convolution
    const GIGA_tensor_t *kernel; //!< A pointer to a tensor acting as the matrix A. Should be of dimensions (Wo, Wi).
    const GIGA_tensor_t *bias; //!< A pointer to a tensor acting as the bias B. Should be of dimensions (Wo).
} GIGA_dense_t;

/*! \brief Performs the dense operation of type A*X + B
 *
 * This function performs the matrix multiplication of the tensor using the parameters. The output tensor's dimensions must be consistent with the input dimensions and the kernel dimension.
 * The tensors must have 1 or 2 dimensions and have the same number of dimensions. If the tensors are of dimension 2, the first dimension is the batch dimension.
 * The value of the batch dimension must be the same between the input and the output.
 *
 * \param[in] params A pointer to the dense layer parameters
 * \param[in] in The input tensor
 * \param[out] out The output tensor
 *
 * \return Error
 */
#define giga_dense(params, in, out) giga_dense_(params, in, out , __FILE__, __LINE__)
GIGA_API GIGA_error giga_dense_(const GIGA_dense_t *params, const GIGA_tensor_t *in, GIGA_tensor_t *out, const char *file, int line);

/*! \brief Parameters for the reshape operation.
 */
GIGA_API typedef struct GIGA_reshape_t
{
    uint8_t __empty__;
} GIGA_reshape_t;

/*! \brief Changes the interpretation of a \link GIGA_tensor_t \endlink data into another \link GIGA_tensor_t \endlink
 *
 * This function uses data from the in tensor to create a tensor using the same value with a different shape. The types of the tensors must be the same.
 * The total size (product of all dimensions) must be the same. This function is meant to be used in place of a tensor allocation (it's virtually free).
 * In particular the output tensor shall not be pre-allocated. A contiguous tensor can always be reshaped. For non-contiguous tensors, for instance
 * a tensor which is a view of another tensor, restrictions apply: the new geometry cannot bridge the holes in the input. Strides are filled by the backend.
 *
 * Examples of non-contiguous cases:
 * - 2x3x1 tensor with strides 6,2,1 reshaped to 3x2x1 is OK
 * - 2x1x3 tensor with strides 4,4,1 reshaped to 2x3 is OK
 * - 3x4x5 tensor with strides 30,6,1 reshaped to 5x3x4 is NOT OK
 *
 * \param[in] params A pointer to the reshape parameters
 * \param[in] in The input tensor
 * \param[out] out The output tensor
 *
 * \return Error
 */
#define giga_reshape(params,in,out) giga_reshape_(params,in,out,__FILE__,__LINE__)
GIGA_API GIGA_error giga_reshape_(const GIGA_reshape_t * params, const GIGA_tensor_t *in, GIGA_tensor_t *out, const char *file, int line);

/*! \brief Parameters for the softmax operation.
 */
GIGA_API typedef struct GIGA_softmax_t
{
    uint8_t __empty__;
} GIGA_softmax_t;

/*! \brief Performs the softmax operation on a \link GIGA_tensor_t \endlink
 *
 * This function performs the softmax operation along a dimension dependant on the the number of dimensions.
 * The number of dimensions and their values of the input and output tensors must be equal.
 * If the tensor only has one dimensions it's applied along that dimension.
 * If the tensor has two dimensions, the first one is considered to be the batch dimension and softmax is applied along the second dimension.
 * If the tensor has 3 or 4 dimensions, softmaxing is applied along the second one considered the channel dimension.
 *
 * \param[in] params A pointer to the softmax parameters
 * \param[in] in The input tensor
 * \param[out] out The output tensor
 *
 * \return Error
 */
#define giga_softmax(params,in,out) giga_softmax_(params,in,out,__FILE__,__LINE__)
GIGA_API GIGA_error giga_softmax_(const GIGA_softmax_t * params, const GIGA_tensor_t *in, GIGA_tensor_t *out, const char *file, int line); //Along the first non-batch dimension

/*! \brief Parameters for the addition operation of two \link GIGA_tensor_t \endlink.
 */
GIGA_API typedef struct GIGA_add_t
{
    uint8_t __empty__;
} GIGA_add_t;

/*! \brief Performs the addition operation on two \link GIGA_tensor_t \endlink
 *
 * This function performs the addition operation on two tensors. The number of dimensions and their values of the input and output tensors must be equal.
 *
 * \param[in] params A pointer to the addition parameters
 * \param[in] a,b The input tensors
 * \param[out] out The output tensor
 *
 * \return Error
 */
#define giga_add(params,a,b,out) giga_add_(params,a,b,out,__FILE__,__LINE__)
GIGA_API GIGA_error giga_add_(const GIGA_add_t * params, const GIGA_tensor_t *a, const GIGA_tensor_t *b, GIGA_tensor_t *out, const char *file, int line);

/*! \brief Parameters for the nearest neighbour upsampling operation of a \link GIGA_tensor_t \endlink.
 */
GIGA_API typedef struct GIGA_upsample_t
{
    uint32_t factor; //!< The upsampling factor. Must be equal to 2.
} GIGA_upsample_t;

/*! \brief Performs the nearest neighbour upsampling of a \link GIGA_tensor_t \endlink.
 *
 * This function performs the nearest neighbour upsampling of a tensor. The upsampling is always by a factor of two. The input and output tensors must the same number of dimensions.
 * The channel and batch dimension must be the same. The H and W dimensions of the output tensor must be twice those of the input tensors.
 *
 * \param[in] params A pointer to the upsampling parameters
 * \param[in] in The input tensor
 * \param[out] out The output tensor
 *
 * \return Error
 */
#define giga_upsample(params,in,out) giga_upsample_(params,in,out,__FILE__,__LINE__)
GIGA_API GIGA_error giga_upsample_(const GIGA_upsample_t *params, const GIGA_tensor_t *in, GIGA_tensor_t *out, const char *file, int line);

/*! \brief Parameters for the creation of a view of a \link GIGA_tensor_t \endlink.
 */
GIGA_API typedef struct GIGA_view_t
{
    uint32_t offset[4]; //!< Offset in elements along each direction relative to the start of the data of the parent tensor. Dimensions >= tensor nb_dims are ignored.
} GIGA_view_t;

/*! \brief Creates a view of \link GIGA_tensor_t \endlink.
 *
 * This function is one of the three functions that can declare a tensor to the API. It creates a view of an already existing tensor using the parameters.
 * The view starts at the offset specified by the parameters and is of the size given in the output tensor.
 * The number of dimensions, the device id, the data type and the dimensions must be specified. The device id must be the same as the parent tensor.
 * And the dimensions of the output tensor must fit in the parent tensor. The number of dimensions must be the same. To change the number of dimensions,
 * \link giga_reshape \endlink can be used. Strides are filled by the backend.
 *
 * \param[in] params A pointer to the upsampling parameters
 * \param[in] in The input tensor
 * \param[out] out The output tensor
 *
 * \return Error
 */
#define giga_view(params,in,out) giga_view_(params,in,out,__FILE__,__LINE__)
GIGA_API GIGA_error giga_view_(const GIGA_view_t *params, const GIGA_tensor_t *in, GIGA_tensor_t *out, const char *file, int line);

/* Synchronization function */

/*! \brief Sets the callback for the end of the processing.
 *
 * This function sets the callback for when the string of queued up processing ends in the case the accelerator performs the operations asynchronously.
 *
 * \param[in] device_id the device id for which the callback is set
 * \param[in] callback Function pointer to be called upon completion of previous commands
 * \param[in] user_ptr A pointer to data that's passed as a parameter when the function is called (user must make sure this pointer is valid when the callback is called).
 *
 * \return Error
 */
#define giga_callback(device_id, callback, user_ptr) giga_callback_(device_id, callback, user_ptr, __FILE__, __LINE__)
GIGA_API GIGA_error giga_callback_(uint32_t device_id, void (*callback)(void *user_ptr), void *user_ptr, const char *file, int line);

/*! \brief Waits for processing to be completed
 *
 * Waits for the completion of queued up processing in the case the accelerator performs the operations asynchronously. The callback function is called when processing is over.
 *
 * \return Error
 */
GIGA_API GIGA_error giga_wait_for_completion();

/*! \brief Flush commands
 *
 * Flush work queue of given device to make sure processing is started by the backend. This is for asynchronous implementations since in synchronous processing this would do nothing.
 *
 * \return Error
 */
GIGA_API GIGA_error giga_flush(uint32_t device_id);

/*!
 * \brief Register a callback that will be called when an error occurs during asynchronous GIGA function execution.
 *
 * \param[in] callback
 * \param[in] user_ptr A pointer to data that's passed as a parameter when the callback is called.
 * \param[in] err      The error code returned by the GIGA function execution
 * \param[in] file     The name of the source file where the GIGA function that raised the error has been asynchronously called
 * \param[in] line     The line number in file
 */
GIGA_API GIGA_error giga_register_error_callback(void (*callback)(void *user_ptr, GIGA_error err, const char *file, int line), void *user_ptr);

/*!
 * \brief Copy a buffer to a tensor
 * This function is synchronous so you can safely discard the buffer after calling this function.
 * It is intended to initialize kernels or provide input data.
 *
 * Supported input types must at least include GIGA_Float32, GIGA_SFixed8 and GIGA_UFixed8.
 * In case a fixed point data format is used as input, fp_shift must be set to the precision used.
 * If the destination tensor has a different type, a cast is performed.
 *
 * Data is provided in row major format with no holes. Layout follows GIGA conventions:
 * - NCHW for 4D tensors
 * - CHW for 3D tensors
 * - NC for 2D tensors
 * - C for 1D tensors
 *
 * This function takes into account strides so that backend with different memory layout will reorganize data correctly.
 *
 * \param[in] user_ptr      A pointer to data to be copied (with eventual cast) to the destination tensor
 * \param[in] source_type   Enum value corresponding to the data type of elements pointed by user_ptr
 * \param[in] fp_shift      For fixed point data types only, indicates the number of precision bits
 * \param[in] tensor        The destination tensor
 * \param[in] file          The name of the source file where the GIGA function that raised the error has been asynchronously called
 * \param[in] line          The line number in file
 */
#define giga_copy_to_tensor(user_ptr, source_type, fp_shift, tensor) giga_copy_to_tensor_(user_ptr, source_type, fp_shift, tensor, __FILE__, __LINE__)
GIGA_API GIGA_error giga_copy_to_tensor_(const void *user_ptr, GIGA_data_type source_type, uint32_t fp_shift, GIGA_tensor_t *tensor, const char *file, int line);

/*!
 * \brief Copy a tensor to a buffer
 * This function is synchronous.
 * It is intended to read back output data.
 *
 * Supported target types must at least include GIGA_Float32, GIGA_SFixed8 and GIGA_UFixed8.
 * In case a fixed point data format is used as target, fp_shift must be set to the precision used.
 * If the destination buffer has a different type, a cast is performed.
 *
 * Data is written in row major format with no holes. Layout follows GIGA conventions:
 * - NCHW for 4D tensors
 * - CHW for 3D tensors
 * - NC for 2D tensors
 * - C for 1D tensors
 *
 * This function takes into account strides so that backend with different memory layout will reorganize data correctly.
 *
 * \warning You must ensure user_ptr points to allocated memory of the expected size (number of tensor elements x element size)
 *
 * \param[in] user_ptr      A pointer to data to be copied (with eventual cast) to the destination tensor
 * \param[in] target_type   Enum value corresponding to the data type of elements pointed by user_ptr
 * \param[in] fp_shift      For fixed point data types only, indicates the number of precision bits
 * \param[in] tensor        The source tensor
 * \param[in] file          The name of the source file where the GIGA function that raised the error has been asynchronously called
 * \param[in] line          The line number in file
 */
#define giga_copy_from_tensor(user_ptr, target_type, fp_shift, tensor) giga_copy_from_tensor_(user_ptr, target_type, fp_shift, tensor, __FILE__, __LINE__)
GIGA_API GIGA_error giga_copy_from_tensor_(void *user_ptr, GIGA_data_type target_type, uint32_t fp_shift, const GIGA_tensor_t *tensor, const char *file, int line);

/*!
 * @}
 */

#ifdef __cplusplus
}
#endif

#endif  // GIGA_H_
