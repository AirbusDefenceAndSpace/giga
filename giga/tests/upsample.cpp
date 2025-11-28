/*!
 * (C) 2025 Airbus copyright all rights reserved
 * \date 16/01/2025
 */

#include <giga/giga.h>
#include "utils.h"

GIGA_error upsample_test(GIGA_data_type GT)
{
    ScopedMessage msg;
    msg << "Upsample " << giga_data_type_str(GT) << "\n";

    GIGA_error error;
    uint32_t device_id = giga_get_default_device_id(&error);

    if(error != GIGA_Success)
    {
        std::cerr << "Error getting default device id" << std::endl;
        return error;
    }

    error = giga_initialize_device(device_id);
    if(error != GIGA_Success)
    {
        std::cerr << "Error initializing device" << std::endl;
        return error;
    }

    size_t offset = 0;

    GIGA_tensor_t tensor;
    tensor.nb_dims = 3;
    tensor.dims[0] = 2;
    tensor.dims[1] = 5;
    tensor.dims[2] = 5;
    tensor.device_id = device_id;
    tensor.type = GT;
    tensor.fp_shift = 0;

    GIGA_allocate_t tensor_params;
    tensor_params.memory_zone_id = 0;
    tensor_params.offset = offset;
    offset += align_address(tensor_size_in_bytes(&tensor), element_size_in_bits(&tensor) / 8);
    error = giga_allocate_tensor(&tensor, &tensor_params);
    if(error != GIGA_Success)
    {
        std::cerr << "Error allocating tensor tensor" << std::endl;
        return error;
    }

    const float data[50] = {1, 2, 3, 4, 5,
                      -1, -2, -3, -4, -5,
                      1, 2, 3, 4, 5,
                      -1, -2, -3, -4, -5,
                      1, 2, 3, 4, 5,
                      -1, -2, -3, -4, -5,
                      1, 2, 3, 4, 5,
                      -1, -2, -3, -4, -5,
                      1, 2, 3, 4, 5,
                      -1, -2, -3, -4, -5};

    fill_4d_tensor(data, tensor);
    print_tensor(msg, tensor,"giga_upsample input");

    GIGA_tensor_t upsampled;
    upsampled.nb_dims = 3;
    upsampled.dims[0] = 2;
    upsampled.dims[1] = 10;
    upsampled.dims[2] = 10;
    upsampled.device_id = device_id;
    upsampled.type = GT;
    upsampled.fp_shift = 0;

    GIGA_allocate_t upsampled_params;
    upsampled_params.memory_zone_id = 0;
    upsampled_params.offset = offset;
    offset += align_address(tensor_size_in_bytes(&upsampled), element_size_in_bits(&upsampled) / 8);

    error = giga_allocate_tensor(&upsampled, &upsampled_params);
    if(error != GIGA_Success)
    {
        std::cerr << "Error allocating tensor upsampled" << std::endl;
        return error;
    }

    // Fill output with garbage to make sure we don't test an unwritten tensor
    fill_contiguous_tensor_with_random_data(upsampled, 0.f, 255.f);

    GIGA_upsample_t upsample_params;
    upsample_params.factor = 2;

    error = giga_upsample(&upsample_params, &tensor, &upsampled);
    if(error != GIGA_Success)
    {
        if (error == GIGA_Unimplemented_Type)
        {
            std::cout << "Type not implemented!" << std::endl;
            msg.clear();
            return GIGA_Success;
        }
        std::cerr << "Error performing giga_upsample" << std::endl;
        return error;
    }

    print_tensor(msg, upsampled, "giga_upsample output");

    GIGA_tensor_t result;
    result.nb_dims = 3;
    result.dims[0] = 2;
    result.dims[1] = 10;
    result.dims[2] = 10;
    result.device_id = device_id;
    result.type = GT;
    result.fp_shift = 0;

    GIGA_allocate_t result_params;
    result_params.memory_zone_id = 0;
    result_params.offset = offset;
    offset += align_address(tensor_size_in_bytes(&result), element_size_in_bits(&result) / 8);
    error = giga_allocate_tensor(&result, &result_params);
    if(error != GIGA_Success)
    {
        std::cerr << "Error allocating tensor result" << std::endl;
        return error;
    }

    const float data_result[200] = {1, 1, 2, 2, 3, 3, 4, 4, 5, 5,
                              1, 1, 2, 2, 3, 3, 4, 4, 5, 5,
                              -1, -1, -2, -2, -3, -3, -4, -4, -5, -5,
                              -1, -1, -2, -2, -3, -3, -4, -4, -5, -5,
                              1, 1, 2, 2, 3, 3, 4, 4, 5, 5,
                              1, 1, 2, 2, 3, 3, 4, 4, 5, 5,
                              -1, -1, -2, -2, -3, -3, -4, -4, -5, -5,
                              -1, -1, -2, -2, -3, -3, -4, -4, -5, -5,
                              1, 1, 2, 2, 3, 3, 4, 4, 5, 5,
                              1, 1, 2, 2, 3, 3, 4, 4, 5, 5,
                              -1, -1, -2, -2, -3, -3, -4, -4, -5, -5,
                              -1, -1, -2, -2, -3, -3, -4, -4, -5, -5,
                              1, 1, 2, 2, 3, 3, 4, 4, 5, 5,
                              1, 1, 2, 2, 3, 3, 4, 4, 5, 5,
                              -1, -1, -2, -2, -3, -3, -4, -4, -5, -5,
                              -1, -1, -2, -2, -3, -3, -4, -4, -5, -5,
                              1, 1, 2, 2, 3, 3, 4, 4, 5, 5,
                              1, 1, 2, 2, 3, 3, 4, 4, 5, 5,
                              -1, -1, -2, -2, -3, -3, -4, -4, -5, -5,
                              -1, -1, -2, -2, -3, -3, -4, -4, -5, -5,
                             };

    fill_4d_tensor(data_result, result);

    if(!compare_tensors(&upsampled, &result))
    {
        std::cerr << "Error comparing tensors upsampled and result" << std::endl;
        return GIGA_Unknown_Error;
    }

    error = giga_release_tensor(&upsampled);
    if(error != GIGA_Success)
    {
        std::cerr << "Error releasing tensor upsampled" << std::endl;
        return error;
    }

    error = giga_release_tensor(&tensor);
    if(error != GIGA_Success)
    {
        std::cerr << "Error releasing tensor tensor" << std::endl;
        return error;
    }

    error = giga_release_tensor(&result);
    if(error != GIGA_Success)
    {
        std::cerr << "Error releasing tensor result" << std::endl;
        return error;
    }

    const std::string line = msg.message();
    msg.replaceMessage(line.substr(0, line.find('\n')));

    return GIGA_Success;
}

int main()
{
    GIGA_error error = GIGA_Success;

#define EARLY_ABORT() throw std::runtime_error("Error")

    try
    {
        if((error = upsample_test(GIGA_Float32)) != GIGA_Success)
            EARLY_ABORT();
        if((error = upsample_test(GIGA_Float16)) != GIGA_Success)
            EARLY_ABORT();
        if((error = upsample_test(GIGA_SFixed8)) != GIGA_Success)
            EARLY_ABORT();
        if((error = upsample_test(GIGA_SFixed16)) != GIGA_Success)
            EARLY_ABORT();
        if((error = upsample_test(GIGA_UFixed8)) != GIGA_Success)
            EARLY_ABORT();
        if((error = upsample_test(GIGA_UFixed16)) != GIGA_Success)
            EARLY_ABORT();
    }
    catch(const std::exception &e)
    {
        if (error != GIGA_Success)
            std::cerr << "Error: " << giga_str_error(error) << std::endl;
        else
        {
            std::cerr << "Exception caught: " << e.what() << std::endl;
            error = GIGA_Unknown_Error;
        }
    }

    return error;
}
