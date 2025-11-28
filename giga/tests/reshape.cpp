/*!
 * (C) 2025 Airbus copyright all rights reserved
 * \date 16/01/2025
 */

#include <giga/giga.h>

#include "utils.h"

GIGA_error reshape_test(GIGA_data_type GT)
{
    ScopedMessage msg;
    msg << "Reshape " << giga_data_type_str(GT) << "\n";

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
    tensor.data = NULL;
    tensor.fp_shift = 0;

    GIGA_allocate_t tensor_params;
    tensor_params.memory_zone_id = 0;
    tensor_params.offset = offset;
    offset += tensor_size_in_bytes(&tensor);
    error = giga_allocate_tensor(&tensor, &tensor_params);
    if(error != GIGA_Success)
    {
        std::cerr << "Error allocating tensor tensor" << std::endl;
        return error;
    }

    float data[50] = {1, 2, 3, 4, 5,
                      1, 2, 3, 4, 5,
                      1, 2, 3, 4, 5,
                      1, 2, 3, 4, 5,
                      1, 2, 3, 4, 5,
                      1, 2, 3, 4, 5,
                      1, 2, 3, 4, 5,
                      1, 2, 3, 4, 5,
                      1, 2, 3, 4, 5,
                      1, 2, 3, 4, 5};

    fill_4d_tensor(data, tensor);

    GIGA_tensor_t reshaped;
    reshaped.nb_dims = 2;
    reshaped.dims[0] = 2;
    reshaped.dims[1] = 25;
    reshaped.device_id = device_id;
    reshaped.type = GT;
    reshaped.device_id = device_id;
    reshaped.data = NULL;
    reshaped.fp_shift = 0;

    GIGA_allocate_t reshaped_params;
    reshaped_params.memory_zone_id = 0;
    reshaped_params.offset = 0;
    error = giga_allocate_tensor(&reshaped, &reshaped_params);
    if(error != GIGA_Success)
    {
        std::cerr << "Error allocating tensor reshaped" << std::endl;
        return error;
    }

    GIGA_reshape_t reshape_params;
    error = giga_reshape(&reshape_params, &tensor, &reshaped);
    if(error != GIGA_Success)
    {
        std::cerr << "Error performing giga_reshape" << std::endl;
        return error;
    }

    error = giga_release_tensor(&tensor);
    if(error != GIGA_Success)
    {
        std::cerr << "Error releasing tensor tensor" << std::endl;
        return error;
    }

    error = giga_release_tensor(&reshaped);
    if(error != GIGA_Success)
    {
        std::cerr << "Error releasing tensor reshaped" << std::endl;
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
        if((error = reshape_test(GIGA_Float32)) != GIGA_Success)
            EARLY_ABORT();
        if((error = reshape_test(GIGA_Float16)) != GIGA_Success)
            EARLY_ABORT();
        if((error = reshape_test(GIGA_SFixed8)) != GIGA_Success)
            EARLY_ABORT();
        if((error = reshape_test(GIGA_SFixed16)) != GIGA_Success)
            EARLY_ABORT();
        if((error = reshape_test(GIGA_UFixed8)) != GIGA_Success)
            EARLY_ABORT();
        if((error = reshape_test(GIGA_UFixed16)) != GIGA_Success)
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
