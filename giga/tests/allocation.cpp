/*!
 * (C) 2025 Airbus copyright all rights reserved
 * \date 16/01/2025
 */
#include <giga/giga.h>
#include "utils.h"

GIGA_error allocation_test(GIGA_data_type GT)
{
    ScopedMessage msg;

    msg << "Allocation " << giga_data_type_str(GT) << "\n";

    GIGA_error error;
    uint32_t device_id = giga_get_default_device_id(&error);

    if(error != GIGA_Success)
        return error;

    if((error = giga_initialize_device(device_id)) != GIGA_Success)
       return error;

    size_t offset = 0;

    //Testing 1-dimensional tensor allocation
    {
        ScopedMessage msg("Error allocating 1D tensor");
        GIGA_tensor_t tensor;
        tensor.nb_dims = 1;
        tensor.dims[0] = 5;
        tensor.device_id = device_id;
        tensor.type = GT;

        GIGA_allocate_t tensor_params;
        tensor_params.memory_zone_id = 0;
        tensor_params.offset = offset;
        offset += tensor_size_in_bytes(&tensor);
        if((error = giga_allocate_tensor(&tensor, &tensor_params)) != GIGA_Success)
        {
            std::cerr << "Allocation failed!" << std::endl;
            return error;
        }

        if (tensor.strides[0] != element_size_in_bits(&tensor) / 8)
        {
            std::cerr << "strides[0] is incorrect!" << std::endl;
            return GIGA_Unknown_Error;
        }

        if((error = giga_release_tensor(&tensor)) != GIGA_Success)
        {
            std::cerr << "Releasing tensor failed!" << std::endl;
            return error;
        }

        msg.clear();
    }

    //Testing 2-dimensional tensor allocation
    {
        ScopedMessage msg("Error allocating 2D tensor");
        GIGA_tensor_t tensor;
        tensor.nb_dims = 2;
        tensor.dims[0] = 5;
        tensor.dims[1] = 5;
        tensor.device_id = device_id;
        tensor.type = GT;

        GIGA_allocate_t tensor_params;
        tensor_params.memory_zone_id = 0;
        tensor_params.offset = offset;
        offset += tensor_size_in_bytes(&tensor);
        if((error = giga_allocate_tensor(&tensor, &tensor_params)) != GIGA_Success)
        {
            std::cerr << "Allocation failed!" << std::endl;
            return error;
        }

        if (tensor.strides[0] != element_size_in_bits(&tensor) / 8 * 5)
        {
            std::cerr << "strides[0] is incorrect!" << std::endl;
            return GIGA_Unknown_Error;
        }

        if (tensor.strides[1] != element_size_in_bits(&tensor) / 8)
        {
            std::cerr << "strides[1] is incorrect!" << std::endl;
            return GIGA_Unknown_Error;
        }

        if((error = giga_release_tensor(&tensor)) != GIGA_Success)
        {
            std::cerr << "Releasing tensor failed" << std::endl;
            return error;
        }

        msg.clear();
    }

    //Testing 4-dimensional tensor allocation
    {
        ScopedMessage msg("Error allocating 4D tensor");
        GIGA_tensor_t tensor;
        tensor.nb_dims = 4;
        tensor.dims[0] = 2;
        tensor.dims[1] = 2;
        tensor.dims[2] = 5;
        tensor.dims[3] = 5;
        tensor.device_id = device_id;
        tensor.type = GT;

        GIGA_allocate_t tensor_params;
        tensor_params.memory_zone_id = 0;
        tensor_params.offset = offset;
        offset += tensor_size_in_bytes(&tensor);
        if((error = giga_allocate_tensor(&tensor, &tensor_params)) != GIGA_Success)
        {
            std::cerr << "Allocation failed!" << std::endl;
            return error;
        }

        if (tensor.strides[0] != element_size_in_bits(&tensor) / 8 * 5 * 5 * 2)
        {
            std::cerr << "strides[0] is incorrect!" << std::endl;
            return GIGA_Unknown_Error;
        }

        // In the 4D case, only the batch dimension is expected to be known, the backend being free to permute other dimensions

        if((error = giga_release_tensor(&tensor)) != GIGA_Success)
        {
            std::cerr << "Releasing tensor failed!" << std::endl;
            return error;
        }

        msg.clear();
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
        if((error = allocation_test(GIGA_Float32)) != GIGA_Success)
            EARLY_ABORT();
        if((error = allocation_test(GIGA_Float16)) != GIGA_Success)
            EARLY_ABORT();
        if((error = allocation_test(GIGA_SFixed8)) != GIGA_Success)
            EARLY_ABORT();
        if((error = allocation_test(GIGA_SFixed16)) != GIGA_Success)
            EARLY_ABORT();
        if((error = allocation_test(GIGA_UFixed8)) != GIGA_Success)
            EARLY_ABORT();
        if((error = allocation_test(GIGA_UFixed16)) != GIGA_Success)
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
