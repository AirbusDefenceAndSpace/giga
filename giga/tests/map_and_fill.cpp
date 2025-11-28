/*!
 * (C) 2025 Airbus copyright all rights reserved
 * \date 16/01/2025
 */

#include <giga/giga.h>

#include "utils.h"

GIGA_error map_and_fill_test(GIGA_data_type GT)
{
    ScopedMessage msg;
    msg << "Allocation " << giga_data_type_str(GT) << "\n";

    GIGA_error error;
    uint32_t device_id = giga_get_default_device_id(&error);

    if(error != GIGA_Success)
    {
        std::cerr << "Error getting default device id" << std::endl;
        return error;
    }

    if((error = giga_initialize_device(device_id)) != GIGA_Success)
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

    GIGA_allocate_t tensor_params;
    tensor_params.memory_zone_id = 0;
    tensor_params.offset = offset;
    offset += tensor_size_in_bytes(&tensor);
    if((error = giga_allocate_tensor(&tensor, &tensor_params)) != GIGA_Success)
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

    if((error = giga_release_tensor(&tensor)) != GIGA_Success)
    {
        std::cerr << "Error releasing tensor tensor" << std::endl;
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
        if((error = map_and_fill_test(GIGA_Float32)) != GIGA_Success)
            EARLY_ABORT();
        if((error = map_and_fill_test(GIGA_Float16)) != GIGA_Success)
            EARLY_ABORT();
        if((error = map_and_fill_test(GIGA_SFixed8)) != GIGA_Success)
            EARLY_ABORT();
        if((error = map_and_fill_test(GIGA_SFixed16)) != GIGA_Success)
            EARLY_ABORT();
        if((error = map_and_fill_test(GIGA_UFixed8)) != GIGA_Success)
            EARLY_ABORT();
        if((error = map_and_fill_test(GIGA_UFixed16)) != GIGA_Success)
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
