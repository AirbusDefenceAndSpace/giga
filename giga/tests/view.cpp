/*!
 * (C) 2025 Airbus copyright all rights reserved
 * \date 16/01/2025
 */

#include <giga/giga.h>
#include "utils.h"

GIGA_error view_test(GIGA_data_type GT)
{
    ScopedMessage msg;
    msg << "View " << giga_data_type_str(GT) << "\n";

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
    tensor.dims[3] = 0;
    tensor.device_id = device_id;
    tensor.type = GT;
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

    const float data[50] = {1, 2, 3, 4, 5,
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
    print_tensor(msg, tensor, "giga_view input");

    GIGA_tensor_t tensor_view;
    tensor_view.nb_dims = 3;
    tensor_view.dims[0] = 2;
    tensor_view.dims[1] = 2;
    tensor_view.dims[2] = 2;
    tensor_view.dims[3] = 0;
    tensor_view.device_id = device_id;
    tensor_view.type = GT;
    tensor_view.fp_shift = 0;

    GIGA_view_t view_params;
    view_params.offset[0] = 0;
    view_params.offset[1] = 0;
    view_params.offset[2] = 1;
    view_params.offset[3] = 100;    // Check this is properly ignored

    error = giga_view(&view_params, &tensor, &tensor_view);
    if(error != GIGA_Success)
    {
        std::cerr << "Error performing giga_view" << std::endl;
        return error;
    }

    print_tensor(msg, tensor_view, "giga_view output");

    GIGA_tensor_t result;
    result.nb_dims = 3;
    result.dims[0] = 2;
    result.dims[1] = 2;
    result.dims[2] = 2;
    result.device_id = device_id;
    result.type = GT;
    result.fp_shift = 0;

    GIGA_allocate_t result_params;
    result_params.memory_zone_id = 0;
    result_params.offset = offset;
    offset += tensor_size_in_bytes(&result);
    error = giga_allocate_tensor(&result, &result_params);
    if(error != GIGA_Success)
    {
        std::cerr << "Error allocating tensor result" << std::endl;
        return error;
    }

    const float data_result[8] = {2, 3,
                             2, 3,
                             2, 3,
                             2, 3};

    fill_4d_tensor(data_result, result);

    if(!compare_tensors(&tensor_view, &result, 0.0001))
    {
        std::cerr << "Error comparing tensors tensor_view and result" << std::endl;
        return GIGA_Unknown_Error;
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

    error = giga_release_tensor(&tensor_view);
    if(error != GIGA_Success)
    {
        std::cerr << "Error releasing tensor tensor_view" << std::endl;
        return error;
    }

    const std::string line = msg.message();
    msg.replaceMessage(line.substr(0, line.find('\n')) + " OK");

    return GIGA_Success;
}

int main()
{
    GIGA_error error = GIGA_Success;

#define EARLY_ABORT() throw std::runtime_error("Error")

    try
    {
        if((error = view_test(GIGA_Float32)) != GIGA_Success)
            EARLY_ABORT();
        if((error = view_test(GIGA_Float16)) != GIGA_Success)
            EARLY_ABORT();
        if((error = view_test(GIGA_SFixed8)) != GIGA_Success)
            EARLY_ABORT();
        if((error = view_test(GIGA_SFixed16)) != GIGA_Success)
            EARLY_ABORT();
        if((error = view_test(GIGA_UFixed8)) != GIGA_Success)
            EARLY_ABORT();
        if((error = view_test(GIGA_UFixed16)) != GIGA_Success)
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

