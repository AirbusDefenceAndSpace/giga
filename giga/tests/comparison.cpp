/*!
 * (C) 2025 Airbus copyright all rights reserved
 * \date 16/01/2025
 */
#include <giga/giga.h>
#include "utils.h"

GIGA_error comparison_test(GIGA_data_type a_GT, GIGA_data_type b_GT, uint8_t a_shift = 0, uint8_t b_shift = 0)
{
    ScopedMessage msg;

    msg << "Comparison, a " << giga_data_type_str(a_GT)
        << ", b " << giga_data_type_str(b_GT)
        << ", a_shift " << int(a_shift)
        << ", b_shift " << int(b_shift) << "\n";

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

    GIGA_tensor_t tensor_1;
    tensor_1.nb_dims = 4;
    tensor_1.dims[0] = 1;
    tensor_1.dims[1] = 2;
    tensor_1.dims[2] = 5;
    tensor_1.dims[3] = 5;
    tensor_1.device_id = device_id;
    tensor_1.type = a_GT;
    tensor_1.fp_shift = a_shift;

    GIGA_allocate_t tensor_1_params;
    tensor_1_params.memory_zone_id = 0;
    tensor_1_params.offset = offset;
    offset += tensor_size_in_bytes(&tensor_1);
    if((error = giga_allocate_tensor(&tensor_1, &tensor_1_params)) != GIGA_Success)
    {
        std::cerr << "Error allocating tensor tensor_1" << std::endl;
        return error;
    }

    GIGA_tensor_t tensor_2;
    tensor_2.nb_dims = 4;
    tensor_2.dims[0] = 1;
    tensor_2.dims[1] = 2;
    tensor_2.dims[2] = 5;
    tensor_2.dims[3] = 5;
    tensor_2.device_id = device_id;
    tensor_2.type = b_GT;
    tensor_2.fp_shift = b_shift;

    GIGA_allocate_t tensor_2_params;
    tensor_2_params.memory_zone_id = 0;
    tensor_2_params.offset = offset;
    offset += tensor_size_in_bytes(&tensor_2);
    if((error = giga_allocate_tensor(&tensor_2, &tensor_2_params)) != GIGA_Success)
    {
        std::cerr << "Error allocating tensor tensor_2" << std::endl;
        return error;
    }

    float data1[50] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                       1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                       1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                       1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                       1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                       1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                       1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                       1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                       1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                       1.0f, 2.0f, 3.0f, 4.0f, 5.0f};


    float data2[50] =   {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 6.0f};


    fill_4d_tensor(data1, tensor_1);
    fill_4d_tensor(data1, tensor_2);

    if(!compare_tensors(&tensor_1, &tensor_2))
    {
        std::cerr << "Error comparing tensors tensor_1 and tensor_2 with same content" << std::endl;
        return GIGA_Unknown_Error;
    }

    fill_4d_tensor(data2, tensor_2);

    if(compare_tensors(&tensor_1, &tensor_2))
    {
        std::cerr << "Error comparing tensors tensor_1 and tensor_2 with different content" << std::endl;
        return GIGA_Unknown_Error;
    }

    if((error = giga_release_tensor(&tensor_1)) != GIGA_Success)
    {
        std::cerr << "Error releasing tensor tensor_1" << std::endl;
        return error;
    }

    if((error = giga_release_tensor(&tensor_2))  != GIGA_Success)
    {
        std::cerr << "Error releasing tensor tensor_2" << std::endl;
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
        if((error = comparison_test(GIGA_Float32, GIGA_Float32)) != GIGA_Success)
            EARLY_ABORT();
        if((error = comparison_test(GIGA_Float16, GIGA_Float16)) != GIGA_Success)
            EARLY_ABORT();
        for(uint8_t a_shift = 0; a_shift < 4; a_shift++)
        {
            for(uint8_t b_shift = 0; b_shift < 4; b_shift++)
            {
                if((error = comparison_test(GIGA_SFixed8, GIGA_SFixed8, a_shift, b_shift)) != GIGA_Success)
                    EARLY_ABORT();
                if((error = comparison_test(GIGA_SFixed16, GIGA_SFixed16, a_shift, b_shift)) != GIGA_Success)
                    EARLY_ABORT();
                if((error = comparison_test(GIGA_UFixed8, GIGA_UFixed8, a_shift, b_shift)) != GIGA_Success)
                    EARLY_ABORT();
                if((error = comparison_test(GIGA_UFixed16, GIGA_UFixed16, a_shift, b_shift)) != GIGA_Success)
                    EARLY_ABORT();

                if((error = comparison_test(GIGA_SFixed8, GIGA_SFixed16, a_shift, b_shift)) != GIGA_Success)
                    EARLY_ABORT();
                if((error = comparison_test(GIGA_UFixed8, GIGA_UFixed16, a_shift, b_shift)) != GIGA_Success)
                    EARLY_ABORT();
            }
        }
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
