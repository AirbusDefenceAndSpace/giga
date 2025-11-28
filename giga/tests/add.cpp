/*!
 * (C) 2025 Airbus copyright all rights reserved
 * \date 16/01/2025
 */
#include <giga/giga.h>
#include "utils.h"

GIGA_error addition_test(GIGA_data_type GT, uint8_t a_shift = 0, uint8_t b_shift = 0, uint8_t out_shift = 0)
{
    ScopedMessage msg;
    msg << "Add " << giga_data_type_str(GT) << ", a_shift " << int(a_shift) << ", b_shift " << int(b_shift) << ", out_shift " << int(out_shift) << "\n";

    GIGA_error error;
    uint32_t device_id = giga_get_default_device_id(&error);

    if(error != GIGA_Success)
        return error;

    if((error = giga_initialize_device(device_id)) != GIGA_Success)
        return error;

    size_t offset = 0;

    GIGA_tensor_t a;
    a.nb_dims = 4;
    a.dims[0] = 1;
    a.dims[1] = 1;
    a.dims[2] = 5;
    a.dims[3] = 5;
    a.device_id = device_id;
    a.type = GT;
    a.data = NULL;
    a.fp_shift = a_shift;

    GIGA_allocate_t a_params;
    a_params.memory_zone_id = 0;
    a_params.offset = offset;
    offset += tensor_size_in_bytes(&a);
    if((error = giga_allocate_tensor(&a, &a_params)) != GIGA_Success)
    {
        std::cerr << "Error allocating tensor a" << std::endl;
        return error;
    }

    const float  data_a[25]= {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                              1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                              1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                              1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                              1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    if ((error = giga_copy_to_tensor(data_a, GIGA_Float32, 0, &a)) != GIGA_Success)
    {
        if (error == GIGA_Unimplemented_Type)
        {
            msg.clear();
            return GIGA_Success;
        }
        std::cerr << "Error filling tensor with data" << std::endl;
        return error;
    }

    GIGA_tensor_t b;
    b.nb_dims = 4;
    b.dims[0] = 1;
    b.dims[1] = 1;
    b.dims[2] = 5;
    b.dims[3] = 5;
    b.data = NULL;
    b.fp_shift = b_shift,

    b.device_id = device_id;
    b.type = GT;

    GIGA_allocate_t b_params;
    b_params.memory_zone_id = 0;
    b_params.offset = offset;
    offset += tensor_size_in_bytes(&b);
    if((error = giga_allocate_tensor(&b, &b_params) ) != GIGA_Success)
    {
        std::cerr << "Error allocating tensor b" << std::endl;
        return error;
    }

    const float data_b[25]= {-1.0f, 2.0f, -3.0f, 4.0f, -5.0f,
                             1.0f, -2.0f, 3.0f, -4.0f, 5.0f,
                             -1.0f, 2.0f, -3.0f, 4.0f, -5.0f,
                             1.0f, -2.0f, 3.0f, -4.0f, 5.0f,
                             -1.0f, 2.0f, -3.0f, 4.0f, -5.0f};

    if ((error = giga_copy_to_tensor(data_b, GIGA_Float32, 0, &b)) != GIGA_Success)
    {
        std::cerr << "Error filling tensor with data" << std::endl;
        return error;
    }

    GIGA_tensor_t out;
    out.nb_dims = 4;
    out.dims[0] = 1;
    out.dims[1] = 1;
    out.dims[2] = 5;
    out.dims[3] = 5;
    out.device_id = device_id;
    out.type = GT;
    out.data = NULL;
    out.fp_shift = out_shift;

    GIGA_allocate_t out_params;
    out_params.memory_zone_id = 0;
    out_params.offset = offset;
    offset += tensor_size_in_bytes(&out);
    if((error = giga_allocate_tensor(&out, &out_params) ) != GIGA_Success)
    {
        std::cerr << "Error allocating tensor out" << std::endl;
        return error;
    }

    // Fill output with garbage to make sure we don't test an unwritten tensor
    fill_contiguous_tensor_with_random_data(out, 0.f, 255.f);

    GIGA_add_t add_params;
    if((error = giga_add(&add_params, &a, &b, &out) ) != GIGA_Success)
    {
        if (error == GIGA_Unimplemented_Type)
        {
            msg.clear();
            return GIGA_Success;
        }
        std::cerr << "Error performing add on a and b to out" << std::endl;
        return error;
    }

    GIGA_tensor_t result;
    result.nb_dims = 4;
    result.dims[0] = 1;
    result.dims[1] = 1;
    result.dims[2] = 5;
    result.dims[3] = 5;
    result.data = NULL;
    result.fp_shift = out_shift;

    result.device_id = device_id;
    result.type = GT;

    GIGA_allocate_t result_params;
    result_params.memory_zone_id = 0;
    result_params.offset = offset;
    offset += tensor_size_in_bytes(&result);
    if((error = giga_allocate_tensor(&result, &result_params) ) != GIGA_Success)
    {
        std::cerr << "Error allocating tensor result" << std::endl;
        return error;
    }

    const float data_result[25]= {0.f,  4.f,  0.f,  8.f,  0.f,
                                  2.f,  0.f,  6.f,  0.f, 10.f,
                                  0.f,  4.f,  0.f,  8.f,  0.f,
                                  2.f,  0.f,  6.f,  0.f, 10.f,
                                  0.f,  4.f,  0.f,  8.f,  0.f};

    if ((error = giga_copy_to_tensor(data_result, GIGA_Float32, 0, &result)) != GIGA_Success)
    {
        std::cerr << "Error filling tensor with data" << std::endl;
        return error;
    }

    if(!compare_tensors(&out, &result))
    {
        print_tensor(msg, a, "a");
        print_tensor(msg, b, "b");
        print_tensor(msg, out, "out");
        print_tensor(msg, result, "result");
        std::cerr << "Error comparing tensors" << std::endl;
        return GIGA_Unknown_Error;
    }

    //Clean up
    if((error = giga_release_tensor(&a) ) != GIGA_Success)
    {
        std::cerr << "Error releasing tensor a" << std::endl;
        return error;
    }

    if((error = giga_release_tensor(&b) ) != GIGA_Success)
    {
        std::cerr << "Error releasing tensor b" << std::endl;
        return error;
    }

    if((error = giga_release_tensor(&out) ) != GIGA_Success)
    {
        std::cerr << "Error releasing tensor out" << std::endl;
        return error;
    }

    if((error = giga_release_tensor(&result) ) != GIGA_Success)
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
        if((error = addition_test(GIGA_Float32)) != GIGA_Success)
            EARLY_ABORT();
        if((error = addition_test(GIGA_Float16)) != GIGA_Success)
            EARLY_ABORT();
        for(uint8_t a_shift = 0; a_shift < 4; a_shift++)
        {
            for(uint8_t b_shift = 0; b_shift < 4; b_shift++)
            {
                for(uint8_t out_shift = 0; out_shift < 4; out_shift++)
                {
                    if((error = addition_test(GIGA_SFixed8, a_shift, b_shift, out_shift)) != GIGA_Success)
                        EARLY_ABORT();
                    if((error = addition_test(GIGA_SFixed16, a_shift, b_shift, out_shift)) != GIGA_Success)
                        EARLY_ABORT();
                }
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
