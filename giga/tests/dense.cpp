/*!
 * (C) 2025 Airbus copyright all rights reserved
 * \date 16/01/2025
 */
#include <giga/giga.h>
#include "utils.h"

GIGA_error dense_test(GIGA_data_type i_GT, GIGA_data_type o_GT, GIGA_data_type k_GT, uint8_t in_shift = 0, uint8_t ker_shift = 0, uint8_t out_shift = 0)
{
    ScopedMessage msg;

    msg << "Dense, in " << giga_data_type_str(i_GT)
        << ", out " << giga_data_type_str(o_GT)
        << ", params " << giga_data_type_str(k_GT)
        << ", in_shift " << int(in_shift)
        << ", ker_shift " << int(ker_shift)
        << ", out_shift " << int(out_shift) << "\n";
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

    GIGA_tensor_t in;
    in.nb_dims = 2;
    in.dims[0] = 2;
    in.dims[1] = 3;
    in.device_id = device_id;
    in.type = i_GT;
    in.fp_shift = in_shift;

    GIGA_allocate_t in_params;
    in_params.memory_zone_id = 0;
    in_params.offset = offset;
    offset += tensor_size_in_bytes(&in);
    if((error = giga_allocate_tensor(&in, &in_params)) != GIGA_Success)
    {
        std::cerr << "Error allocating tensor in" << std::endl;
        return error;
    }

    float data_in[6] = {1.0f, 2.0f, 3.0f,
                        4.0f, 5.0f, 6.0f};
    if (is_signed(in.type))
        for(size_t i = 0 ; i < sizeof(data_in) / sizeof(data_in[0]) ; ++i)
            data_in[i] = -data_in[i];

    fill_4d_tensor(data_in, in);
    print_tensor(msg, in, "giga_dense input");

    GIGA_tensor_t out;
    out.nb_dims = 2;
    out.dims[0] = 2;
    out.dims[1] = 3;
    out.device_id = device_id;
    out.type = o_GT;
    out.fp_shift = out_shift;

    GIGA_allocate_t out_params;
    out_params.memory_zone_id = 0;
    out_params.offset = offset;
    offset += tensor_size_in_bytes(&out);
    if((error = giga_allocate_tensor(&out, &out_params)) != GIGA_Success)
    {
        std::cerr << "Error allocating tensor out" << std::endl;
        return error;
    }

    // Fill output with garbage to make sure we don't test an unwritten tensor
    fill_contiguous_tensor_with_random_data(out, 0.f, 255.f);

    GIGA_tensor_t ker;
    ker.nb_dims = 2;
    ker.dims[0] = 3;
    ker.dims[1] = 3;
    ker.device_id = device_id;
    ker.type = k_GT;
    ker.fp_shift = ker_shift;

    GIGA_allocate_t ker_params;
    ker_params.memory_zone_id = 0;
    ker_params.offset = offset;
    offset += tensor_size_in_bytes(&ker);
    if((error = giga_allocate_tensor(&ker, &ker_params)) != GIGA_Success)
    {
        std::cerr << "Error allocating tensor ker" << std::endl;
        return error;
    }

    float data_ker[9] =   {1.0f, 0.f, 0.f,
                           0.0f, 0.f, 1.f,
                           0.0f, 1.f, 0.f};

    if (is_signed(ker.type))
        for(size_t i = 0 ; i < sizeof(data_ker) / sizeof(data_ker[0]) ; ++i)
            data_ker[i] = -data_ker[i];

    fill_4d_tensor(data_ker, ker);

    print_tensor(msg, ker, "giga_dense kernel");

    GIGA_dense_t params;
    params.kernel = &ker;
    params.b_ReLU = false;
    params.bias = NULL;

    if((error = giga_dense(&params, &in, &out)) != GIGA_Success)
    {
        if (error == GIGA_Unimplemented_Type)
        {
            msg.clear();
            return GIGA_Success;
        }
        std::cerr << "Error performing giga_dense" << std::endl;
        return error;
    }

    print_tensor(msg, out, "giga_dense output");

    GIGA_tensor_t result;
    result.nb_dims = 2;
    result.dims[0] = 2;
    result.dims[1] = 3;
    result.device_id = device_id;
    result.type = o_GT;
    result.fp_shift = out_shift;

    GIGA_allocate_t result_params;
    result_params.memory_zone_id = 0;
    result_params.offset = offset;
    offset += tensor_size_in_bytes(&result);
    if((error = giga_allocate_tensor(&result, &result_params)) != GIGA_Success)
    {
        std::cerr << "Error allocating tensor result" << std::endl;
        return error;
    }

    float data_result[6] = {1.0f, 3.0f, 2.0f,
                            4.0f, 6.0f, 5.0f};
    if (is_signed(ker.type) ^ is_signed(in.type))
        for(size_t i = 0 ; i < sizeof(data_result) / sizeof(data_result[0]) ; ++i)
            data_result[i] = -data_result[i];

    fill_4d_tensor(data_result, result);
    print_tensor(msg, result, "expected output");

    if(!compare_tensors(&out, &result, 0.001))
    {
        std::cerr << "Error comparing tensors out and result" << std::endl;
        return GIGA_Unknown_Error;
    }

    if((error = giga_release_tensor(&in)) != GIGA_Success)
    {
        std::cerr << "Error releasing tensor in" << std::endl;
        return error;
    }

    if((error = giga_release_tensor(&out)) != GIGA_Success)
    {
        std::cerr << "Error releasing tensor out" << std::endl;
        return error;
    }

    if((error = giga_release_tensor(&ker)) != GIGA_Success)
    {
        std::cerr << "Error releasing tensor ker" << std::endl;
        return error;
    }

    if((error = giga_release_tensor(&result)) != GIGA_Success)
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
        if((error = dense_test(GIGA_Float32, GIGA_Float32, GIGA_Float32)) != GIGA_Success)
            EARLY_ABORT();
        if((error = dense_test(GIGA_Float16, GIGA_Float16, GIGA_Float16)) != GIGA_Success)
            EARLY_ABORT();

        for(uint8_t in_shift = 0; in_shift < 4; in_shift++)
        {
            for(uint8_t ker_shift = 0; ker_shift < 4; ker_shift++)
            {
                for(uint8_t out_shift = 0; out_shift < 2; out_shift++)
                {
#define IMPL_TYPE3(T1,T2,T3)    if(is_signed(T2) == (is_signed(T1) | is_signed(T3)) &&\
                        (error = dense_test(T1, T2, T3, in_shift, ker_shift, out_shift)) != GIGA_Success)  EARLY_ABORT()
#define IMPL_TYPE2(T1,T2)\
                    IMPL_TYPE3(T1, T2, GIGA_SFixed8);\
                    IMPL_TYPE3(T1, T2, GIGA_SFixed16);\
                    IMPL_TYPE3(T1, T2, GIGA_UFixed8);\
                    IMPL_TYPE3(T1, T2, GIGA_UFixed16)
#define IMPL_TYPE1(T1)\
                    IMPL_TYPE2(T1, GIGA_SFixed8);\
                    IMPL_TYPE2(T1, GIGA_SFixed16);\
                    IMPL_TYPE2(T1, GIGA_UFixed8);\
                    IMPL_TYPE2(T1, GIGA_UFixed16)
#define IMPL_TYPE0()\
                    IMPL_TYPE1(GIGA_SFixed8);\
                    IMPL_TYPE1(GIGA_SFixed16);\
                    IMPL_TYPE1(GIGA_UFixed8);\
                    IMPL_TYPE1(GIGA_UFixed16)

                    IMPL_TYPE0();
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
