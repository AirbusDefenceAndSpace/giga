/*!
 * (C) 2025 Airbus copyright all rights reserved
 * \date 16/01/2025
 */
#include <giga/giga.h>
#include "utils.h"

#include <cstring>

GIGA_error average_pooling_test(GIGA_data_type i_GT, GIGA_data_type o_GT, GIGA_data_type k_GT, uint8_t in_shift = 0, uint8_t ker_shift = 0, uint8_t out_shift = 0)
{
    ScopedMessage msg;

    msg << "Average Pooling, in " << giga_data_type_str(i_GT)
        << ", out " << giga_data_type_str(o_GT)
        << ", params " << giga_data_type_str(k_GT)
        << ", in_shift " << int(in_shift)
        << ", ker_shift " << int(ker_shift)
        << ", out_shift " << int(out_shift) << "\n";


    GIGA_error err;
    uint32_t device_id = giga_get_default_device_id(&err);

    if(err != GIGA_Success)
    {
        std::cerr << "Error getting default device id" << std::endl;
        return err;
    }

    err = giga_initialize_device(device_id);
    if(err != GIGA_Success)
    {
        std::cerr << "Error initializing device" << std::endl;
        return err;
    }

    size_t offset = 0;

    /*Allocate all the necessary tensors*/
    /* input tensor */
    GIGA_tensor_t in;
    in.nb_dims = 4;
    in.dims[0] = 1;
    in.dims[1] = 1;
    in.dims[2] = 6;
    in.dims[3] = 6;
    in.device_id = device_id;
    in.type = i_GT;
    in.fp_shift = in_shift;

    GIGA_allocate_t in_params;
    in_params.memory_zone_id = 0;
    in_params.offset = offset;
    offset += tensor_size_in_bytes(&in);
    err = giga_allocate_tensor(&in, &in_params);
    if(err != GIGA_Success)
    {
        std::cerr << "Error allocating tensor in" << std::endl;
        return err;
    }

    const float data_in[]= {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    fill_4d_tensor(data_in, in);
    print_tensor(msg, in, "in");

    /*output tensor*/
    GIGA_tensor_t out;
    out.nb_dims = 4;
    out.dims[0] = 1;
    out.dims[1] = 1;
    out.dims[2] = 3;
    out.dims[3] = 3;
    out.device_id = device_id;
    out.type = o_GT;
    out.data = NULL;
    out.fp_shift = out_shift;

    GIGA_allocate_t out_params;
    out_params.memory_zone_id = 0;
    out_params.offset = offset;
    offset += tensor_size_in_bytes(&out);
    err = giga_allocate_tensor(&out, &out_params);
    if(err != GIGA_Success)
    {
        std::cerr << "Error allocating tensor out" << std::endl;
        return err;
    }

    /* kernel */
    GIGA_tensor_t kernel;
    kernel.nb_dims = 4;
    kernel.dims[0] = 1;
    kernel.dims[1] = 1;
    kernel.dims[2] = 3;
    kernel.dims[3] = 3;
    kernel.device_id = device_id;
    kernel.type = k_GT;
    kernel.data = NULL;
    kernel.fp_shift = ker_shift;

    const float data_ker[] = {0.25f, 0.25f, 0.f,
                       0.25f, 0.25f, 0.f,
                       0.f, 0.f, 0.f};

    GIGA_allocate_t kernel_params;
    kernel_params.memory_zone_id = 0;
    kernel_params.offset = offset;
    offset += tensor_size_in_bytes(&kernel);
    err = giga_allocate_tensor(&kernel, &kernel_params);
    if(err != GIGA_Success)
    {
        std::cerr << "Error allocating tensor kernel" << std::endl;
        return err;
    }

    fill_4d_tensor(data_ker, kernel);
    print_tensor(msg, kernel, "kernel");

    GIGA_conv2d_t conv_params;
    conv_params.kernel = &kernel;
    conv_params.padding[0][0] = 0;
    conv_params.padding[0][1] = 1;
    conv_params.padding[1][0] = 0;
    conv_params.padding[1][1] = 1;
    conv_params.dilation[0] = 1;
    conv_params.dilation[1] = 1;
    conv_params.stride[0] = 2;
    conv_params.stride[1] = 2;
    conv_params.bias = NULL;
    conv_params.b_ReLU = false;

    /*Convolution*/
    err = giga_conv2d(&conv_params, &in, &out);
    if(err != GIGA_Success)
    {
        if (err == GIGA_Unimplemented_Type)
        {
            msg.clear();
            return GIGA_Success;
        }
        std::cerr << "Error calling giga_conv2d" << std::endl;
        return err;
    }
    print_tensor(msg, out, "giga_conv2d output");

    /* ground truth tensor*/
    GIGA_tensor_t result;
    result.nb_dims = 4;
    result.dims[0] = 1;
    result.dims[1] = 1;
    result.dims[2] = 3;
    result.dims[3] = 3;
    result.device_id = device_id;
    result.type = o_GT;
    result.data = NULL;
    result.fp_shift = out_shift;

    const float data_result[] = {1.5f, 3.5f, 5.5f,
                             1.5f, 3.5f, 5.5f,
                             1.5f, 3.5f, 5.5f};

    GIGA_allocate_t result_params;
    result_params.memory_zone_id = 0;
    result_params.offset = offset;
    offset += tensor_size_in_bytes(&result);
    err = giga_allocate_tensor(&result, &result_params);
    if(err != GIGA_Success)
    {
        std::cerr << "Error allocating tensor result" << std::endl;
        return err;
    }

    fill_4d_tensor(data_result, result);
    print_tensor(msg, result, "expected output");

    if(!compare_tensors(&out, &result, 0.0001))
    {
        std::cerr << "Error comparing tensors" << std::endl;
        return GIGA_Unknown_Error;
    }

    //Clean up
    err = giga_release_tensor(&in);
    if(err != GIGA_Success)
    {
        std::cerr << "Error releasing tensor in" << std::endl;
        return err;
    }

    err = giga_release_tensor(&out);
    if(err != GIGA_Success)
    {
        std::cerr << "Error releasing tensor out" << std::endl;
        return err;
    }

    err = giga_release_tensor(&kernel);
    if(err != GIGA_Success)
    {
        std::cerr << "Error releasing tensor kernel" << std::endl;
        return err;
    }

    err = giga_release_tensor(&result);
    if(err != GIGA_Success)
    {
        std::cerr << "Error releasing tensor result" << std::endl;
        return err;
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
        if((error = average_pooling_test(GIGA_Float32, GIGA_Float32, GIGA_Float32)) != GIGA_Success)
            EARLY_ABORT();
        if((error = average_pooling_test(GIGA_Float16, GIGA_Float16, GIGA_Float16)) != GIGA_Success)
            EARLY_ABORT();

        for(uint8_t in_shift = 0; in_shift < 4; in_shift++)
        {
            for(uint8_t ker_shift = 2; ker_shift < 4; ker_shift++)
            {
                for(uint8_t out_shift = 2; out_shift < 3; out_shift++)
                {
                    if((error = average_pooling_test(GIGA_SFixed8, GIGA_SFixed8, GIGA_SFixed8, in_shift, ker_shift, out_shift)) != GIGA_Success)
                        EARLY_ABORT();
                    if((error = average_pooling_test(GIGA_UFixed8, GIGA_UFixed8, GIGA_SFixed8, in_shift, ker_shift, out_shift)) != GIGA_Success)
                        EARLY_ABORT();

                    if((error = average_pooling_test(GIGA_SFixed16, GIGA_SFixed16, GIGA_SFixed16, in_shift, ker_shift, out_shift)) != GIGA_Success)
                        EARLY_ABORT();
                    if((error = average_pooling_test(GIGA_UFixed16, GIGA_UFixed16, GIGA_SFixed16, in_shift, ker_shift, out_shift)) != GIGA_Success)
                        EARLY_ABORT();

                    if((error = average_pooling_test(GIGA_SFixed16, GIGA_SFixed16, GIGA_SFixed8, in_shift, ker_shift, out_shift)) != GIGA_Success)
                        EARLY_ABORT();
                    if((error = average_pooling_test(GIGA_UFixed16, GIGA_UFixed16, GIGA_SFixed8, in_shift, ker_shift, out_shift)) != GIGA_Success)
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
