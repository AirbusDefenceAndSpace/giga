/*!
 * (C) 2025 Airbus copyright all rights reserved
 * \date 17/01/2025
 */
#include <giga/giga.h>
#include "../tests/utils.h"

#include <cstring>


GIGA_error conv2d_benchmark(GIGA_data_type i_GT, GIGA_data_type o_GT, GIGA_data_type k_GT, int nb_runs, uint8_t in_shift = 0, uint8_t ker_shift = 0, uint8_t out_shift = 0)
{
    ScopedMessage on_error_message(std::string("Error on ")
                                   + "Conv2d, in " + giga_data_type_str(i_GT)
                                   + ", out " + giga_data_type_str(o_GT)
                                   + ", params " + giga_data_type_str(k_GT)
                                   + ", in_shift " + std::to_string(int(in_shift))
                                   + ", ker_shift " + std::to_string(int(ker_shift))
                                   + ", out_shift " + std::to_string(int(out_shift)));

    std::cout << "Conv2d, in " << giga_data_type_str(i_GT)
              << ", out " << giga_data_type_str(o_GT)
              << ", params " << giga_data_type_str(k_GT)
              << ", in_shift " << int(in_shift)
              << ", ker_shift " << int(ker_shift)
              << ", out_shift " << int(out_shift) << " : " << std::flush;

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

    /*Allocate all the necessary tensors*/

    size_t offset = 0;

    /* input tensor */
    GIGA_tensor_t in;
    in.nb_dims = 4;
    in.dims[0] = 1;
    in.dims[1] = 2;
    in.dims[2] = 1024;
    in.dims[3] = 1024;
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

    fill_contiguous_tensor_with_random_data(in, 0, 1);

    /*output tensor*/
    GIGA_tensor_t out = in;
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
    kernel.dims[0] = 2;
    kernel.dims[1] = 2;
    kernel.dims[2] = 3;
    kernel.dims[3] = 3;
    kernel.device_id = device_id;
    kernel.type = k_GT;
    kernel.data = NULL;
    kernel.fp_shift = ker_shift;

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

    fill_contiguous_tensor_with_random_data(kernel, -1.f, 1.f);

    /* bias */
    GIGA_tensor_t bias;
    bias.nb_dims = 1;
    bias.dims[0] = 2;
    bias.device_id = device_id;
    bias.type = k_GT;
    bias.data = NULL;
    bias.fp_shift = ker_shift; //Should probably be different

    GIGA_allocate_t bias_params;
    bias_params.memory_zone_id = 0;
    bias_params.offset = offset;
    offset += tensor_size_in_bytes(&bias);
    err = giga_allocate_tensor(&bias, &bias_params);
    if(err != GIGA_Success)
    {
        std::cerr << "Error allocating tensor bias" << std::endl;
        return err;
    }

    fill_contiguous_tensor_with_random_data(bias, 0.f, 1.f);

    GIGA_conv2d_t conv_params;
    conv_params.kernel = &kernel;
    conv_params.padding[0][0] = 1;
    conv_params.padding[0][1] = 1;
    conv_params.padding[1][0] = 1;
    conv_params.padding[1][1] = 1;
    conv_params.dilation[0] = 1;
    conv_params.dilation[1] = 1;
    conv_params.stride[0] = 1;
    conv_params.stride[1] = 1;
    conv_params.bias = &bias;
    conv_params.b_ReLU = false;

    /*Convolution*/
    const size_t start = usec_timer();
    for(int it = 0 ; it < nb_runs ; ++it)
    {
        err = giga_conv2d(&conv_params, &in, &out);
        if(err != GIGA_Success)
        {
            if (err == GIGA_Unimplemented_Type)
            {
                std::cout << "Type not implemented" << std::endl;
                on_error_message.clear();
                return GIGA_Success;
            }
            if (err == GIGA_Not_Implemented)
            {
                std::cout << "Function not implemented" << std::endl;
                on_error_message.clear();
                return GIGA_Success;
            }
            std::cerr << "Error performing giga_conv2d" << std::endl;
            return err;
        }
    }
    if ((err = giga_flush(device_id)) != GIGA_Success)
    {
        std::cerr << "Error flushing device" << std::endl;
        return err;
    }
    if ((err = giga_wait_for_completion()) != GIGA_Success)
    {
        std::cerr << "Error waiting for completion" << std::endl;
        return err;
    }
    const size_t end = usec_timer();
    std::cout << (end - start) / nb_runs << "µs per call" << std::endl;

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

    on_error_message.clear();

    return GIGA_Success;
}

int main()
{
    GIGA_error error = GIGA_Success;

    const int nb_runs = 10;

#define EARLY_ABORT() throw std::runtime_error("Error")

    try
    {
        if((error = conv2d_benchmark(GIGA_Float32, GIGA_Float32, GIGA_Float32, nb_runs)) != GIGA_Success)
            EARLY_ABORT();
        if((error = conv2d_benchmark(GIGA_Float16, GIGA_Float16, GIGA_Float16, nb_runs)) != GIGA_Success)
            EARLY_ABORT();

        if((error = conv2d_benchmark(GIGA_SFixed8, GIGA_SFixed8, GIGA_SFixed8, nb_runs, 4, 4, 4)) != GIGA_Success)
            EARLY_ABORT();
        if((error = conv2d_benchmark(GIGA_UFixed8, GIGA_SFixed8, GIGA_SFixed8, nb_runs, 4, 4, 4)) != GIGA_Success)
            EARLY_ABORT();

        if((error = conv2d_benchmark(GIGA_SFixed16, GIGA_SFixed16, GIGA_SFixed16, nb_runs, 4, 4, 4)) != GIGA_Success)
            EARLY_ABORT();
        if((error = conv2d_benchmark(GIGA_UFixed16, GIGA_SFixed16, GIGA_SFixed16, nb_runs, 4, 4, 4)) != GIGA_Success)
            EARLY_ABORT();

        if((error = conv2d_benchmark(GIGA_SFixed8, GIGA_SFixed16, GIGA_SFixed16, nb_runs, 4, 4, 4)) != GIGA_Success)
            EARLY_ABORT();
        if((error = conv2d_benchmark(GIGA_UFixed8, GIGA_SFixed16, GIGA_SFixed16, nb_runs, 4, 4, 4)) != GIGA_Success)
            EARLY_ABORT();

        if((error = conv2d_benchmark(GIGA_SFixed16, GIGA_SFixed8, GIGA_SFixed16, nb_runs, 4, 4, 4)) != GIGA_Success)
            EARLY_ABORT();
        if((error = conv2d_benchmark(GIGA_UFixed16, GIGA_SFixed8, GIGA_SFixed16, nb_runs, 4, 4, 4)) != GIGA_Success)
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
