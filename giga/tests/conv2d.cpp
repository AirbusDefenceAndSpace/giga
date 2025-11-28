/*!
 * (C) 2025 Airbus copyright all rights reserved
 * \date 16/01/2025
 */
#include <giga/giga.h>
#include "utils.h"
#include <cstring>


GIGA_error conv2d_test(GIGA_data_type i_GT, GIGA_data_type o_GT, GIGA_data_type k_GT, uint8_t in_shift = 0, uint8_t ker_shift = 0, uint8_t out_shift = 0, bool b_activation = false)
{
    ScopedMessage msg;

    msg << "Conv2d, in " << giga_data_type_str(i_GT)
        << ", out " << giga_data_type_str(o_GT)
        << ", params " << giga_data_type_str(k_GT)
        << ", in_shift " << int(in_shift)
        << ", ker_shift " << int(ker_shift)
        << ", out_shift " << int(out_shift)
        << ", activation " << int(b_activation) << "\n";

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
    in.dims[2] = 5;
    in.dims[3] = 5;
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

    float data_in[50]= {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                        1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                        1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                        1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                        1.0f, 2.0f, 3.0f, 4.0f, 5.0f,

                        2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                        2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                        2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                        2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                        2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    if (is_signed(in.type))
        for(size_t i = 0 ; i < sizeof(data_in) / sizeof(data_in[0]) ; ++i)
            data_in[i] = -data_in[i];

    fill_4d_tensor(data_in, in);
    print_tensor(msg, in, "in");

    /*output tensor*/
    GIGA_tensor_t out;
    out.nb_dims = 4;
    out.dims[0] = 1;
    out.dims[1] = 2;
    out.dims[2] = 5;
    out.dims[3] = 5;
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

    // Fill output with garbage to make sure we don't test an unwritten tensor
    fill_contiguous_tensor_with_random_data(out, 0.f, 255.f);

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

    float data_ker[9*4] = {1.0f, 0.0f, 1.0f,
                           2.0f, 0.0f, 2.0f,
                           1.0f, 0.0f, 1.0f,

                           1.0f, 1.0f, 1.0f,
                           2.0f, 2.0f, 2.0f,
                           1.0f, 1.0f, 1.0f,


                           1.0f, 0.0f, 1.0f,
                           1.0f, 0.0f, 1.0f,
                           1.0f, 0.0f, 1.0f,

                           1.0f, 1.0f, 1.0f,
                           0.0f, 0.0f, 0.0f,
                           1.0f, 1.0f, 1.0f};
    if (is_signed(kernel.type))
        for(size_t i = 0 ; i < sizeof(data_ker) / sizeof(data_ker[0]) ; ++i)
            data_ker[i] = -data_ker[i];

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

    float data_bias[2] = {1.0f, 2.0f};
    if (is_signed(bias.type) && !is_signed(in.type))
        for(size_t i = 0 ; i < sizeof(data_bias) / sizeof(data_bias[0]) ; ++i)
            data_bias[i] = -data_bias[i];

    fill_4d_tensor(data_bias, bias);
    print_tensor(msg, bias, "bias");

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
    conv_params.b_ReLU = b_activation;

    /*Convolution*/
    err = giga_conv2d(&conv_params, &in, &out);
    if(err != GIGA_Success)
    {
        if (err == GIGA_Unimplemented_Type)
        {
            msg.clear();
            return GIGA_Success;
        }
        std::cerr << "Error performing giga_conv2d" << std::endl;
        return err;
    }

    print_tensor(msg, out, "giga_conv2d output");

    /* ground truth tensor*/
    GIGA_tensor_t result;
    result.nb_dims = 4;
    result.dims[0] = 1;
    result.dims[1] = 2;
    result.dims[2] = 5;
    result.dims[3] = 5;
    result.device_id = device_id;
    result.type = o_GT;
    result.data = NULL;
    result.fp_shift = out_shift;

    float data_result[50] = {22., 40., 55., 70., 46.,
                             29., 53., 73., 93., 61.,
                             29., 53., 73., 93., 61.,
                             29., 53., 73., 93., 61.,
                             22., 40., 55., 70., 46.,

                             11., 19., 26., 33., 21.,
                             18., 32., 44., 56., 36.,
                             18., 32., 44., 56., 36.,
                             18., 32., 44., 56., 36.,
                             11., 19., 26., 33., 21.};

    if (is_signed(in.type) && !is_signed(bias.type))
    {
        // Variant when bias keeps its sign
        const float data_result_2[50] = {20, 38, 53, 68, 44,
                                         27, 51, 71, 91, 59,
                                         27, 51, 71, 91, 59,
                                         27, 51, 71, 91, 59,
                                         20, 38, 53, 68, 44,

                                         7,  15, 22, 29, 17,
                                         14, 28, 40, 52, 32,
                                         14, 28, 40, 52, 32,
                                         14, 28, 40, 52, 32,
                                         7,  15, 22, 29, 17};
        memcpy(data_result, data_result_2, sizeof(data_result_2));
    }

    if (is_signed(kernel.type) ^ is_signed(in.type))
        for(size_t i = 0 ; i < sizeof(data_result) / sizeof(data_result[0]) ; ++i)
            data_result[i] = -data_result[i];

    if (b_activation)
        for(size_t i = 0 ; i < sizeof(data_result) / sizeof(data_result[0]) ; ++i)
            data_result[i] = std::max(0.f, data_result[i]);

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
    print_tensor(msg, result, "Expected output");

    const char * const EPSILON = getenv("EPSILON");
    const double epsilon = EPSILON ? strtod(EPSILON, nullptr) : 0.001;
    if(!compare_tensors(&out, &result, epsilon))
    {
        std::cerr << "Error comparing tensors out and result" << std::endl;
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

    err = giga_release_tensor(&bias);
    if(err != GIGA_Success)
    {
        std::cerr << "Error releasing tensor bias" << std::endl;
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
        if((error = conv2d_test(GIGA_Float32, GIGA_Float32, GIGA_Float32, 0, 0, 0, true)) != GIGA_Success)
            EARLY_ABORT();
        if((error = conv2d_test(GIGA_Float32, GIGA_Float32, GIGA_Float32, 0, 0, 0, false)) != GIGA_Success)
            EARLY_ABORT();
        if((error = conv2d_test(GIGA_Float16, GIGA_Float16, GIGA_Float16, 0, 0, 0, true)) != GIGA_Success)
            EARLY_ABORT();
        if((error = conv2d_test(GIGA_Float16, GIGA_Float16, GIGA_Float16, 0, 0, 0, false)) != GIGA_Success)
            EARLY_ABORT();

        for(uint8_t in_shift = 0; in_shift <= 4; in_shift++)
        {
            for(uint8_t ker_shift = 0; ker_shift <= 4; ker_shift++)
            {
                for(uint8_t out_shift = 0; out_shift <= 4; out_shift++)
                {
#define IMPL_TYPE3(T1,T2,T3)\
                    if(is_signed(T2) == (is_signed(T1) | is_signed(T3)) &&\
                        (error = conv2d_test(T1, T2, T3, in_shift, ker_shift, out_shift, false)) != GIGA_Success)   EARLY_ABORT();\
                    if(is_signed(T2) == (is_signed(T1) | is_signed(T3)) &&\
                        (error = conv2d_test(T1, T2, T3, in_shift, ker_shift, out_shift, true)) != GIGA_Success)   EARLY_ABORT()
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
