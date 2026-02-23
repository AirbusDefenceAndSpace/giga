#include <iostream>
#include <fstream>
#include <vector>
#include <iterator>
#include <iomanip>
#include <cstring>
#include <giga/giga.h>
#include "../../tested_outputs/output/main_graph.h"
#include "../../tested_outputs/output/main_graph.c"

/**
 * giga_python_bridge.cpp
 *
 * C++ bridge between Python (via ctypes) and the GIGA inference engine.
 * Compiles into a shared library (.so) that Python loads at runtime.
 *
 * Workflow:
 *   1. Initializes the GIGA device.
 *   2. Allocates and fills model tensors.
 *   3. Reads input data from a text file (space-separated floats).
 *   4. Runs inference via the generated main_graph functions.
 *   5. Writes output tensor values to a text file.
 *
 */


class GigaInferenceContext {
public:
    main_graph_tensors tensors;
    main_graph_io io;
    main_graph_ops ops;
    bool initialized = false;
    GigaInferenceContext() {
        std::memset(&tensors, 0, sizeof(tensors));
        std::memset(&io, 0, sizeof(io));
        std::memset(&ops, 0, sizeof(ops));
    }
    ~GigaInferenceContext() {
        GIGA_tensor_t* t_ptr = reinterpret_cast<GIGA_tensor_t*>(&tensors);
        size_t count = sizeof(main_graph_tensors) / sizeof(GIGA_tensor_t);
        for (size_t i = 0; i < count; ++i) {
            if (t_ptr[i].data) {
                giga_release_tensor(&t_ptr[i]);
                t_ptr[i].data = nullptr;
            }
        }
        if (io.input.data) giga_release_tensor(&io.input);
        if (io.output.data) giga_release_tensor(&io.output);
    }
};
extern "C" {
    void run_inference(const char* input_path, const char* output_path) {
        GigaInferenceContext ctx;
        uint32_t dev = 0;
        GIGA_error err = (GIGA_error)initialize_main_graph(&dev);
        if (err != GIGA_Success) {
            std::cerr << "CRITICAL: Device init failed: " << giga_str_error(err) << std::endl;
            return; 
        }
        err = (GIGA_error)allocate_main_graph_tensors(&ctx.tensors, &ctx.io, dev);
        if (err != GIGA_Success) {
            std::cerr << "CRITICAL: Allocation failed: " << giga_str_error(err) << std::endl;
            return;
        }
        fill_main_graph_tensors(&ctx.tensors);
        set_main_graph_ops(&ctx.ops, &ctx.tensors);
        std::ifstream file(input_path);
        if (!file.is_open()) {
            std::cerr << "ERROR: Input file not found: " << input_path << std::endl;
            return;
        }
        std::vector<float> data((std::istream_iterator<float>(file)), std::istream_iterator<float>());
        
        if (data.empty()) {
            std::cerr << "ERROR: No data in input file" << std::endl;
            return;
        }
        giga_copy_to_tensor(data.data(), GIGA_Float32, 0, &ctx.io.input);
        
        process_main_graph_tensors(&ctx.tensors, &ctx.ops, &ctx.io);
        float* out_ptr = nullptr;
        if (giga_map_tensor(&ctx.io.output, (void**)&out_ptr, GIGA_Memory_Sync) == GIGA_Success) {
            size_t out_size = 1;
            for (uint32_t i = 0; i < ctx.io.output.nb_dims; i++) {
                if (ctx.io.output.dims[i] > 0) out_size *= ctx.io.output.dims[i];
            }
            std::ofstream res(output_path);
            res << std::setprecision(8) << std::fixed;
            for (size_t i = 0; i < out_size; ++i) res << out_ptr[i] << "\n";
            
            giga_unmap_tensor(&ctx.io.output, out_ptr, GIGA_Memory_Discard);
        } else {
            std::cerr << "ERROR: Could not map output tensor" << std::endl;
        }
    }
}