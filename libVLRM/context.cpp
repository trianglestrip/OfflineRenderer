#include <VLRM/VLRM.h>
#include "shared/common_internal.h"
#include "utils/optix_util.h"
#include "utils/cuda_util.h"

#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

namespace vlrm {
    struct ContextImpl {
        CUcontext cuContext;
        OptixDeviceContext optixContext;
        
        ContextImpl() : cuContext(nullptr), optixContext(nullptr) {}
    };

    Context::Context() {
    }

    Context::~Context() {
    }

    void Context::initialize() {
        VLRMAssert_NotImplemented();
    }

    void Context::finalize() {
        VLRMAssert_NotImplemented();
    }
}
