#include "../include/VLRM/VLRM.h"
#include "../include/VLRM/common.h"

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>

#include <vector>
#include <string>
#include <memory>
#include <fstream>

namespace vlrm {

// 声明OptiX Pipeline管理器接口
class OptixPipelineManager {
private:
    OptixDeviceContext optix_context = nullptr;
    OptixModule module = nullptr;
    OptixPipeline pipeline = nullptr;
    OptixProgramGroup raygen_prog_group = nullptr;
    OptixProgramGroup miss_prog_group = nullptr;
    OptixProgramGroup hitgroup_prog_group = nullptr;

public:
    OptixPipelineManager(OptixDeviceContext context) : optix_context(context) {}

    ~OptixPipelineManager() {
        cleanup();
    }

    bool loadModuleFromPTX(const std::string& ptx_content) {
        OptixModuleCompileOptions module_compile_options = {};
        module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

        OptixPipelineCompileOptions pipeline_compile_options = {};
        pipeline_compile_options.usesMotionBlur = false;
        pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pipeline_compile_options.numPayloadValues = 2;
        pipeline_compile_options.numAttributeValues = 2;
        pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
        pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

        char log[2048];
        size_t sizeof_log = sizeof(log);

        OptixResult result = optixModuleCreate(
            optix_context,
            &module_compile_options,
            &pipeline_compile_options,
            ptx_content.c_str(),
            ptx_content.size(),
            log,
            &sizeof_log,
            &module
        );

        if (result != OPTIX_SUCCESS) {
            vlrmprintf("OptiX module creation failed: %s\n", log);
            return false;
        }



    bool createProgramGroups() {
        OptixProgramGroupOptions prog_group_options = {}; // Initialize to zeros

        // 创建光线生成程序组
        OptixProgramGroupDesc raygen_desc = {};
        raygen_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_desc.raygen.module = module;
        raygen_desc.raygen.entryFunctionName = "__raygen__rg_main";

        char log[2048];
        size_t sizeof_log = sizeof(log);

        OptixResult result = optixProgramGroupCreate(
            optix_context,
            &raygen_desc,
            1, // count
            &prog_group_options,
            log,
            &sizeof_log,
            &raygen_prog_group
        );

        if (result != OPTIX_SUCCESS) {
            vlrmprintf("OptiX raygen program group creation failed: %s\n", log);
            return false;
        }

        // 创建miss程序组
        OptixProgramGroupDesc miss_desc = {};
        miss_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_desc.miss.module = module;
        miss_desc.miss.entryFunctionName = "__miss__ms_main";

        sizeof_log = sizeof(log);
        result = optixProgramGroupCreate(
            optix_context,
            &miss_desc,
            1, // count
            &prog_group_options,
            log,
            &sizeof_log,
            &miss_prog_group
        );

        if (result != OPTIX_SUCCESS) {
            vlrmprintf("OptiX miss program group creation failed: %s\n", log);
            return false;
        }

        // 创建hitgroup程序组
        OptixProgramGroupDesc hitgroup_desc = {};
        hitgroup_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hitgroup_desc.hitgroup.moduleCH = module;
        hitgroup_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch_main";
        hitgroup_desc.hitgroup.moduleAH = module;
        hitgroup_desc.hitgroup.entryFunctionNameAH = "__anyhit__ah_main";

        sizeof_log = sizeof(log);
        result = optixProgramGroupCreate(
            optix_context,
            &hitgroup_desc,
            1, // count
            &prog_group_options,
            log,
            &sizeof_log,
            &hitgroup_prog_group
        );

        if (result != OPTIX_SUCCESS) {
            vlrmprintf("OptiX hitgroup program group creation failed: %s\n", log);
            return false;
        }

        return true;
    }

    bool createPipeline() {
        OptixProgramGroup program_groups[] = {
            raygen_prog_group,
            miss_prog_group,
            hitgroup_prog_group
        };

        OptixPipelineLinkOptions pipeline_link_options = {};
        pipeline_link_options.maxTraceDepth = 5; // 设置最大追踪深度
        pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

        char log[2048];
        size_t sizeof_log = sizeof(log);

        OptixResult result = optixPipelineCreate(
            optix_context,
            &pipeline_link_options,
            program_groups,
            sizeof(program_groups) / sizeof(program_groups[0]),
            log,
            &sizeof_log,
            &pipeline
        );

        if (result != OPTIX_SUCCESS) {
            vlrmprintf("OptiX pipeline creation failed: %s\n", log);
            return false;
        }

        // 设置栈大小
        result = optixPipelineSetStackSize(
            pipeline,
            2 * 1024,  // direct stack size
            2 * 1024,  // continuation stack size per launch
            2,         // max graph depth
            0          // flags
        );

        if (result != OPTIX_SUCCESS) {
            vlrmprintf("OptiX pipeline stack size setting failed: %u\n", result);
            return false;
        }

        return true;
    }

    OptixPipeline getPipeline() const { return pipeline; }
    OptixModule getModule() const { return module; }

    void cleanup() {
        if (raygen_prog_group) {
            optixProgramGroupDestroy(raygen_prog_group);
            raygen_prog_group = nullptr;
        }
        if (miss_prog_group) {
            optixProgramGroupDestroy(miss_prog_group);
            miss_prog_group = nullptr;
        }
        if (hitgroup_prog_group) {
            optixProgramGroupDestroy(hitgroup_prog_group);
            hitgroup_prog_group = nullptr;
        }
        if (module) {
            optixModuleDestroy(module);
            module = nullptr;
        }
        if (pipeline) {
            optixPipelineDestroy(pipeline);
            pipeline = nullptr;
        }
    }
};

} // namespace vlrm