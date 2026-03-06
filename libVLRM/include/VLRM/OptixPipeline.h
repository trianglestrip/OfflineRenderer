#pragma once

#include "common.h"
#include "basic_types.h"

#include <span>
#include <functional>
#include <memory>
#include <string>

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>

namespace vlrm {
    // 声明OptiX Pipeline管理器接口
    class OptixPipelineManager {
    private:
        OptixDeviceContext optix_context;
        OptixModule module;
        OptixPipeline pipeline;
        OptixProgramGroup raygen_prog_group;
        OptixProgramGroup miss_prog_group;
        OptixProgramGroup hitgroup_prog_group;

    public:
        OptixPipelineManager(OptixDeviceContext context);
        ~OptixPipelineManager();

        // PTX模块加载
        bool loadModuleFromPTX(const std::string& ptx_content);
        bool loadModuleFromPTXFile(const std::string& ptx_filename);

        // 程序组创建
        bool createProgramGroups();

        // 管线创建
        bool createPipeline();

        // 获取管线和模块
        OptixPipeline getPipeline() const;
        OptixModule getModule() const;

        // 清理函数
        void cleanup();
    };
}