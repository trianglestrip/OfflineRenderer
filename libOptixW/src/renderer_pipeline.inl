    void createPipeline() {
        if (pipelineCreated) return;
        
        std::cout << "[Renderer] Creating OptiX pipeline..." << std::endl;
        
        // Load PTX
        std::vector<char> tracePTX = loadPTX("ptx/trace.ptx");
        
        // Create module
        OptixModuleCompileOptions moduleCompileOptions = {};
        moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
        
        OptixPipelineCompileOptions pipelineCompileOptions = {};
        pipelineCompileOptions.usesMotionBlur = false;
        pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pipelineCompileOptions.numPayloadValues = 1;
        pipelineCompileOptions.numAttributeValues = 0;
        pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
        pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";
        
        char log[2048];
        size_t logSize = sizeof(log);
        
        OPTIX_CHECK(optixModuleCreate(
            g_optixContext,
            &moduleCompileOptions,
            &pipelineCompileOptions,
            tracePTX.data(),
            tracePTX.size(),
            log, &logSize,
            &traceModule
        ));
        
        if (logSize > 1) {
            std::cout << "[OptiX Module] " << log << std::endl;
        }
        
        // Create program groups
        OptixProgramGroupOptions pgOptions = {};
        
        // Raygen
        OptixProgramGroupDesc raygenPGDesc = {};
        raygenPGDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygenPGDesc.raygen.module = traceModule;
        raygenPGDesc.raygen.entryFunctionName = "__raygen__trace";
        
        logSize = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(
            g_optixContext,
            &raygenPGDesc,
            1,
            &pgOptions,
            log, &logSize,
            &raygenPG
        ));
        
        // Miss
        OptixProgramGroupDesc missPGDesc = {};
        missPGDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        missPGDesc.miss.module = traceModule;
        missPGDesc.miss.entryFunctionName = "__miss__trace";
        
        logSize = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(
            g_optixContext,
            &missPGDesc,
            1,
            &pgOptions,
            log, &logSize,
            &missPG
        ));
        
        // Hitgroup
        OptixProgramGroupDesc hitgroupPGDesc = {};
        hitgroupPGDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hitgroupPGDesc.hitgroup.moduleCH = traceModule;
        hitgroupPGDesc.hitgroup.entryFunctionNameCH = "__closesthit__trace";
        
        logSize = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(
            g_optixContext,
            &hitgroupPGDesc,
            1,
            &pgOptions,
            log, &logSize,
            &hitgroupPG
        ));
        
        // Link pipeline
        OptixProgramGroup programGroups[] = { raygenPG, missPG, hitgroupPG };
        
        OptixPipelineLinkOptions pipelineLinkOptions = {};
        pipelineLinkOptions.maxTraceDepth = 1;
        
        logSize = sizeof(log);
        OPTIX_CHECK(optixPipelineCreate(
            g_optixContext,
            &pipelineCompileOptions,
            &pipelineLinkOptions,
            programGroups,
            3,
            log, &logSize,
            &pipeline
        ));
        
        if (logSize > 1) {
            std::cout << "[OptiX Pipeline] " << log << std::endl;
        }
        
        // Build SBT
        buildSBT();
        
        pipelineCreated = true;
        std::cout << "[Renderer] Pipeline created successfully" << std::endl;
    }
    
    void buildSBT() {
        // Raygen record
        struct RaygenRecord {
            __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        };
        RaygenRecord raygenRecord;
        OPTIX_CHECK(optixSbtRecordPackHeader(raygenPG, &raygenRecord));
        
        CUDA_CHECK(cudaMalloc(&sbt.raygenRecord, sizeof(RaygenRecord)));
        CUDA_CHECK(cudaMemcpy(
            (void*)sbt.raygenRecord,
            &raygenRecord,
            sizeof(RaygenRecord),
            cudaMemcpyHostToDevice
        ));
        
        // Miss record
        struct MissRecord {
            __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        };
        MissRecord missRecord;
        OPTIX_CHECK(optixSbtRecordPackHeader(missPG, &missRecord));
        
        CUDA_CHECK(cudaMalloc(&sbt.missRecordBase, sizeof(MissRecord)));
        CUDA_CHECK(cudaMemcpy(
            (void*)sbt.missRecordBase,
            &missRecord,
            sizeof(MissRecord),
            cudaMemcpyHostToDevice
        ));
        sbt.missRecordStrideInBytes = sizeof(MissRecord);
        sbt.missRecordCount = 1;
        
        // Hitgroup record
        struct HitgroupRecord {
            __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        };
        HitgroupRecord hitgroupRecord;
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPG, &hitgroupRecord));
        
        CUDA_CHECK(cudaMalloc(&sbt.hitgroupRecordBase, sizeof(HitgroupRecord)));
        CUDA_CHECK(cudaMemcpy(
            (void*)sbt.hitgroupRecordBase,
            &hitgroupRecord,
            sizeof(HitgroupRecord),
            cudaMemcpyHostToDevice
        ));
        sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
        sbt.hitgroupRecordCount = 1;
        
        std::cout << "[Renderer] SBT built" << std::endl;
    }
