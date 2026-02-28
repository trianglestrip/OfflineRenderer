    void renderSample(
        OptixTraversableHandle gasHandle,
        CUdeviceptr d_vertices,
        CUdeviceptr d_indices,
        CUdeviceptr d_triangleMaterialIds,
        const CameraData& camera,
        uint32_t width,
        uint32_t height,
        uint32_t sampleIndex)
    {
        // Initialize active rays (all pixels)
        std::vector<uint32_t> activeIndices(numPixels);
        for (uint32_t i = 0; i < numPixels; ++i) {
            activeIndices[i] = i;
        }
        
        CUDA_CHECK(cudaMemcpy(
            (void*)d_activeIndices,
            activeIndices.data(),
            numPixels * sizeof(uint32_t),
            cudaMemcpyHostToDevice
        ));
        
        uint32_t numActive = numPixels;
        
        // Wavefront rendering loop
        const uint32_t maxDepth = 8;
        for (uint32_t depth = 0; depth < maxDepth && numActive > 0; ++depth) {
            // Setup launch parameters
            struct LaunchParams {
                OptixTraversableHandle traversable;
                RayState* rayPool;
                uint32_t* activeIndices;
                HitInfo* hitBuffer;
                const float* vertices;
                const uint32_t* indices;
                const uint32_t* triangleMaterialIds;
                CameraData camera;
                uint32_t width;
                uint32_t height;
                uint32_t sampleIndex;
                uint32_t numActive;
            };
            
            LaunchParams params;
            params.traversable = gasHandle;
            params.rayPool = reinterpret_cast<RayState*>(d_rayPool);
            params.activeIndices = reinterpret_cast<uint32_t*>(d_activeIndices);
            params.hitBuffer = reinterpret_cast<HitInfo*>(d_hitBuffer);
            params.vertices = reinterpret_cast<const float*>(d_vertices);
            params.indices = reinterpret_cast<const uint32_t*>(d_indices);
            params.triangleMaterialIds = reinterpret_cast<const uint32_t*>(d_triangleMaterialIds);
            params.camera = camera;
            params.width = width;
            params.height = height;
            params.sampleIndex = sampleIndex;
            params.numActive = numActive;
            
            CUDA_CHECK(cudaMemcpy(
                (void*)d_launchParams,
                &params,
                sizeof(LaunchParams),
                cudaMemcpyHostToDevice
            ));
            
            // Launch OptiX trace
            OPTIX_CHECK(optixLaunch(
                pipeline,
                0,  // CUDA stream
                d_launchParams,
                sizeof(LaunchParams),
                &sbt,
                numActive,
                1,
                1
            ));
            
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // TODO: Shade, compact, continue loop
            // For now, break after one iteration
            break;
        }
    }
