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
            // Setup launch parameters (defined in kernels/launch_params.cuh)
            struct LaunchParams {
                // Scene data
                OptixTraversableHandle traversable;
                const float* vertices;
                const uint32_t* indices;
                const uint32_t* triangleMaterialIds;
                
                // Ray pool and buffers
                RayState* rayPool;
                uint32_t* activeIndices;
                HitInfo* hitBuffer;
                
                // Camera
                CameraData camera;
                
                // Render settings
                uint32_t width;
                uint32_t height;
                uint32_t sampleIndex;
                uint32_t numActive;
            };
            
            LaunchParams params;
            params.traversable = gasHandle;
            params.vertices = reinterpret_cast<const float*>(d_vertices);
            params.indices = reinterpret_cast<const uint32_t*>(d_indices);
            params.triangleMaterialIds = reinterpret_cast<const uint32_t*>(d_triangleMaterialIds);
            params.rayPool = reinterpret_cast<RayState*>(d_rayPool);
            params.activeIndices = reinterpret_cast<uint32_t*>(d_activeIndices);
            params.hitBuffer = reinterpret_cast<HitInfo*>(d_hitBuffer);
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
            
            // Shade on CPU (temporary simple implementation)
            {
                // Skip shading if no materials (safety check)
                if (numMaterials == 0 || d_materials == 0) {
                    std::cout << "[Warning] No materials available for shading" << std::endl;
                    break;
                }
                
                std::vector<RayState> rayStates(numPixels);
                std::vector<HitInfo> hitInfos(numPixels);
                std::vector<MaterialData> materials(numMaterials);
                std::vector<float3> accumBuffer(numPixels);
                
                CUDA_CHECK(cudaMemcpy(
                    rayStates.data(),
                    (void*)d_rayPool,
                    numPixels * sizeof(RayState),
                    cudaMemcpyDeviceToHost
                ));
                
                CUDA_CHECK(cudaMemcpy(
                    hitInfos.data(),
                    (void*)d_hitBuffer,
                    numPixels * sizeof(HitInfo),
                    cudaMemcpyDeviceToHost
                ));
                
                CUDA_CHECK(cudaMemcpy(
                    materials.data(),
                    (void*)d_materials,
                    numMaterials * sizeof(MaterialData),
                    cudaMemcpyDeviceToHost
                ));
                
                CUDA_CHECK(cudaMemcpy(
                    accumBuffer.data(),
                    (void*)d_accumBuffer,
                    numPixels * sizeof(float3),
                    cudaMemcpyDeviceToHost
                ));
                
                // Simple shading: check for emissive materials
                for (uint32_t idx = 0; idx < numActive; ++idx) {
                    uint32_t rayIndex = activeIndices[idx];
                    RayState& ray = rayStates[rayIndex];
                    
                    if (ray.stage == RayState::Shade) {
                        const HitInfo& hit = hitInfos[rayIndex];
                        const MaterialData& mat = materials[hit.materialId];
                        
                        // Check if emissive
                        bool isEmissive = (mat.emission.x > 0.0f || mat.emission.y > 0.0f || mat.emission.z > 0.0f);
                        
                        if (isEmissive) {
                            // Accumulate emission
                            ray.radiance.x += ray.throughput.x * mat.emission.x;
                            ray.radiance.y += ray.throughput.y * mat.emission.y;
                            ray.radiance.z += ray.throughput.z * mat.emission.z;
                            
                            accumBuffer[ray.pixelIndex] = ray.radiance;
                            ray.stage = RayState::Terminated;
                        } else {
                            // For non-emissive, set to albedo * 0.5 (ambient lighting approximation)
                            accumBuffer[ray.pixelIndex].x = mat.albedo.x * 0.5f;
                            accumBuffer[ray.pixelIndex].y = mat.albedo.y * 0.5f;
                            accumBuffer[ray.pixelIndex].z = mat.albedo.z * 0.5f;
                            ray.stage = RayState::Terminated;
                        }
                    }
                }
                
                // Upload back
                CUDA_CHECK(cudaMemcpy(
                    (void*)d_rayPool,
                    rayStates.data(),
                    numPixels * sizeof(RayState),
                    cudaMemcpyHostToDevice
                ));
                
                CUDA_CHECK(cudaMemcpy(
                    (void*)d_accumBuffer,
                    accumBuffer.data(),
                    numPixels * sizeof(float3),
                    cudaMemcpyHostToDevice
                ));
            }
            
            // Compact active rays (simple version: just count non-terminated rays)
            {
                std::vector<RayState> rayStates(numPixels);
                CUDA_CHECK(cudaMemcpy(
                    rayStates.data(),
                    (void*)d_rayPool,
                    numPixels * sizeof(RayState),
                    cudaMemcpyDeviceToHost
                ));
                
                activeIndices.clear();
                for (uint32_t i = 0; i < numPixels; ++i) {
                    if (rayStates[i].stage == RayState::Trace) {
                        activeIndices.push_back(i);
                    }
                }
                
                numActive = activeIndices.size();
                
                if (numActive > 0) {
                    CUDA_CHECK(cudaMemcpy(
                        (void*)d_activeIndices,
                        activeIndices.data(),
                        numActive * sizeof(uint32_t),
                        cudaMemcpyHostToDevice
                    ));
                }
            }
        }
    }
