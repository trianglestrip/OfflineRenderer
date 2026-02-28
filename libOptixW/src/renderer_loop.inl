    void renderSample(
        OptixTraversableHandle gasHandle,
        CUdeviceptr d_vertices,
        CUdeviceptr d_texcoords,
        CUdeviceptr d_indices,
        CUdeviceptr d_triangleMaterialIds,
        const CameraData& camera,
        uint32_t width,
        uint32_t height,
        uint32_t sampleIndex)
    {
        if (numMaterials == 0 || d_materials == 0) {
            std::cout << "[Warning] No materials available for shading" << std::endl;
            return;
        }

        // Each sample must restart path state from primary rays.
        CUDA_CHECK(cudaMemset((void*)d_rayPool, 0, maxRays * sizeof(RayState)));

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
        
        CUdeviceptr activeIn = d_activeIndices;
        CUdeviceptr activeOut = d_compactIndices;
        uint32_t numActive = numPixels;
        
        // Wavefront rendering loop.
        // Shadow visibility uses extra trace stages, so allocate more iteration budget.
        const uint32_t maxDepth = 8;
        const uint32_t maxIterations = maxDepth * 2;
        const uint32_t blockSize = 256;
        for (uint32_t depth = 0; depth < maxIterations && numActive > 0; ++depth) {
            LaunchParams launchParams = {};
            launchParams.traversable = gasHandle;
            launchParams.vertices = reinterpret_cast<const float*>(d_vertices);
            launchParams.texcoords = reinterpret_cast<const float*>(d_texcoords);
            launchParams.indices = reinterpret_cast<const uint32_t*>(d_indices);
            launchParams.triangleMaterialIds = reinterpret_cast<const uint32_t*>(d_triangleMaterialIds);
            launchParams.rayPool = reinterpret_cast<RayState*>(d_rayPool);
            launchParams.activeIndices = reinterpret_cast<uint32_t*>(activeIn);
            launchParams.hitBuffer = reinterpret_cast<HitInfo*>(d_hitBuffer);
            launchParams.camera = camera;
            launchParams.width = width;
            launchParams.height = height;
            launchParams.sampleIndex = sampleIndex;
            launchParams.numActive = numActive;
            launchParams.environmentRadiance = environmentRadiance;
            launchParams.environmentMap = reinterpret_cast<const float4*>(d_environmentMap);
            launchParams.environmentMapWidth = environmentMapWidth;
            launchParams.environmentMapHeight = environmentMapHeight;
            launchParams.environmentMapScale = environmentMapScale;
            
            CUDA_CHECK(cudaMemcpy(
                (void*)d_launchParams,
                &launchParams,
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

            const uint32_t numBlocks = (numActive + blockSize - 1) / blockSize;

            ShadeKernelParams shadeParams = {};
            shadeParams.rayPool = reinterpret_cast<RayState*>(d_rayPool);
            shadeParams.activeIndices = reinterpret_cast<const uint32_t*>(activeIn);
            shadeParams.hitBuffer = reinterpret_cast<const HitInfo*>(d_hitBuffer);
            shadeParams.vertices = reinterpret_cast<const float*>(d_vertices);
            shadeParams.texcoords = reinterpret_cast<const float*>(d_texcoords);
            shadeParams.indices = reinterpret_cast<const uint32_t*>(d_indices);
            shadeParams.triangleMaterialIds = reinterpret_cast<const uint32_t*>(d_triangleMaterialIds);
            shadeParams.materials = reinterpret_cast<const MaterialData*>(d_materials);
            shadeParams.textures = reinterpret_cast<const Texture2DData*>(d_textures);
            shadeParams.accumBuffer = reinterpret_cast<float3*>(d_accumBuffer);
            shadeParams.numTriangles = numTriangles;
            shadeParams.numMaterials = numMaterials;
            shadeParams.numTextures = numTextures;
            shadeParams.numActive = numActive;
            shadeParams.environmentRadiance = environmentRadiance;
            shadeParams.environmentMap = reinterpret_cast<const float4*>(d_environmentMap);
            shadeParams.environmentMapWidth = environmentMapWidth;
            shadeParams.environmentMapHeight = environmentMapHeight;
            shadeParams.environmentMapScale = environmentMapScale;

            void* shadeArgs[] = { &shadeParams };
            CU_CHECK(cuLaunchKernel(
                shadeKernel,
                numBlocks, 1, 1,
                blockSize, 1, 1,
                0,
                0,
                shadeArgs,
                nullptr
            ));

            CUDA_CHECK(cudaMemset((void*)d_compactCounter, 0, sizeof(uint32_t)));

            CompactKernelParams compactParams = {};
            compactParams.rayPool = reinterpret_cast<const RayState*>(d_rayPool);
            compactParams.activeIndicesIn = reinterpret_cast<const uint32_t*>(activeIn);
            compactParams.activeIndicesOut = reinterpret_cast<uint32_t*>(activeOut);
            compactParams.counter = reinterpret_cast<uint32_t*>(d_compactCounter);
            compactParams.numActive = numActive;

            void* compactArgs[] = { &compactParams };
            CU_CHECK(cuLaunchKernel(
                compactKernel,
                numBlocks, 1, 1,
                blockSize, 1, 1,
                0,
                0,
                compactArgs,
                nullptr
            ));

            CUDA_CHECK(cudaMemcpy(
                &numActive,
                (void*)d_compactCounter,
                sizeof(uint32_t),
                cudaMemcpyDeviceToHost
            ));

            CUdeviceptr temp = activeIn;
            activeIn = activeOut;
            activeOut = temp;
        }
    }
