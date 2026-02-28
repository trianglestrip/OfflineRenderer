// Cornell Box classic variant (similar to CornellBox_var.png)
// - Red left wall + Blue right wall
// - White floor/ceiling/back wall
// - Area light on ceiling
// - Gold metallic box (left)
// - Sphere (right) - reflective metal
void buildClassicCornellBox(vlrw::Scene* scene) {
    const float L = -1.0f, R = 1.0f;   // Left/Right
    const float B = 0.0f, T = 2.0f;    // Bottom/Top
    const float N = -1.0f, F = 1.0f;   // Near/Far

    // Materials
    // 0: White floor
    vlrw::MaterialDesc white;
    white.albedo = vlrw::RGB(0.82f, 0.82f, 0.82f);
    white.roughness = 1.0f;
    white.metallic = 0.0f;
    white.ior = 1.5f;
    white.emission = vlrw::RGB::black();
    scene->setMaterial(0, white);

    // 1: White ceiling
    scene->setMaterial(1, white);

    // 2: White back wall
    scene->setMaterial(2, white);

    // 3: White front wall (camera side, not visible)
    scene->setMaterial(3, white);

    // 4: Red left wall
    vlrw::MaterialDesc red;
    red.albedo = vlrw::RGB(0.73f, 0.14f, 0.14f);  // Classic Cornell Box red
    red.roughness = 1.0f;
    red.metallic = 0.0f;
    red.ior = 1.5f;
    red.emission = vlrw::RGB::black();
    scene->setMaterial(4, red);

    // 5: Blue right wall
    vlrw::MaterialDesc blue;
    blue.albedo = vlrw::RGB(0.14f, 0.14f, 0.73f);  // Classic Cornell Box blue
    blue.roughness = 1.0f;
    blue.metallic = 0.0f;
    blue.ior = 1.5f;
    blue.emission = vlrw::RGB::black();
    scene->setMaterial(5, blue);

    // 6: Gold metallic box
    vlrw::MaterialDesc gold;
    gold.albedo = vlrw::RGB(1.0f, 0.84f, 0.0f);  // Gold color
    gold.roughness = 0.2f;  // Shiny
    gold.metallic = 1.0f;   // Full metallic
    gold.ior = 1.5f;
    gold.emission = vlrw::RGB::black();
    scene->setMaterial(6, gold);

    // 7: Reflective sphere (approximate glass/metal)
    vlrw::MaterialDesc sphere;
    sphere.albedo = vlrw::RGB(0.9f, 0.9f, 0.9f);  // Bright for reflection
    sphere.roughness = 0.05f;  // Very shiny
    sphere.metallic = 1.0f;    // Metallic reflection
    sphere.ior = 1.5f;
    sphere.emission = vlrw::RGB::black();
    scene->setMaterial(7, sphere);

    // Floor (y=0, xz plane)
    float floor[] = {
        L, B, N,  R, B, N,
        R, B, F,  L, B, F
    };
    std::vector<float> floorVerts(floor, floor + 12);
    std::vector<uint32_t> floorIdx = { 0, 1, 2, 0, 2, 3 };
    scene->addTriangleMesh(std::span<const float>(floorVerts), std::span<const uint32_t>(floorIdx), 0);

    // Ceiling (y=2)
    float ceil[] = {
        L, T, N,  L, T, F,
        R, T, F,  R, T, N
    };
    std::vector<float> ceilVerts(ceil, ceil + 12);
    scene->addTriangleMesh(std::span<const float>(ceilVerts), std::span<const uint32_t>(floorIdx), 1);

    // Back wall (z=-1.001, slightly offset to break perfect plane and avoid float precision artifacts)
    // Match libVLR vertex order: left-bottom, right-bottom, right-top, left-top
    const float backZ = -1.001f;  // Slightly offset from -1.0 to avoid numerical instability
    float back[] = {
        L, B, backZ,  R, B, backZ,
        R, T, backZ,  L, T, backZ
    };
    std::vector<float> backVerts(back, back + 12);
    std::vector<uint32_t> backIdx = { 0, 1, 2, 0, 2, 3 };  // Match libVLR: LB-RB-RT, LB-RT-LT
    scene->addTriangleMesh(std::span<const float>(backVerts), std::span<const uint32_t>(backIdx), 2);

    // Front wall (z=+1, facing -z, usually not visible)
    float front[] = {
        L, B, F,  R, B, F,
        R, T, F,  L, T, F
    };
    std::vector<float> frontVerts(front, front + 12);
    scene->addTriangleMesh(std::span<const float>(frontVerts), std::span<const uint32_t>(floorIdx), 3);

    // Left wall (x=-1, RED, facing +x)
    float left[] = {
        L, B, N,  L, B, F,
        L, T, F,  L, T, N
    };
    std::vector<float> leftVerts(left, left + 12);
    scene->addTriangleMesh(std::span<const float>(leftVerts), std::span<const uint32_t>(floorIdx), 4);

    // Right wall (x=+1, BLUE, facing -x)
    float right[] = {
        R, B, N,  R, T, N,
        R, T, F,  R, B, F
    };
    std::vector<float> rightVerts(right, right + 12);
    scene->addTriangleMesh(std::span<const float>(rightVerts), std::span<const uint32_t>(floorIdx), 5);

    // Gold box (left side, ~0.4m cube)
    // Position: left-back corner, slightly elevated
    addBox(scene, -0.7f, 0.0f, -0.7f, -0.3f, 0.4f, -0.3f, 6);

    // Sphere (right side, radius ~0.3)
    // Position: right-front area, on floor
    // Note: We'll use a box as placeholder since we don't have sphere primitive
    // For better approximation, use smaller boxes or actual sphere mesh
    const float sphereR = 0.3f;
    const float sphereCenterX = 0.5f;
    const float sphereCenterY = sphereR;  // Resting on floor
    const float sphereCenterZ = 0.3f;
    
    // Approximate sphere with subdivided box (or load actual sphere mesh)
    // For now, use a cube as placeholder
    addBox(scene, 
        sphereCenterX - sphereR, sphereCenterY - sphereR, sphereCenterZ - sphereR,
        sphereCenterX + sphereR, sphereCenterY + sphereR, sphereCenterZ + sphereR,
        7);
}