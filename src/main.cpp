/** 
Primarly based on University of Pennsylvania's CIS 5650 Boids flocking simulation starter code
*/

#define TINYGLTF_IMPLEMENTATION     
#define STB_IMAGE_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_WRITE

#include "tiny_gltf.h"
#include "main.hpp"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "kernel.h"

#define IMGUI_IMPL_OPENGL_LOADER_GLEW
#include "../imgui/imgui.h"
#include "../imgui/imgui_impl_glfw.h"
#include "../imgui/imgui_impl_opengl3.h"

// ================
// Configuration
// ================

#define VISUALIZE 1
#define UNIFORM_GRID 0
#define COHERENT_GRID 0


const int N_FOR_VIS = 100000;
const float DT = 0.06f;



SimulationType currentSimulation = SimulationType::SPH;
BoidsParams boidsSettings;
SPHParams sphSettings;

// should be uniform from the kernel scene scale
const float SCENE_SCALE = 100.0f;

//const float DT = 0.4f;

/**
* C main function.
*/
int main(int argc, char* argv[]) {
  projectName = "CuBoids";

  if (init(argc, argv)) {
    mainLoop();
    Simulator::endSimulation();
    return 0;
  } else {
    return 1;
  }
}

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

std::string deviceName;
GLFWwindow *window;
static ImGuiWindowFlags windowFlags = ImGuiWindowFlags_None | ImGuiWindowFlags_NoMove;
/**
* Initialization of CUDA and GLFW.
*/
bool init(int argc, char **argv) {
   cudaDeviceProp deviceProp;
  int gpuDevice = 0;
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (gpuDevice > device_count) {
    std::cout
    << "Error: GPU device number is greater than the number of devices!"
    << " Perhaps a CUDA-capable GPU is not installed?"
    << std::endl;
    return false;
  }
  cudaGetDeviceProperties(&deviceProp, gpuDevice);
  int major = deviceProp.major;
  int minor = deviceProp.minor;

  std::ostringstream ss;
  ss << "Cuboids" << " [SM " << major << "." << minor << " " << deviceProp.name << "]";
  deviceName = ss.str();

  // Window setup stuff
  glfwSetErrorCallback(errorCallback);

  if (!glfwInit()) {
    std::cout
    << "Error: Could not initialize GLFW!"
    << " Perhaps OpenGL 3.3 isn't available?"
    << std::endl;
    return false;
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  window = glfwCreateWindow(width, height, deviceName.c_str(), NULL, NULL);
  if (!window) {
    glfwTerminate();
    return false;
  }
  glfwMakeContextCurrent(window);
  glfwSetKeyCallback(window, keyCallback);
  glfwSetCursorPosCallback(window, mousePositionCallback);
  glfwSetMouseButtonCallback(window, mouseButtonCallback);

  glewExperimental = GL_TRUE;
  if (glewInit() != GLEW_OK) {
    return false;
  }





  // Initialize drawing state
  initVAO();

  // Default to device ID 0. If you have more than one GPU and want to test a non-default one,
  // change the device ID.
  cudaGLSetGLDevice(0);

  cudaGLRegisterBufferObject(boidVBO_positions);
  cudaGLRegisterBufferObject(boidVBO_velocities);

  // Initialize N-body simulation
  std::vector<glm::vec3> mverts; std::vector<glm::vec2> muvs;
  std::vector<unsigned char> texPixels; int texW, texH, texCh;
  if (!loadGltf("C:\\Dev\\Project1-CUDA-Flocking-main\\Project1-CUDA-Flocking-main\\ducky.gltf", mverts, meshNormals, muvs, mtris, texPixels, texW, texH, texCh))
      std::cout << "could not load ducky\n";
  originalVerts = mverts;

  updateMeshTransform();
  std::vector<glm::vec3> boundary =  sampleMeshSurface(meshVerts, mtris, 8.0f, 1);
  std::cout << "boundary samples size: " << boundary.size() << "\n";
  Simulator::initSPHSimulation(N_FOR_VIS, sphSettings, boundary.data(), (int)boundary.size());
  initMesh(meshVerts,meshNormals, muvs, mtris, texPixels, texW, texH, texCh);
  updateCamera();

  initShaders(program);

  glEnable(GL_DEPTH_TEST);

  // Setup Dear ImGui context
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();

  //// Setup Dear ImGui style
  ImGui::StyleColorsDark();

  // Setup Platform/Renderer bindings
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 330");



  return true;
}
void updateMeshTransform()
{
    meshVerts = originalVerts;

    glm::vec3 lo(FLT_MAX), hi(-FLT_MAX);

    for (const auto& v : meshVerts)
    {
        lo = glm::min(lo, v);
        hi = glm::max(hi, v);
    }

    glm::vec3 center = 0.5f * (lo + hi);

    glm::vec3 d = hi - lo;

    float extent = std::max(d.x, std::max(d.y, d.z));

    float s = (extent > 0.0f)
        ? meshScale / extent
        : 1.0f;

    glm::mat4 M(1.0f);

    M = glm::translate(M, meshPosition);

    M = glm::rotate(M,
        glm::radians(meshRotation.z),
        glm::vec3(0, 0, 1));

    M = glm::rotate(M,
        glm::radians(meshRotation.y),
        glm::vec3(0, 1, 0));

    M = glm::rotate(M,
        glm::radians(meshRotation.x),
        glm::vec3(1, 0, 0));

    M = glm::scale(M, glm::vec3(s));

    for (auto& v : meshVerts)
        v = glm::vec3(M * glm::vec4(v - center, 1.0f));
}
void uploadMeshVertices()
{
    glBindBuffer(GL_ARRAY_BUFFER, meshVBO);

    glBufferSubData(
        GL_ARRAY_BUFFER,
        0,
        meshVerts.size() * sizeof(glm::vec3),
        meshVerts.data());

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}
void drawGui(int windowWidth, int windowHeight) {
    // Dear imgui new frame
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Dear imgui define
    ImVec2 minSize(300.f, 220.f);
    ImVec2 maxSize((float)windowWidth * 0.5, (float)windowHeight * 0.3);
    ImGui::SetNextWindowSizeConstraints(minSize, maxSize);

    ImGui::SetNextWindowPos(ui_hide ? ImVec2(-1000.f, -1000.f) : ImVec2(0.0f, 0.0f));

    ImGui::Begin("Control Panel", 0, windowFlags);
    ImGui::SetWindowFontScale(1);

    if (ImGui::RadioButton("Boids", currentSimulation == SimulationType::Boids)) {
        if (currentSimulation != SimulationType::Boids) {
            currentSimulation = SimulationType::Boids;
            restartSimulation();
        }
    }
    if (ImGui::RadioButton("SPH", currentSimulation == SimulationType::SPH)) {
        if (currentSimulation != SimulationType::SPH) {
            currentSimulation = SimulationType::SPH;
            restartSimulation();
        }
    }

    ImGui::Separator();
    if (currentSimulation == SimulationType::Boids) {
        bool changed = false;
        changed |= ImGui::SliderFloat("Cohesion Dist", &boidsSettings.rule1Distance, 0.0f, 20.0f);
        changed |= ImGui::SliderFloat("Separation Dist", &boidsSettings.rule2Distance, 0.0f, 20.0f);
        changed |= ImGui::SliderFloat("Alignment Dist", &boidsSettings.rule3Distance, 0.0f, 20.0f);
        changed |= ImGui::SliderFloat("Cohesion Scale", &boidsSettings.rule1Scale, 0.0f, 0.2f);
        changed |= ImGui::SliderFloat("Separation Scale", &boidsSettings.rule2Scale, 0.0f, 1.0f);
        changed |= ImGui::SliderFloat("Alignment Scale", &boidsSettings.rule3Scale, 0.0f, 1.0f);
        changed |= ImGui::SliderFloat("Max Speed", &boidsSettings.maxSpeed, 0.1f, 5.0f);
        changed |= ImGui::SliderFloat("Boundary Dist", &boidsSettings.boundaryDistance, 0.0f, 20.0f);
        changed |= ImGui::SliderFloat("Boundary Push", &boidsSettings.boundaryScale, 0.0f, 2.0f);
        if (changed) Simulator::setBoidsParams(boidsSettings);
    }
    else {
        bool changed = false;
        changed |= ImGui::SliderFloat("Rest Density", &sphSettings.restDensity, 0.01f, 1.0f);
        changed |= ImGui::SliderFloat("Gas Const", &sphSettings.gasConst, 1.0f, 1000.0f);
        changed |= ImGui::SliderFloat("Viscosity", &sphSettings.viscosity, 0.0f, 5.0f);
        changed |= ImGui::SliderFloat("Gravity", &sphSettings.gravity, -30.0f, 30.0f);
        if (changed) Simulator::setSPHParams(sphSettings);
        // Note: changing `h` needs a restart since grid cell size depends on it.
    }

    if (ImGui::Button("Restart Simulation")) restartSimulation();


    ImGui::Separator();
    bool changed = false;

    changed |= ImGui::DragFloat3(
        "Mesh Position",
        &meshPosition.x,
        0.5f);

    changed |= ImGui::DragFloat3(
        "Mesh Rotation",
        &meshRotation.x,
        1.0f);

    changed |= ImGui::DragFloat(
        "Mesh Scale",
        &meshScale,
        0.5f,
        1.0f,
        500.0f);

    if (changed)
    {
        updateMeshTransform();

        auto boundary =
            sampleMeshSurface(meshVerts, mtris, 8.0f, 1);

        Simulator::updateBoundaryParticles(boundary.data(), boundary.size());

        uploadMeshVertices();
    }

    ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}
bool loadGltf(const char* path, std::vector<glm::vec3>& verts, std::vector<glm::vec3>& normals, std::vector<glm::vec2>& uvs, std::vector<unsigned int>& indices, std::vector<unsigned char>& texPixels, int& texW, int& texH, int& texCh)
{
    tinygltf::Model model; tinygltf::TinyGLTF loader; std::string err, warn;
    bool ok = loader.LoadASCIIFromFile(&model, &err, &warn, path);
    if (!warn.empty()) std::cerr << "glTF warn: " << warn << "\n";
    if (!err.empty())  std::cerr << "glTF err:  " << err << "\n";
    if (!ok) return false;

    for (const auto& mesh : model.meshes)
        for (const auto& prim : mesh.primitives) {
            size_t base = verts.size();

            const auto& pAcc = model.accessors[prim.attributes.at("POSITION")];
            const auto& pView = model.bufferViews[pAcc.bufferView];
            const float* pData = reinterpret_cast<const float*>(
                &model.buffers[pView.buffer].data[pView.byteOffset + pAcc.byteOffset]);
            for (size_t i = 0; i < pAcc.count; i++)
                verts.emplace_back(pData[3 * i + 0], pData[3 * i + 1], pData[3 * i + 2]);

            if (prim.attributes.count("TEXCOORD_0")) {
                const auto& tAcc = model.accessors[prim.attributes.at("TEXCOORD_0")];
                const auto& tView = model.bufferViews[tAcc.bufferView];
                const float* tData = reinterpret_cast<const float*>(
                    &model.buffers[tView.buffer].data[tView.byteOffset + tAcc.byteOffset]);
                for (size_t i = 0; i < tAcc.count; i++)
                    uvs.emplace_back(tData[2 * i + 0], tData[2 * i + 1]);
            }

            // Read normals 
            if (prim.attributes.count("NORMAL")) {
                const auto& nAcc = model.accessors[prim.attributes.at("NORMAL")];
                const auto& nView = model.bufferViews[nAcc.bufferView];
                const float* nData = reinterpret_cast<const float*>(
                    &model.buffers[nView.buffer].data[nView.byteOffset + nAcc.byteOffset]);
                for (size_t i = 0; i < nAcc.count; i++)
                    normals.emplace_back(nData[3 * i + 0], nData[3 * i + 1], nData[3 * i + 2]);
            }


            else {
                uvs.resize(verts.size(), glm::vec2(0.0f));  // keep arrays aligned
            }

            const auto& iAcc = model.accessors[prim.indices];
            const auto& iView = model.bufferViews[iAcc.bufferView];
            const unsigned char* iData =
                &model.buffers[iView.buffer].data[iView.byteOffset + iAcc.byteOffset];
            for (size_t i = 0; i < iAcc.count; i++) {
                unsigned int id = (iAcc.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT)
                    ? ((const uint16_t*)iData)[i] : ((const uint32_t*)iData)[i];
                indices.push_back((unsigned int)(base + id));
            }
        }

    if (!model.images.empty()) {                 // grab the base-color image
        const auto& img = model.images[0];
        texW = img.width; texH = img.height; texCh = img.component;
        texPixels = img.image;                   // decoded by stb: 8-bit RGB/RGBA
    }
    else { texW = texH = texCh = 0; }
    return true;
}
void initVAO() {

  std::unique_ptr<GLfloat[]> bodies{ new GLfloat[4 * (N_FOR_VIS)] };
  std::unique_ptr<GLuint[]> bindices{ new GLuint[N_FOR_VIS] };

  glm::vec4 ul(-1.0, -1.0, 1.0, 1.0);
  glm::vec4 lr(1.0, 1.0, 0.0, 0.0);

  for (int i = 0; i < N_FOR_VIS; i++) {
    bodies[4 * i + 0] = 0.0f;
    bodies[4 * i + 1] = 0.0f;
    bodies[4 * i + 2] = 0.0f;
    bodies[4 * i + 3] = 1.0f;
    bindices[i] = i;
  }


  glGenVertexArrays(1, &boidVAO); // Attach everything needed to draw a particle to this
  glGenBuffers(1, &boidVBO_positions);
  glGenBuffers(1, &boidVBO_velocities);
  glGenBuffers(1, &boidIBO);

  glBindVertexArray(boidVAO);

  // Bind the positions array to the boidVAO by way of the boidVBO_positions
  glBindBuffer(GL_ARRAY_BUFFER, boidVBO_positions); // bind the buffer
  glBufferData(GL_ARRAY_BUFFER, 4 * (N_FOR_VIS) * sizeof(GLfloat), bodies.get(), GL_DYNAMIC_DRAW); // transfer data

  glEnableVertexAttribArray(positionLocation);
  glVertexAttribPointer((GLuint)positionLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);

  // Bind the velocities array to the boidVAO by way of the boidVBO_velocities
  glBindBuffer(GL_ARRAY_BUFFER, boidVBO_velocities);
  glBufferData(GL_ARRAY_BUFFER, 4 * (N_FOR_VIS) * sizeof(GLfloat), bodies.get(), GL_DYNAMIC_DRAW);
  glEnableVertexAttribArray(velocitiesLocation);
  glVertexAttribPointer((GLuint)velocitiesLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, boidIBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, (N_FOR_VIS) * sizeof(GLuint), bindices.get(), GL_STATIC_DRAW);

  glBindVertexArray(0);
}
void initMesh(const std::vector<glm::vec3>& verts, const std::vector<glm::vec3>&  meshNormals,
    const std::vector<glm::vec2>& uvs,
    const std::vector<unsigned int>& indices,
    const std::vector<unsigned char>& texPixels, int texW, int texH, int texCh)
{
    meshIndexCount = (GLsizei)indices.size();

    static const char* meshAttributeLocations[] = { "a_pos", "a_normal", "a_uv" };
    meshProgram = glslUtility::createProgram(
        "shaders/mesh.vert.glsl",
        "shaders/mesh.frag.glsl",
        meshAttributeLocations, 3);

    meshProjLoc = glGetUniformLocation(meshProgram, "u_projMatrix");
    glUseProgram(meshProgram);
    glUniform1f(glGetUniformLocation(meshProgram, "u_scale"), -1.0f / SCENE_SCALE);
    glUniform1i(glGetUniformLocation(meshProgram, "u_tex"), 0);
    glUseProgram(0);

    glGenVertexArrays(1, &meshVAO);
    glBindVertexArray(meshVAO);

    glGenBuffers(1, &meshVBO);
    glBindBuffer(GL_ARRAY_BUFFER, meshVBO);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(glm::vec3), verts.data(), GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);

    GLuint meshNormalVBO;  
    glGenBuffers(1, &meshNormalVBO);
    glBindBuffer(GL_ARRAY_BUFFER, meshNormalVBO);
    glBufferData(GL_ARRAY_BUFFER, meshNormals.size() * sizeof(glm::vec3), meshNormals.data(), GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);

    GLuint meshUVBO;
    glGenBuffers(1, &meshUVBO);
    glBindBuffer(GL_ARRAY_BUFFER, meshUVBO);
    glBufferData(GL_ARRAY_BUFFER, uvs.size() * sizeof(glm::vec2), uvs.data(), GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void*)0);

    glGenBuffers(1, &meshIBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, meshIBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int),
        indices.data(), GL_STATIC_DRAW);
    glBindVertexArray(0);

    if (texW > 0 && texH > 0 && !texPixels.empty()) {
        glGenTextures(1, &meshTex);
        glBindTexture(GL_TEXTURE_2D, meshTex);
        GLenum fmt = (texCh == 4) ? GL_RGBA : (texCh == 3 ? GL_RGB : GL_RED);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);  // RGB rows aren't 4-aligned
        glTexImage2D(GL_TEXTURE_2D, 0, fmt, texW, texH, 0, fmt, GL_UNSIGNED_BYTE, texPixels.data());
        glGenerateMipmap(GL_TEXTURE_2D);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
}
std::vector<glm::vec3> sampleMeshSurface(const std::vector<glm::vec3>& verts,  std::vector<unsigned int>& indices, float spacing, int layers)
{
    // first we obtain a mesh, then we sample some of its vertices and save these positions. these samples are then added to the particles Buffer and sent to the GPU, now the collision avoidance is simplified. since the mesh is already covered
    // by particles, the pressure computation will take care of ensuring the particles dont go through the mesh. that does require the sampling to be high enough = h.
    // these particles also have to be treated differently since their positions shouldnt change. Unless the mesh moves ofcourse. which would be nice to see. 
    std::vector<glm::vec3> samples;
    for (size_t t = 0; t + 2 < indices.size(); t += 3) {
        glm::vec3 a = verts[indices[t]], b = verts[indices[t + 1]], c = verts[indices[t + 2]];
        glm::vec3 nrm = glm::normalize(glm::cross(b - a, c - a)); 
        float maxEdge = std::max({ glm::length(b - a), glm::length(c - a), glm::length(c - b) });
        int n = std::max(1, (int)std::ceil(maxEdge / spacing));

        for (int i = 0; i <= n; i+=2)
            for (int j = 0; j <= n - i; j+=2) {
                float u = (float)i / n, v = (float)j / n, w = 1.0f - u - v;
                glm::vec3 p = u * a + v * b + w * c;
                for (int L = 0; L < layers; L++)
                    samples.push_back(p - nrm * (L * spacing)); // extra layers behind the surface
            }
    }
    return samples;
}
void initShaders(GLuint * program) {
  GLint location;

  program[PROG_BOID] = glslUtility::createProgram(
    "shaders/boid.vert.glsl",
    "shaders/boid.geom.glsl",
    "shaders/boid.frag.glsl", attributeLocations, 2);
    glUseProgram(program[PROG_BOID]);

    if ((location = glGetUniformLocation(program[PROG_BOID], "u_projMatrix")) != -1) {
      glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
    }
    if ((location = glGetUniformLocation(program[PROG_BOID], "u_cameraPos")) != -1) {
      glUniform3fv(location, 1, &cameraPosition[0]);
    }
  }

  //====================================
  // Main loop
  //====================================
  void runCUDA() {
    // Map OpenGL buffer object for writing from CUDA on a single GPU
    // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not
    // use this buffer

    float4 *dptr = NULL;
    float *dptrVertPositions = NULL;
    float *dptrVertVelocities = NULL;

    cudaGLMapBufferObject((void**)&dptrVertPositions, boidVBO_positions);
    cudaGLMapBufferObject((void**)&dptrVertVelocities, boidVBO_velocities);

  
    switch (currentSimulation) {
    case SimulationType::Boids:
        Simulator::stepBoidsSimulationCoherentGrid(boidsSettings.dt);
        break;
    case SimulationType::SPH:
        Simulator::stepSPHSimulationCoherentGrid(sphSettings.dt);
        break;
    }

    #if VISUALIZE
        Simulator::copyToVBO(dptrVertPositions, dptrVertVelocities);
    #endif
    // unmap buffer object
    cudaGLUnmapBufferObject(boidVBO_positions);
    cudaGLUnmapBufferObject(boidVBO_velocities);
  }

  void mainLoop() {
    double fps = 0;
    double timebase = 0;
    int frame = 0;
        
    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();

      frame++;
      double time = glfwGetTime();

      if (time - timebase > 1.0) {
        fps = frame / (time - timebase);
        timebase = time;
        frame = 0;
      }

      runCUDA();

      std::ostringstream ss;
      ss << "[";
      ss.precision(1);
      ss << std::fixed << fps;
      ss << " fps] " << deviceName;
      glfwSetWindowTitle(window, ss.str().c_str());

      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      #if VISUALIZE
      glUseProgram(program[PROG_BOID]);
      glBindVertexArray(boidVAO);
      glPointSize((GLfloat)pointSize);
      glDrawElements(GL_POINTS, N_FOR_VIS + 1, GL_UNSIGNED_INT, 0);
      glPointSize(1.0f);

      glUseProgram(0);
      glBindVertexArray(0);

      glUseProgram(meshProgram);
      glUniformMatrix4fv(meshProjLoc, 1, GL_FALSE, &projection[0][0]);
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, meshTex);
      glBindVertexArray(meshVAO);
      glDrawElements(GL_TRIANGLES, meshIndexCount, GL_UNSIGNED_INT, 0);
      glBindVertexArray(0);
      glUseProgram(0);


      // Draw imgui
      int display_w, display_h;
      glfwGetFramebufferSize(window, &display_w, &display_h);
      drawGui(display_w, display_h);


      glfwSwapBuffers(window);
      #endif
    }
    glfwDestroyWindow(window);
    glfwTerminate();
  }

  void restartSimulation() {
      Simulator::endSimulation();
      auto boundary = sampleMeshSurface(meshVerts, mtris, 8.0f, 1);
      if (currentSimulation == SimulationType::Boids)
          Simulator::initBoidsSimulation(N_FOR_VIS, boidsSettings, boundary.data(), (int)boundary.size());
      else
          Simulator::initSPHSimulation(N_FOR_VIS, sphSettings, boundary.data(), (int)boundary.size());
  }
  void errorCallback(int error, const char *description) {
    fprintf(stderr, "error %d: %s\n", error, description);
  }

  void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
      glfwSetWindowShouldClose(window, GL_TRUE);
    }
  }

  void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    if (ImGui::GetIO().WantCaptureMouse) return;
    leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
    rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
  }

  void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
    if (leftMousePressed) {
      // compute new camera parameters
      phi += (xpos - lastX) / width;
      theta -= (ypos - lastY) / height;
      theta = std::fmax(0.01f, std::fmin(theta, 3.14f));
      updateCamera();
    }
    else if (rightMousePressed) {
      zoom += (ypos - lastY) / height;
      zoom = std::fmax(0.1f, std::fmin(zoom, 5.0f));
      updateCamera();
    }

	lastX = xpos;
	lastY = ypos;
  }

  void updateCamera() {
    cameraPosition.x = zoom * sin(phi) * sin(theta);
    cameraPosition.z = zoom * cos(theta);
    cameraPosition.y = zoom * cos(phi) * sin(theta);
    cameraPosition += lookAt;

    projection = glm::perspective(fovy, float(width) / float(height), zNear, zFar);
    glm::mat4 view = glm::lookAt(cameraPosition, lookAt, glm::vec3(0, 0, 1));
    projection = projection * view;

    GLint location;

    glUseProgram(program[PROG_BOID]);
    if ((location = glGetUniformLocation(program[PROG_BOID], "u_projMatrix")) != -1) {
      glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
    }
  }
