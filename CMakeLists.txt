cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(rte)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3")

add_executable(rte)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(rte PUBLIC OpenMP::OpenMP_CXX)
endif()

add_subdirectory(third_party/glm)
target_link_libraries(rte PUBLIC glm::glm)

target_sources(rte PRIVATE
    src/main.cc
    src/sphere.cc
    src/scene.cc
    src/object.cc
    src/camera.cc
    src/materials/diffuse.cc
    src/materials/metal.cc
    src/materials/dielectric.cc
    src/materials/diffuse_light.cc
    src/materials/material_base.cc
    )

