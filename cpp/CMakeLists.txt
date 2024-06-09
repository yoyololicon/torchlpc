cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(torchlpc)

find_package(Torch REQUIRED)

# Define our library target
add_library(torchlpc SHARED torchlpc.cpp)
# Enable C++14
target_compile_features(torchlpc PRIVATE cxx_std_14)
# Link against LibTorch
target_link_libraries(torchlpc "${TORCH_LIBRARIES}")
