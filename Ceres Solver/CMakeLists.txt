cmake_minimum_required(VERSION 3.10)
project(Reconstruction)

# Find the required packages
find_package(Ceres REQUIRED)
find_package(OpenSSL REQUIRED)  # Add this line
find_package(Boost 1.83 REQUIRED COMPONENTS system filesystem)

# If Boost is still not found, explicitly set the Boost directories
if(NOT Boost_FOUND)
    message(FATAL_ERROR "Boost not found")
else()
    message(STATUS "Boost found at ${Boost_INCLUDE_DIRS}")
    message(STATUS "Boost libraries: ${Boost_LIBRARIES}")
endif()

# Create the executable
add_executable(reconstruction main.cpp)

# Link the required libraries
target_link_libraries(reconstruction
  Ceres::ceres
  Boost::system
  Boost::filesystem
  OpenSSL::SSL
  OpenSSL::Crypto
)