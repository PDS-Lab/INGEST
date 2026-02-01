cmake_minimum_required(VERSION 3.15...4.0)
project(
  ranns
  VERSION 1.0
  LANGUAGES CXX
)

set(${PROJECT_NAME}_ENABLE_APPS "AUTO" CACHE STRING "Enable building applications (AUTO/ON/OFF)")
set_property(CACHE ${PROJECT_NAME}_ENABLE_APPS PROPERTY STRINGS "AUTO" "ON" "OFF")
option(${PROJECT_NAME}_ENABLE_TESTS "Enable tests" ON)

if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(
    FATAL_ERROR
      "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there."
  )
endif()

set(${PROJECT_NAME}_EXPORT_LINK_LIBS "")
set(${PROJECT_NAME}_EXPORT_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(${PROJECT_NAME}_EXPORT_OPTIONS "")
set(${PROJECT_NAME}_PCH_HEADERS "")

list(APPEND ${PROJECT_NAME}_EXPORT_OPTIONS "-ffast-math")

include(cmake/CPM.cmake)
find_package(PkgConfig REQUIRED)

macro(find_or_download DEP_NAME FAILBACK)
    if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/external/${DEP_NAME}")
        CPMAddPackage(NAME ${DEP_NAME} SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/external/${DEP_NAME})
    elseif(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${DEP_NAME}")
        CPMAddPackage(NAME ${DEP_NAME} SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/${DEP_NAME})
    elseif(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../${DEP_NAME}")
        CPMAddPackage(NAME ${DEP_NAME} SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/../${DEP_NAME})
    else()
        CPMAddPackage(${FAILBACK})
    endif()
endmacro()

find_or_download(myutils "gh:eiclpy/myutils")
list(APPEND ${PROJECT_NAME}_EXPORT_LINK_LIBS myutils)

find_or_download(lcoro "gh:eiclpy/lcoro")
list(APPEND ${PROJECT_NAME}_EXPORT_LINK_LIBS lcoro)

CPMAddPackage("gh:martinus/unordered_dense@4.8.1")
list(APPEND ${PROJECT_NAME}_EXPORT_LINK_LIBS unordered_dense)
list(APPEND ${PROJECT_NAME}_EXPORT_PCH_HEADERS <ankerl/unordered_dense.h>)

CPMAddPackage("gh:microsoft/mimalloc@3.1.5")
list(APPEND ${PROJECT_NAME}_EXPORT_LINK_LIBS mimalloc-static)

CPMAddPackage(
  NAME Eigen3
  GITLAB_REPOSITORY libeigen/eigen
  GIT_TAG 3.4
  OPTIONS
    "EIGEN_BUILD_TESTING OFF"
    "EIGEN_BUILT_BTL OFF"
    "EIGEN_BUILD_SPBENCH OFF"
    "EIGEN_BUILD_DOC OFF"
    "EIGEN_BUILD_DEMOS OFF"
)
list(APPEND ${PROJECT_NAME}_EXPORT_LINK_LIBS Eigen3::Eigen)

find_package(OpenMP REQUIRED)
list(APPEND ${PROJECT_NAME}_EXPORT_LINK_LIBS OpenMP::OpenMP_CXX)

file(GLOB_RECURSE ${PROJECT_NAME}_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h")
file(GLOB_RECURSE ${PROJECT_NAME}_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc")

add_library(${PROJECT_NAME} ${${PROJECT_NAME}_HEADERS} ${${PROJECT_NAME}_SOURCES})
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_23)
target_compile_options(${PROJECT_NAME} PUBLIC ${${PROJECT_NAME}_EXPORT_OPTIONS})
target_include_directories(${PROJECT_NAME} PUBLIC ${${PROJECT_NAME}_EXPORT_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PUBLIC ${${PROJECT_NAME}_EXPORT_LINK_LIBS})
target_precompile_headers(${PROJECT_NAME} PUBLIC ${${PROJECT_NAME}_EXPORT_PCH_HEADERS})

if((CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND NOT ${PROJECT_NAME}_ENABLE_APPS STREQUAL "OFF") OR 
   (NOT CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND ${PROJECT_NAME}_ENABLE_APPS STREQUAL "ON"))
    file(GLOB ${PROJECT_NAME}_BINS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
    foreach(BIN_FILE ${${PROJECT_NAME}_BINS})
        get_filename_component(TARGET_NAME ${BIN_FILE} NAME_WE)
        set(EXE_NAME "${TARGET_NAME}.exe")

        message(STATUS "Building executable: ${EXE_NAME} from ${BIN_FILE}")
        add_executable(${EXE_NAME} ${BIN_FILE})
        target_link_libraries(${EXE_NAME} PRIVATE ${PROJECT_NAME})
        target_precompile_headers(${EXE_NAME} REUSE_FROM ${PROJECT_NAME})
    endforeach()
endif()
