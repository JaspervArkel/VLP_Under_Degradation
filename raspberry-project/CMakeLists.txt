# == DO NOT EDIT THE FOLLOWING LINES for the Raspberry Pi Pico VS Code Extension to work ==
if(WIN32)
    set(USERHOME $ENV{USERPROFILE})
else()
    set(USERHOME $ENV{HOME})
endif()
set(sdkVersion 2.2.0)
set(toolchainVersion 14_2_Rel1)
set(picotoolVersion 2.2.0-a4)
set(picoVscode ${USERHOME}/.pico-sdk/cmake/pico-vscode.cmake)
if (EXISTS ${picoVscode})
    include(${picoVscode})
endif()
# ====================================================================================

set(PICO_BOARD pico CACHE STRING "Board type")

cmake_minimum_required(VERSION 3.13)

include($ENV{PICO_SDK_PATH}/external/pico_sdk_import.cmake)

project(pico_mlp_inference C CXX ASM)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

pico_sdk_init()

add_subdirectory(
    "E:/Users/jaspe/OneDrive/Documents/GitHub/pico-tflmicro"
    "pico-tflmicro-build"
    EXCLUDE_FROM_ALL
)

add_executable(pico_mlp_inference
    main.cpp
    model_data.cc
)

target_include_directories(pico_mlp_inference PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    "E:/Users/jaspe/OneDrive/Documents/GitHub/pico-tflmicro/src"
)

target_link_libraries(pico_mlp_inference
    pico_stdlib
    pico-tflmicro
)

pico_enable_stdio_usb(pico_mlp_inference 1)
pico_enable_stdio_uart(pico_mlp_inference 0)

pico_add_extra_outputs(pico_mlp_inference)