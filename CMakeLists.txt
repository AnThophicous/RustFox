cmake_minimum_required(VERSION 3.16)
project(rustfox VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)

option(FOX_NATIVE      "Tune for the building machine"   ON)
option(FOX_WERROR      "Treat warnings as errors"        OFF)
option(FOX_BUILD_TESTS "Build the test suite"            ON)
option(FOX_STATIC      "Link the CLI fully statically"   OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(FOX_SOURCES
    src/platform.c
    src/platform_posix.c
    src/platform_win32.c
    src/cpuid.c
    src/log.c
    src/sysprobe.c
    src/pressure.c
    src/governor.c
    src/gguf.c
    src/planner.c
    src/ternary.c
    src/tokenizer.c
    src/threadpool.c
    src/gemv.c
    src/weights.c
    src/kernels.c
    src/ops.c
    src/model.c
    src/forward.c
    src/sample.c
)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_library(foxcore STATIC ${FOX_SOURCES})

target_include_directories(foxcore
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if(UNIX)
    target_compile_definitions(foxcore PRIVATE _GNU_SOURCE)
endif()

if(MSVC)
    target_compile_options(foxcore PRIVATE /W4 /D_CRT_SECURE_NO_WARNINGS)
    if(FOX_WERROR)
        target_compile_options(foxcore PRIVATE /WX)
    endif()
else()
    target_compile_options(foxcore PRIVATE
        -Wall -Wextra -Wshadow -Wpointer-arith -Wvla -Wstrict-prototypes)
    if(FOX_WERROR)
        target_compile_options(foxcore PRIVATE -Werror)
    endif()
endif()

if(FOX_NATIVE AND NOT MSVC)
    include(CheckCCompilerFlag)
    check_c_compiler_flag("-march=native" FOX_HAS_MARCH_NATIVE)
    if(FOX_HAS_MARCH_NATIVE)
        target_compile_options(foxcore PUBLIC -march=native)
        message(STATUS "fox: tuning for this machine (-march=native). "
                       "The binary will not be portable. Pass -DFOX_NATIVE=OFF "
                       "to build something you can copy elsewhere.")
    endif()
endif()

target_link_libraries(foxcore PUBLIC Threads::Threads)

if(UNIX AND NOT APPLE)
    target_link_libraries(foxcore PUBLIC m)
endif()

find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" describe --tags --always --dirty
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        OUTPUT_VARIABLE FOX_GIT_DESCRIBE
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(FOX_GIT_DESCRIBE)
        target_compile_definitions(foxcore PRIVATE
            FOX_GIT_DESCRIBE="${PROJECT_VERSION}-${FOX_GIT_DESCRIBE}")
    endif()
endif()

add_executable(fox src/cli.c)
target_link_libraries(fox PRIVATE foxcore)
target_include_directories(fox PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
if(UNIX)
    target_compile_definitions(fox PRIVATE _GNU_SOURCE)
endif()

if(FOX_STATIC AND NOT MSVC AND NOT APPLE)
    target_link_options(fox PRIVATE -static)
endif()

if(FOX_BUILD_TESTS)
    enable_testing()
    set(FOX_TESTS test_probe test_governor test_gguf test_gguf_arrays test_planner
                  test_ternary test_gemv test_weights test_tokenizer test_kernels test_forward)
    foreach(t IN LISTS FOX_TESTS)
        add_executable(${t} tests/${t}.c)
        target_link_libraries(${t} PRIVATE foxcore)
        target_include_directories(${t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
        if(UNIX)
            target_compile_definitions(${t} PRIVATE _GNU_SOURCE)
        endif()
        add_test(NAME ${t} COMMAND ${t})
    endforeach()
endif()

include(GNUInstallDirs)
install(TARGETS fox foxcore
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/fox DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
