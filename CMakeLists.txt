set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "Força compatibilidade com deps antigas")

cmake_minimum_required(VERSION 3.24)
project(FlappyBird)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
    SFML
    GIT_REPOSITORY https://github.com/SFML/SFML.git
    GIT_TAG 3.0.0
)

set(SFML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SFML_BUILD_TEST_SUITE OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(SFML)

file(GLOB SOURCES CONFIGURE_DEPENDS
    "main.cpp"
    "objects/src/*.cpp"
    "core/src/World.cpp"
    "population/src/*.cpp"
)

add_executable(FlappyBird ${SOURCES})

target_include_directories(FlappyBird PRIVATE 
    "${CMAKE_CURRENT_SOURCE_DIR}/objects/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/population/include"

)

set(FONT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/fonts/FontAwesome6_Regular.ttf")

add_custom_command(TARGET FlappyBird POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${FONT_FILE}"
    "$<TARGET_FILE_DIR:FlappyBird>/FontAwesome6_Regular.ttf"
)

target_link_libraries(FlappyBird PRIVATE SFML::Graphics)
