project(my_guitar)

file(GLOB MY_GUITAR_SRC "./*.cpp")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}")

add_executable(${PROJECT_NAME} ${MY_GUITAR_SRC})
target_include_directories(${PROJECT_NAME} PRIVATE "./")
target_link_libraries(${PROJECT_NAME} PUBLIC stk)
