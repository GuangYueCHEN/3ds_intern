cmake_minimum_required (VERSION 2.8.11 FATAL_ERROR)
project (mesh_generator)

set(SHARED_INTERFACES ${CMAKE_CURRENT_SOURCE_DIR}/Interfaces)
enable_testing()
include_directories(${SHARED_INTERFACES})
include_directories(${CMAKE_EXPORTED_HDD})

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(TheSlash "/")
set(ThePath ${CMAKE_CURRENT_SOURCE_DIR})
if (MSVC)
	STRING(REGEX REPLACE "/" "\\\\\\\\" ThePath ${ThePath})
	STRING(REGEX REPLACE "/" "\\\\\\\\" TheSlash ${TheSlash})
else(MSVC)
	set(CMAKE_C_FLAGS "-Wall -Wextra -pedantic")
endif(MSVC)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/built/bin)
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/built/bin )
foreach( OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES} )
    string( TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG )
    set( CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_SOURCE_DIR}/built/bin )
endforeach( OUTPUTCONFIG CMAKE_CONFIGURATION_TYPES )
# recherche des lib externes
# pour openscenegraph
set(InputDirectory "${ThePath}${TheSlash}TestData")
set(OutputDirectory "${ThePath}${TheSlash}TestResults")

file( GLOB mesh_generator_src ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file( GLOB mesh_generator_hdd ${CMAKE_CURRENT_SOURCE_DIR}/*.h)

source_group(Src FILES ${mesh_generator_src})
source_group(Headers FILES ${mesh_generator_hdd})


add_definitions("-DTheSlash=\"${TheSlash}\"")
add_definitions("-DInputDirectory=\"${InputDirectory}\"")
add_definitions("-DOutputDirectory=\"${OutputDirectory}\"")

file(MAKE_DIRECTORY ${OutputDirectory})

add_executable(mesh_generator ${mesh_generator_src} ${mesh_generator_hdd} ${SHARED_INTERFACES})

add_test(mesh_test_is_closed ${EXECUTABLE_OUTPUT_PATH}/${PROJECT_NAME} mesh_test_is_closed)
add_test(mesh_test_is_closed_2 ${EXECUTABLE_OUTPUT_PATH}/${PROJECT_NAME} mesh_test_is_closed_2)
add_test(mesh_test_volume ${EXECUTABLE_OUTPUT_PATH}/${PROJECT_NAME} mesh_test_volume)
add_test(mesh_test_point_position ${EXECUTABLE_OUTPUT_PATH}/${PROJECT_NAME} mesh_test_point_position)
add_test(point_distance_with ${EXECUTABLE_OUTPUT_PATH}/${PROJECT_NAME} point_distance_with)
add_test(polygon_area ${EXECUTABLE_OUTPUT_PATH}/${PROJECT_NAME} polygon_area)
add_test(plan_test ${EXECUTABLE_OUTPUT_PATH}/${PROJECT_NAME} plan_test)
add_test(generate_cube ${EXECUTABLE_OUTPUT_PATH}/${PROJECT_NAME} generate_cube)
add_test(generate_sphere ${EXECUTABLE_OUTPUT_PATH}/${PROJECT_NAME} generate_sphere)
add_test(triangle_plan_support ${EXECUTABLE_OUTPUT_PATH}/${PROJECT_NAME} triangle_plan_support)
add_test(test_loader ${EXECUTABLE_OUTPUT_PATH}/${PROJECT_NAME} test_loader)
add_test(test_write ${EXECUTABLE_OUTPUT_PATH}/${PROJECT_NAME} test_write)
add_test(mesh_test_augmentation ${EXECUTABLE_OUTPUT_PATH}/${PROJECT_NAME} mesh_test_augmentation)

add_test(mesh_add_point ${EXECUTABLE_OUTPUT_PATH}/${PROJECT_NAME} mesh_add_point)
add_test(mesh_add_triangle ${EXECUTABLE_OUTPUT_PATH}/${PROJECT_NAME} mesh_add_triangle)
add_test(mesh_get_triangles ${EXECUTABLE_OUTPUT_PATH}/${PROJECT_NAME} mesh_get_triangles)
add_test(mesh_split_collapse ${EXECUTABLE_OUTPUT_PATH}/${PROJECT_NAME} mesh_split_collapse)
add_test(mesh_flip_edge ${EXECUTABLE_OUTPUT_PATH}/${PROJECT_NAME} mesh_flip_edge)
add_test(test_mesh_smoothing ${EXECUTABLE_OUTPUT_PATH}/${PROJECT_NAME} test_mesh_smoothing)

