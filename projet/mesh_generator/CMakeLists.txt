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
add_test(plan_test ${EXECUTABLE_OUTPUT_PATH}/${PROJECT_NAME} plan_test)