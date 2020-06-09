#include "catch.h"
#include "MeshLoader.h"
#include "Mesh.h"
#include<vector>
#include "SimpleShapeGenerator.h"
#include <iostream>  
#include <fstream> 


TEST_CASE("test_loader")
{
	MeshLoader loader("E:/3ds_intern/projet/mesh_generator/TestData/bunny.obj");
	Mesh mesh = loader.load();
	REQUIRE( mesh.nb_triangles()>4000);
}

TEST_CASE("test_write")
{
	SphereGenerator g(4., 6, 6);
	Mesh mesh = g.run();
	MeshWriter writer("E:/3ds_intern/projet/mesh_generator/TestResults/sphere.obj");
	writer.write(mesh);
	
	MeshLoader loader("E:/3ds_intern/projet/mesh_generator/TestResults/sphere.obj");
	Mesh mesh2 = loader.load();
	REQUIRE(mesh2.is_closed_2());
}

