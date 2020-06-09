#include "catch.h"
#include "MeshLoader.h"
#include "Mesh.h"
#include<vector>
#include "SimpleShapeGenerator.h"
#include <iostream>  
#include <fstream> 

std::string InputPath{ InputDirectory };
std::string Slash{ TheSlash };
std::string Outpath{ OutputDirectory };

TEST_CASE("test_loader")
{
	MeshLoader loader(InputPath  + Slash + "bunny.obj");
	Mesh mesh = loader.load();
	REQUIRE( mesh.nb_triangles()>4000);
}

TEST_CASE("test_write")
{
	SphereGenerator g(4., 6, 6);
	Mesh mesh = g.run();
	MeshWriter writer(OutputDirectory + Slash + "sphere.obj");
	writer.write(mesh);
	
	MeshLoader loader(OutputDirectory + Slash + "sphere.obj");
	Mesh mesh2 = loader.load();
	REQUIRE(mesh2.is_closed_2());
}

