#include "catch.h"
#include "SimpleShapeGenerator.h"
#include "Mesh.h"
#include<vector>

TEST_CASE("generate_cube")
{
	CubeGenerator g(2.);
	Mesh mesh = g.run();
	REQUIRE(true == mesh.is_closed_2());
}


TEST_CASE("generate_sphere")
{
	SphereGenerator g(2., 3, 3);
	Mesh mesh = g.run();
	REQUIRE(true == mesh.is_closed_2());
}