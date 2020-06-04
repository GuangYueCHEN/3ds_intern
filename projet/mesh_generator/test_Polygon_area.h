#include "catch.h"
#include "Point3D.h"
#include "Triangle.h"
#include<vector>

TEST_CASE("polygon_area")
{
	std::vector<Point3D> pts;
	pts.reserve(3);
	pts.push_back(Point3D());
	pts.push_back(Point3D(1., 1., 0.));
	pts.push_back(Point3D(1., 0., 0.));

	Triangle tri(pts);

	REQUIRE(0.5 == tri.area());
}