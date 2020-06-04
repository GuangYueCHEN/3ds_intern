#include "catch.h"
#include "Point3D.h"
#include "Triangle.h"
#include<vector>

TEST_CASE("triangle_plan_support")
{
	std::vector<Point3D> pts;
	pts.reserve(3);
	pts.push_back(Point3D());
	pts.push_back(Point3D(1., 1., 0.));
	pts.push_back(Point3D(1., 0., 0.));

	Triangle tri(pts);

	REQUIRE(true == tri.plan_support(Point3D(1., 0., 0.)) );
	REQUIRE(true == tri.plan_support(Point3D(0.2, 0.2, 0.)));
	REQUIRE(false == tri.plan_support(Point3D(0.2, 0.2, 1.)));
	REQUIRE(false == tri.plan_support(Point3D(1.2, 1.2, 0.)));
}