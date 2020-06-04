#include "catch.h"
#include "Point3D.h"

TEST_CASE("point_distance_with")
{
	Point3D pt1();
	Point3D pt2(1.,1.,1.)

	REQUIRE(3. == pt1.distance_with(pt2));
}