#include "Plan.h"
#include <vector>
#include "Vector.h"
#include "Point3D.h"

#include "catch.h"



TEST_CASE("plan_test")
{
  Point3D pt(1., 1., 1.);
  Point3D pt2(1., 1., 2.);
  Vector norm = pt2 - pt;
  Plan plan(pt, norm);
  REQUIRE(true == plan.is_in_plane(pt));
  REQUIRE(false== plan.is_in_plane(pt2));
  
}
