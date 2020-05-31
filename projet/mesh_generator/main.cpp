#define CATCH_CONFIG_RUNNER
#define CATCH_CONFIG_COUNTER
#include "catch.h"

#include "test_Mesh.h"
#include "test_Plan_is_in_plan.h"
#include "test_Polygon_area.h"
#include "test_simplegenerator.h"
#include "test_triangle_plan_support.h"

////////////////////////////////////////////////////////////////////////////////////
// main
////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char * argv[])
{
  std::cout << "Starting test" << std::endl;
  int result = Catch::Session().run(argc, argv);

  return 0;
}
