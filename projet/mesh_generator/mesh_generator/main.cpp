#include <iostream>
#include "Point3D.h"
#include "Polygon.h"
#include "Vector.h"
#include "Plan.h"
#include "Triangle.h"
#include "Mesh.h"
#include "SimpleShapeGenerator.h"

#include <vector>

using namespace std;
int main() {



  try
  {

	  /*
	  Mesh mesh = GenerateSphere(2, 3, 3);

	  for(size_t i = 0; i < mesh.nb_triangles(); i++){
		  std::cout << "the " << i << "th triangle:" << std::endl;
		  mesh.get_triangle(i).get_point(0).print_position();

		  mesh.get_triangle(i).get_point(1).print_position();

		  mesh.get_triangle(i).get_point(2).print_position();
	  }*/
	  CubeGenerator generator(2.);
	  Mesh mesh = generator.run();
	  std::cout << mesh.is_closed_2() << endl;

	  SphereGenerator generator2(2., 5, 5);
	  Mesh mesh2 = generator2.run();
	  std::cout << mesh2.is_closed_2() << endl;

  }
  catch (const char* msg) {
    cerr << msg << endl;
  }
  system("pause");
  return 0;
}