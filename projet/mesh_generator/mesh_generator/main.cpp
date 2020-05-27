#include <iostream>
#include "Point3D.h"
#include "Polygon.h"
#include "Vector.h"
#include "Plan.h"
#include "Triangle.h"
#include "Mesh.h"

#include <vector>

using namespace std;
int main() {

  Point3D a(1., 0., 0.);
  Point3D b(0., 1., 0.);
  Point3D c(0., 0., 1.);
  Point3D d(0., 0., 0.);
  Point3D in(0.1, 0.1, 0.1);
  Point3D out(-1, -1, -1);

  try
  {

    std::vector<Point3D> pts;
	std::vector<Triangle> tris;
    pts.reserve(3);
    pts.push_back(a);
    pts.push_back(b);
    pts.push_back(c);
	Triangle tri(pts);
	tris.push_back(tri);
	pts.clear();

	pts.push_back(d);
	pts.push_back(b);
	pts.push_back(c);
	tri = Triangle(pts);
	tris.push_back(tri);
	pts.clear();

	pts.push_back(a);
	pts.push_back(b);
	pts.push_back(d);
	tri = Triangle(pts);
	tris.push_back(tri);
	pts.clear();

	pts.push_back(a);
	pts.push_back(d);
	pts.push_back(c);
	tri = Triangle(pts);
	tris.push_back(tri);

	cout << "construct mesh model" << endl;
	Mesh mesh(tris);
	for (const auto  & tri: tris) {
		tri.get_point(0).print_position();
		tri.get_point(1).print_position();
		tri.get_point(2).print_position();
	}


	cout << "remeshing model" << endl;
	Mesh remesh = mesh.remeshing(d, out);
	for (size_t i = 0; i < mesh.nb_triangles(); i++)
	{
		Triangle tri(remesh.get_triangle(i));
		tri.get_point(0).print_position();
		tri.get_point(1).print_position();
		tri.get_point(2).print_position();
	}

	cout << "testing if model is closed" << endl;
  cout << mesh.is_closed() << endl;

  bool b = mesh.is_closed_2();

	cout << "mesh volume" << endl;
	cout << mesh.volume() << endl;

	cout << "remesh volume" << endl;
	cout << remesh.volume() << endl;

	cout << "testing inner point position" << endl;
	cout << mesh.point_position(in) << endl;

	cout << "testing outside point position" << endl;
	cout << mesh.point_position(out) << endl;

	cout << "testing onsurface point position" << endl;
	cout << mesh.point_position(d) << endl;
  }
  catch (const char* msg) {
    cerr << msg << endl;
  }
  system("pause");
  return 0;
}