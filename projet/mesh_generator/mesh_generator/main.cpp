#include <iostream>
#include "Point3D.h"
#include "Polygon.h"
#include "Vector.h"
#include "Plan.h"
#include "Triangle.h"
using namespace std;
int main() {

	Point3D a(1., 0., 0.);
	Point3D b(0., 0., 0.);
	Point3D c(0., 0., 1.);
	Point3D d(0., 1., 1.);
	try {
	//Plan plan(a, b, c);
	//cout << plan.norm().get_x() << plan.norm().get_y() << plan.normal().get_z() << endl << plan.plan_support(d)<< endl << plan.plan_support(c);
	Point3D pts[3] = { a, b, c };
	//Polygon pol(pts,3);
	//pol.center().print_position() ;
	//cout << pol.area() << endl;
	Triangle tri(pts, 3);
	cout << tri.on_triangle(c) << tri.on_triangle(tri.center()) << tri.on_triangle(d) << endl;
	}catch (const char* msg) {
		cerr << msg << endl;
	}
	system("pause");
	return 0;
}