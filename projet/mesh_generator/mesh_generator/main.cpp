#include <iostream>
#include "Point3D.h"
#include "Polygon.h"
#include "Vector.h"
#include "Plan.h"
#include "Triangle.h"

#include <vector>

using namespace std;
int main() {

  Point3D a(1., 0., 0.);
  Point3D b(0., 0., 0.);
  Point3D c(0., 0., 1.);
  Point3D d(0., 1., 1.);

  try
  {

    std::vector<Point3D> pts;
    pts.reserve(3);

    pts.push_back(a);
    pts.push_back(b);
    pts.push_back(c);

    
    Triangle tri(pts);
    cout << tri.plan_support(c) << tri.plan_support(tri.center()) << tri.plan_support(d) << endl;
  }
  catch (const char* msg) {
    cerr << msg << endl;
  }
  system("pause");
  return 0;
}