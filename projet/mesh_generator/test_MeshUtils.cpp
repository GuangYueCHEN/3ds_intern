#include "test_MeshUtils.h"

Mesh generate_mesh()
{
  
  Point3D a(1., 0., 0.);
  Point3D b(0., 1., 0.);
  Point3D c(0., 0., 1.);
  Point3D d(-1., -1., -1.);

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

  return Mesh(tris);
}
