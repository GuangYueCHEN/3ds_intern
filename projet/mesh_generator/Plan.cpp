#include"Plan.h"
#include "Vector.h"
#include "constant.h"

Plan::Plan(const Point3D & pt1, const Point3D & pt2, const Point3D & pt3)
{
  Vector vec1(pt1, pt2);
  Vector vec2(pt3, pt2);
  Vector norm = vec1 ^ vec2;
  if (norm.norme_infini() == 0) 
  {
    throw "error: can not build a plane";
  }
  else 
  {
    d = -pt1.get_x()*norm.get_x() - pt1.get_y()*norm.get_y() - pt1.get_z()*norm.get_z();
    a = norm.get_x();
    b = norm.get_y();
    c = norm.get_z();
  }
}

Plan::Plan(const std::array<Point3D, 3> & iPoints)
{
	Vector vec1(iPoints[0], iPoints[1]);
	Vector vec2(iPoints[2], iPoints[1]);
	Vector norm = vec1 ^ vec2;
	if (norm.norme_infini() == 0)
	{
		throw "error: can not build a plane";
	}
	else
	{
		d = -iPoints[0].get_x()*norm.get_x() - iPoints[0].get_y()*norm.get_y() - iPoints[0].get_z()*norm.get_z();
		a = norm.get_x();
		b = norm.get_y();
		c = norm.get_z();
	}
}

Plan::Plan(const Point3D & pt1, const Vector & normal) 
{
  d = -pt1.get_x()*normal.get_x() - pt1.get_y()*normal.get_y() - pt1.get_z()*normal.get_z();
  a = normal.get_x();
  b = normal.get_y();
  c = normal.get_z();
}

bool Plan::is_in_plane(const Point3D & pt) const
{
  if (std::abs (pt.get_x()*a + pt.get_y()*b + pt.get_z()*c + d) < Tolerance) 
  {
    return true;
  }
  else 
  {
    return false;
  }
}

Vector Plan::normal_vector() const
{
  return Vector {a, b, c};
}
