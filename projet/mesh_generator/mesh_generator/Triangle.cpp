#include "Triangle.h"

#include "Point3D.h"
#include "Vector.h"
#include "Plan.h"

#include <assert.h>

Triangle::Triangle(const std::vector<Point3D> & iPoints)
  : Polygon(iPoints)
{
}

bool Triangle::SameSide(const Point3D & A, const Point3D & B, const Point3D & C, const Point3D & P)
{
	Vector AB(A,B);
	Vector AC(A,C);
	Vector AP(A,P);

	Vector v1 = AB ^AC;
	Vector v2 = AB^AP;

	// v1 and v2 should point to the same direction
	return v1*v2 >= 0;
}



bool Triangle::plan_support(const Point3D & pt) const
{
  assert(m_Points.size() >= 3);
  Plan plan(m_Points[0], m_Points[1], m_Points[2]);
	if (!plan.is_in_plane(pt))
  {
		return false;
	} 
  else 
  {
		return SameSide(m_Points[0], m_Points[1], m_Points[2], pt) &&
		SameSide(m_Points[1], m_Points[2], m_Points[0], pt) &&
		SameSide(m_Points[2], m_Points[0], m_Points[1], pt);
	}
}