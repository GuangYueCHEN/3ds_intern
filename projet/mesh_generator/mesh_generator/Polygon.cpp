#include "Polygon.h"
#include "Plan.h"
#include "Vector.h"

Polygon::Polygon(const std::vector<Point3D> & iPoints) : m_Points(iPoints) {}

Point3D Polygon::get_point(size_t index) 
{
  if (index >= m_Points.size()) 
  {
    throw "out of index";
  }
  else 
  {
    return m_Points[index];
  }
}

//bool Polygon::plan_support() 
//{
//  if (nb_points == 3) return true;
//  Plan plan(points[0], points[1], points[2]);
//  bool res = true;
//  for (int i = 3; i < nb_points; i++) {
//    res = res && plan.plan_support(points[i]);
//  }
//  return res;
//}

bool Polygon::plan_support(const Point3D & pt) const
{
  Plan plan(m_Points[0], m_Points[1], m_Points[2]);
  return plan.is_in_plane(pt);
}

Point3D Polygon::center() const
{
  Point3D res;
  for (const auto & p : m_Points) 
  {
    res = res + p;
  }
  return res;
}

double Polygon::area() const
{
  double s = 0;
  Point3D zero;
  const size_t nb_points = m_Points.size();
  for (size_t i = 0; i < m_Points.size(); i++) 
  {
    Vector vec1 = zero - m_Points[i];
    Vector vec2 = zero - m_Points[(i + 1) % nb_points];
    Vector vec = vec1 ^ vec2;
    s += vec.norme2();
  }
  return s / 2.;
}

