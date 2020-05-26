#include "Vector.h"
#include <cmath>
#include <algorithm>


Vector::Vector() : m_Coords{ 0.,0.,0. }
{
	
}
Vector::Vector(Point3D from, Point3D to) 
{
  for (size_t i = 0; i < 3; i++)
    m_Coords[i] = to.get_coord(i) - from.get_coord(i);
}

Vector::Vector(double axis_x, double axis_y, double axis_z) : m_Coords{ axis_x, axis_y, axis_z }
{
	
}

double Vector::norme1() 
{
  double res = 0;
  for (size_t i = 0; i < 3; i++)
    res += std::abs(m_Coords[i]);
  return res;
     
}

double Vector::norme2() 
{
  double res = 0;
  for (size_t i = 0; i < 3; i++)
    res += m_Coords[i] * m_Coords[i];
  return std::sqrt(res);
}


double Vector::norme_infini() 
{
  return *std::max_element(m_Coords.begin(), m_Coords.end(), [](double a, double b) {return std::abs(a) < std::abs(b); });
}

double Vector::get_x() const
{
	return m_Coords [0];
}

double Vector::get_y() const
{
  return m_Coords[1];
}

double Vector::get_z() const
{
  return m_Coords[2];
}

double Vector::operator * (const Vector & vec) const
{
  double res = 0.;
  for (size_t i = 0; i < 3; i++)
    res += vec.get_coord(i) * get_coord(i);
  return res;
}

Vector Vector::operator ^ (const Vector & vec) const
{
  return Vector(get_y()*vec.get_z() - this->get_z()*vec.get_y(), this->get_z()*vec.get_x() - this->get_x()*vec.get_z(), this->get_x()*vec.get_y() - this->get_y()*vec.get_x());

}