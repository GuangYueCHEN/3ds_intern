#include "Point3D.h"
#include "Vector.h"

#include <cmath>
#include <iostream>


Point3D::Point3D() : m_Coords{0.,0.,0.}
{
	
}

Point3D::Point3D(double axis_x, double axis_y, double axis_z) : m_Coords{axis_x, axis_y, axis_z}
{
	
}

double Point3D::get_x() const
{
	return m_Coords[0];
}

double Point3D::get_y() const
{
	return m_Coords[1];
}

double Point3D::get_z() const
{
	return m_Coords[2];
}

void Point3D::print_position() const
{
	std::cout << "Printing point position:" << std::endl << "x:" << m_Coords[0] << ", y:" << m_Coords[1] << ", z:" << m_Coords[2] << std::endl;
}

double Point3D::distance_with(const Point3D & point) 
{
  return std::sqrt
  (
    (point.get_x() - m_Coords[0])*(point.get_x() - m_Coords[0]) + 
    (point.get_y() - m_Coords[1])*(point.get_y() - m_Coords[1]) + 
    (point.get_z() - m_Coords[2])*(point.get_z() - m_Coords[2])
  );
}


