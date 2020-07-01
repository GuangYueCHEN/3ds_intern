#pragma once
#include <array>

class Vector;

class Point3D
{
public:
  /*Constructor*/
  Point3D();
  Point3D(double axis_x, double axis_y, double axis_z);
  Point3D(std::array<double, 3> Coords);
  Point3D(const Point3D& point) = default;
  
  /*get values*/
  double get_coord(size_t i) const { return m_Coords[i]; }
  double get_x() const;
  double get_y() const;
  double get_z() const;
  void print_position() const; // print x, y, z

  double distance_with(const Point3D & point) const; // distance with another vertice

  inline friend Point3D operator+(const Point3D & p1, const Point3D & p2) ;

private:
  std::array<double,3> m_Coords;
};

inline Point3D operator+(const Point3D & p1, const Point3D & p2)
{
  return Point3D(p1.get_x() + p2.get_x(), p1.get_y() + p2.get_y(), p1.get_z() + p2.get_z());
}