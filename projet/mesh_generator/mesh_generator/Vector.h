#pragma once
#include "Point3D.h"
#include <array>

class Vector
{
public:
  /*Constructors*/
  Vector();
  Vector(Point3D from, Point3D to);
  Vector(double axis_x, double axis_y, double axis_z);
  Vector(const Vector & vec) = default;

  /*norms of vector*/
  double norme1();
  double norme2();
  double norme_infini();
  
  /*get values*/
  double get_coord(size_t i) const { return m_Coords[i]; }
  double get_x() const;
  double get_y() const;
  double get_z() const;
  /*overload operators*/
  
  double operator *(const Vector & vec) const;
  Vector operator ^(const Vector & vec) const;

  inline friend Vector operator- (const Point3D & iPoint1, const Point3D & iPoint2);
  inline friend Point3D operator+ (const Vector & v, const Point3D & p);

private:
  std::array<double, 3> m_Coords;
};

inline Vector operator- (const Point3D & iPoint1, const Point3D & iPoint2)
{
  return Vector(iPoint2.get_x() - iPoint1.get_x(), iPoint2.get_y() - iPoint1.get_y(), iPoint2.get_z() - iPoint1.get_z());
}
inline Point3D operator+ (const Vector & v, const Point3D & p)
{
  return Point3D(p.get_x() + v.get_x(), p.get_y() + v.get_y(), p.get_z() + v.get_z());
}
