#pragma once

#include <array>


class Point3D;
class Vector;

class Plan
{
public:
  /*Constructors*/
  Plan(const Point3D & pt1, const Point3D & pt2, const Point3D & pt3); //throw error
  Plan(const std::array<Point3D,3> & iPoints);
  Plan(const Point3D & pt, const Vector & norm);
  
  bool is_in_plane(const Point3D & pt) const ;
  Vector normal_vector() const ; //the normal vector

private:

  double a, b, c, d;
};