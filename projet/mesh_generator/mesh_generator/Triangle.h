#pragma once
#include "Polygon.h"

class Point3D;

class Triangle : public Polygon 
{
public:
	Triangle(const std::vector<Point3D> & iPoints);
	virtual bool plan_support(const Point3D & pt) const override;//If a point is on the triangle, return true;

private:
  static bool SameSide(const Point3D & A, const Point3D & B, const Point3D & C, const Point3D & P);
};
