#pragma once
#include "Polygon.h"
class Triangle : public Polygon {
public:
	Triangle(Point3D pts[], size_t nb_points);
	bool on_triangle(Point3D pt);//If a point is on the triangle, return true;
};