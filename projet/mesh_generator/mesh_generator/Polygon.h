#pragma once

#include "Point3D.h"

#include <vector>
#include <array>
 
class Plan;

class Polygon 
{
public:
	/*Constructors*/
	Polygon(const std::vector<Point3D> & iPoints); //throw error
	Polygon(const Polygon & pol) = default;
	~Polygon() {};

	Point3D get_point(size_t index) const; // get the vertice

  // a ecrire
	// Plan plan_support(); // if all of vertices in this plan, return true;
	
	virtual bool plan_support(const Point3D & pt) const; // If a new point on the surface, return true;
	Point3D center() const;
	double area() const;
	
protected:
  std::vector<Point3D> m_Points;
};
