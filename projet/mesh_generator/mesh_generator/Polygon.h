#pragma once
#include "Point3D.h"
#include "Plan.h"

class Polygon {
public:
	/*Constructors*/
	Polygon(Point3D pts[],size_t nb_points); //throw error
	Polygon(const Polygon & pol);
	~Polygon();

	Point3D get_point(size_t index); // get the vertice
	bool plan_support(); // if all of vertices in this plan, return true;
	bool plan_support(Point3D pt); // If a new point on the surface, return true;
	Point3D center();
	double area();
	
protected:
	int nb_points;
	Point3D *points;
};
