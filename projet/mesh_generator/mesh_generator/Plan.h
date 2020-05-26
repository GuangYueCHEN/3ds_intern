#pragma once
#include "Point3D.h"
#include "Vector.h"

class Plan{
public:
	/*Constructors*/
	Plan(Point3D pt1, Point3D pt2,Point3D pt3); //throw error
	Plan(Point3D pt, Vector norm);
	bool plan_support(Point3D pt); 
	Vector normal(); //the normal vector
private:
	double a, b, c, d;
};