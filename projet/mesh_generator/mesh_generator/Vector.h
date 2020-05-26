#pragma once
#include "Point3D.h"
#include <algorithm>
class Vector {
public:
	/*Constructors*/
	Vector();
	Vector(Point3D from, Point3D to);
	Vector(double axis_x, double axis_y, double axis_z);
	Vector(const Vector & vec);
	/*norms of vector*/
	double norme1();
	double norme2();
	double norme_infini();
	/*get values*/
	double get_x();
	double get_y();
	double get_z();
	/*overload operators*/
	double operator *(const Vector & vec);
	Vector operator ^(const Vector & vec);
private:
	double x;
	double y;
	double z;
};
