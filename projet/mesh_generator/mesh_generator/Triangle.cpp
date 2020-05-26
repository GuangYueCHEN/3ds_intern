#include "Triangle.h"

Triangle::Triangle(Point3D pts[], size_t nb_points):Polygon(pts,nb_points) {
	this->nb_points = nb_points;
	if (nb_points != 3) {
		throw "error: can not built Triangle";
	}
	else {
		points = new Point3D[3];
		for (int i = 0; i < 3; i++) {
			points[i] = pts[i];
		}
	}
}

bool SameSide(Point3D A, Point3D B, Point3D C, Point3D P)
{
	Vector AB(A,B);
	Vector AC(A,C);
	Vector AP(A,P);

	Vector v1 = AB ^AC;
	Vector v2 = AB^AP;

	// v1 and v2 should point to the same direction
	return v1*v2 >= 0;
}



bool Triangle::on_triangle(Point3D pt){
	Plan plan(points[0], points[1], points[2]);
	if (!plan.plan_support(pt)){
		return false;
	} else {
		return SameSide(points[0], points[1], points[2], pt) &&
		SameSide(points[1], points[2], points[0], pt) &&
		SameSide(points[2], points[0], points[1], pt);
	}
}