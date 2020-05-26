#include "Polygon.h"

Polygon::Polygon(Point3D pts[], size_t nb_points) {
	this->nb_points = nb_points;
	if (nb_points <= 2) {
		throw "error: can not built Polygon";
	}else {
		points = new Point3D[nb_points];
		for (int i = 0; i < nb_points; i++) {
			points[i] = pts[i];
		}
	}

}

Polygon::Polygon(const Polygon & pol) {
	if (!pol.points) {
		nb_points = 0;
		points = nullptr;
	}
	else {
		nb_points = pol.nb_points;
		for (int i = 0; i < nb_points; i++) {
			points[i] = pol.points[i];
		}
	}
}

Polygon::~Polygon() {
	nb_points = 0;
	delete[] points;
}

Point3D Polygon::get_point(size_t index) {
	if (index >= nb_points) {
		throw "out of index";
	}
	else {
		return points[index];
	}
}

bool Polygon::plan_support() {
	if (nb_points == 3) return true;
	Plan plan(points[0], points[1], points[2]);
	bool res = true;
	for (int i = 3; i < nb_points; i++) {
		res = res && plan.plan_support(points[i]);
	}
	return res;
}

bool Polygon::plan_support(Point3D pt) {
	Plan plan(points[0], points[1], points[2]);
	return plan.plan_support(pt);
}

Point3D Polygon::center() {
	double x = 0, y = 0, z = 0;
	for (int i = 0; i < nb_points; i++) {
		x += points[i].get_x();
		y += points[i].get_y();
		z += points[i].get_z();
	}
	Point3D center(x / nb_points, y / nb_points, z / nb_points);
	return center;
}

double Polygon::area() {
	double s = 0;
	Vector vec1,vec2,vec;
	Point3D zero;

	for (int i = 0; i < nb_points; i++) {
		vec1 = Vector(zero, points[i]);
		vec2 = Vector(zero, points[(i + 1) % nb_points]);
		vec = vec1 ^ vec2;
		s += vec.get_x() + vec.get_y() + vec.get_z();
		}
	return abs(s) / 2;
}

