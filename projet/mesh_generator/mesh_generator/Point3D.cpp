#include <cmath>
#include <iostream>
#include "Point3D.h"

Point3D::Point3D() {
	x = 0.0;
	y = 0.0;
	z = 0.0;
}

Point3D::Point3D(double axis_x, double axis_y, double axis_z) {
	x = axis_x;
	y = axis_y;
	z = axis_z;
}

Point3D::Point3D(const Point3D & point) {
	this->x = point.x;
	this->y = point.y;
	this->z = point.z;
}

double Point3D::get_x() {
	return x;
}

double Point3D::get_y() {
	return y;
}

double Point3D::get_z() {
	return z;
}

void Point3D::print_position(){
	std::cout << "Printing point position:" << std::endl << "x:" << x << ", y:" << this->y << ", z:" << z << std::endl;
}
double Point3D::distance_with(Point3D point) {
	return std::sqrt(pow(point.get_x() - this->x,2) + pow(point.get_y() - this->y,2) + pow(point.get_z() - this->z,2));
}
