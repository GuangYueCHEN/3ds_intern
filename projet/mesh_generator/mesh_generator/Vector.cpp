#include "Vector.h"

Vector::Vector() {
	x = y = z = 0;
}
Vector::Vector(Point3D from, Point3D to) {
	x = to.get_x() - from.get_x();
	y = to.get_y() - from.get_y();
	z = to.get_z() - from.get_z();
}

Vector::Vector(double axis_x, double axis_y, double axis_z) {
	x = axis_x;
	y = axis_y;
	z = axis_z;
}

Vector::Vector(const Vector & vec) {
	this->x = vec.x;
	this->y = vec.y;
	this->z = vec.z;
}

double Vector::norme1() {
	return x+y+z;
}

double Vector::norme2() {
	return std::sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2));
}


double Vector::norme_infini() {
	return std::max( std::max(abs(x),abs(y)), abs(z));
}

double Vector::get_x() {
	return x;
}

double Vector::get_y() {
	return y;
}

double Vector::get_z() {
	return z;
}

double Vector::operator * (const Vector & vec) {
	return this->x*vec.x + this->y*vec.y + this->z*vec.z;
}

Vector Vector::operator ^ (const Vector & vec) {
	Vector dot;
	dot.x = this->y*vec.z - this->z*vec.y;
	dot.y = this->z*vec.x - this->x*vec.z;
	dot.z = this->x*vec.y - this->y*vec.x;
	return dot;
}