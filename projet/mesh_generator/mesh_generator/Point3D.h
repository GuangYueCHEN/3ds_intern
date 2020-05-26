#pragma once
#include <iostream>


class Point3D{
public:
	/*Constructor*/
	Point3D();
	Point3D(double axis_x, double axis_y, double axis_z);
	Point3D(const Point3D& point);
	/*get values*/
	double get_x();
	double get_y();
	double get_z();
	void print_position(); // print x, y, z

	double distance_with(Point3D point); // distance with another vertice

private:
	double x, y, z;
};