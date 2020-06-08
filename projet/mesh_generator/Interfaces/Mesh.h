#pragma once
#include "Triangle.h"

#include <vector>
#include <array>

class Point3D;

class Mesh {

public:
	/*Constructors*/
	Mesh(const std::vector<Triangle> & iTriangles);
	Mesh(const Mesh & mesh) = default;
  Mesh() = default;

	/*get values*/
	Triangle get_triangle(const size_t & index) const;
	size_t nb_triangles() const;
	Point3D near_with(const Point3D & pt) const; //get the nearest point

	/*methods*/
	bool is_closed() const;
    bool is_closed_2() const;
	double volume() const;
	int point_position(const Point3D & pt) const; //0: on mesh; 1 : in mesh; -1 : out of mesh
	Mesh remeshing(const Point3D & near, const Point3D & pt) const; //remeshing mesh with replaced Point3
	
	void augmentation(const size_t & n_triangle);

	void add_triangle(const Triangle & t)
	  {
		m_triangles.push_back(t);
	  }

private:
	static size_t point_reduce(const  std::vector <Point3D> & pts);
	static double SignedVolumeOfTriangle(const Triangle & tri);
	std::vector<Triangle> m_triangles;
};
