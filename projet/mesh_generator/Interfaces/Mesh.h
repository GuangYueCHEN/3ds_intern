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

	size_t add_point(const Point3D & p, bool SearchForPoint = false);
	size_t add_triangle(const std::array<size_t,3> & indices, bool SearchForTriangle = false);
	void remove_triangle(size_t iNumTriangle);
	void remove_vertex(size_t iNumVertex);
	
	std::vector<size_t> GetTrianglesAroundVertex(size_t vertix) const; //renvoie la liste des triangles qui contiennent un vertex
	std::vector<size_t> GetTrianglesAroundTriangles(size_t itr) const;
	std::vector<size_t> GetTrianglesAroundEdge(size_t Vertex1, size_t Vertex2) const; //renvoie la liste des triangles qui partagent une edge donnée (sans ordre) par 2 vertices
	
  void flip_edge(size_t Vertex1, size_t Vertex2);
	void SplitEdge(size_t Vertex1, size_t Vertex2, const Point3D & p); //coupe une edge en insérant un point 
	void CollapseEdge(size_t Vertex1, size_t Vertex2);

private:
	static size_t point_reduce(const  std::vector <Point3D> & pts);
	static double SignedVolumeOfTriangle(const Triangle & tri);
	std::vector<Point3D> m_points;
	std::vector<std::array<size_t,3>> m_triangles;
};
