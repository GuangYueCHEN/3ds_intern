#include "Mesh.h"
#include "Vector.h"
#include <iostream>
#include <assert.h>
#include <cmath>

Mesh::Mesh(const std::vector<Triangle> & iTriangles) : m_triangles(iTriangles) {};


Triangle Mesh::get_triangle(const size_t & index) const 
{
	assert(index < m_triangles.size());
	return m_triangles[index];
}

size_t Mesh::nb_triangles() const
{
	return m_triangles.size();
}


size_t Mesh::point_reduce(const  std::vector <Point3D> & pts) 
{
	size_t nb_points = 0;
	for (const auto & p : pts)
	{
		size_t same = 0;
		for (const auto & pt : pts)
		{
			if (p.distance_with(pt) < 1e-6)
			{
				same++;
			}
		}

		if (same != 3) {
			return 0;
		}
		else
		{
			nb_points++;
		}
	}
	return nb_points/3;
}

bool Mesh::is_colsed() const 
{
	if (m_triangles.size() < 4 || m_triangles.size() % 2 == 1)
	{
		return false;
	}
	else
	{
		std::vector<Point3D> pts;
		pts.reserve(3 * m_triangles.size());
		for (const auto & t : m_triangles)
		{
			for (size_t i = 0; i < 3; i++)
			{
				Point3D pt = t.get_point(i);
				pts.push_back(pt);
			}
		}
		size_t res = point_reduce(pts);
		//std::cout << "nb points in mesh " << res << std::endl;
		return res == m_triangles.size();
	}

}
	
double Mesh::SignedVolumeOfTriangle(const Triangle & tri)
{
	Point3D zero;
	Vector v1 = tri.get_point(0) - zero;
	Vector v2 = tri.get_point(1) - zero;
	Vector v3 = tri.get_point(2) - zero;
	return v1*(v2^v3) / 6.0;
}


double Mesh::volume() const
{
	assert(is_colsed());
	double sum = 0.;
	for (const auto & tri : m_triangles) 
	{
		sum += SignedVolumeOfTriangle(tri);
	}
	return std::abs(sum);
}

Point3D Mesh::near_with(const Point3D & pt) const
{
	Point3D near = m_triangles[0].get_point(0);
	for (const auto & tri : m_triangles)
	{
		for (size_t i = 0; i < 3; i++)
		{
			if (pt.distance_with(near) < pt.distance_with(tri.get_point(i)))
			{
				near = tri.get_point(i);
			}
		}
	}
	return near;
}

Mesh Mesh::remeshing(const Point3D & near, const Point3D & pt) const
{
	std::vector<Triangle> new_triangles;
	new_triangles.reserve(m_triangles.size());

	for (const auto & tri : m_triangles)
	{
		std::vector<Point3D> pts;
		pts.reserve(3);
		for (size_t i = 0; i < 3; i++)
		{
			if (near.distance_with(tri.get_point(i)) < 1e-10)
			{
				pts.push_back(pt);
			}
			else
			{
				pts.push_back(tri.get_point(i));
			}
		}
		new_triangles.push_back(Triangle(pts));
	}

	return Mesh(new_triangles);
}

int Mesh::point_position(const Point3D & pt) const
{
	for (const auto & tri : m_triangles) 
	{
		if (tri.plan_support(pt))
		{
			return 0;
		}
	}
	
	Point3D near = near_with(pt);

	Mesh new_mesh = remeshing(near, pt);

	if (new_mesh.volume() >= volume())
	{
		return -1;
	}
	else
	{
		return 1;
	}

}