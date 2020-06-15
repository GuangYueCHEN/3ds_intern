#include "Mesh.h"
#include "Vector.h"
#include "constant.h"
#include <iostream>
#include <assert.h>
#include <cmath>

#include <map>

bool operator == (const Point3D & iP1, const Point3D & iP2)
{
	if (iP1.get_x() != iP2.get_x())
		return false;

	if (iP1.get_y() != iP2.get_y())
		return false;

	return iP1.get_z() == iP2.get_z();
}


Mesh::Mesh(const std::vector<Triangle> & iTriangles)
{	
	m_points.reserve(3 * iTriangles.size());
	m_triangles.reserve(iTriangles.size());
	for (const Triangle & tri : iTriangles)
	{
		std::vector<size_t> triangle;
		triangle.reserve(3);
		for (size_t i = 0; i < 3; i++)
		{
			Point3D point = tri.get_point(i);
			bool sign = false;
			for(size_t j=0; j < m_points.size(); j++)
			{
				if (point == m_points.at(j))
				{
					triangle.push_back(j);
					sign = true;
				}
			}
			if(sign == false)
			{
				m_points.push_back(point);
				triangle.push_back(m_points.size()-1);
			}
		}
		m_triangles.push_back(triangle);
	}
}


Triangle Mesh::get_triangle(const size_t & index) const 
{
	assert(index < m_triangles.size());
	std::vector<Point3D> pts;
	pts.reserve(3);
	for (size_t i = 0; i < 3; i++)
	{
		pts.push_back(m_points[m_triangles[index][i]]);
	}
	return Triangle(pts);
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
			if (p.distance_with(pt) < Tolerance)
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

bool Mesh::is_closed() const 
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
				Point3D pt = m_points[ t[i]];
				pts.push_back(pt);
			}
		}
		size_t res = point_reduce(pts);
		//std::cout << "nb points in mesh " << res << std::endl;
		return res == m_triangles.size();
	}

}

bool operator < (const Point3D & iP1, const Point3D & iP2)
{
  if (iP1.get_x() < iP2.get_x())
    return true;
  if (iP1.get_x() > iP2.get_x())
    return false;

  if (iP1.get_y() < iP2.get_y())
    return true;
  if (iP1.get_y() > iP2.get_y())
    return false;

  return iP1.get_z() < iP2.get_z();
}

bool Mesh::is_closed_2() const
{
  std::map<std::array<Point3D, 2>, int> EdgeMultiplicities;

  for (const auto & t : m_triangles)
  {
    for (int k = 0; k < 3; k++)
    {
      std::array<Point3D, 2> E{ m_points[ t[k]], m_points[t[(k + 1) % 3]] };
      if (E[1] < E[0])
        std::swap(E[0], E[1]);
      auto It = EdgeMultiplicities.insert(std::make_pair(E, 0));
      It.first->second++;
    }
  }

  if (std::all_of(EdgeMultiplicities.begin(), EdgeMultiplicities.end(), [](const std::pair<std::array<Point3D, 2>, int> & i) {return i.second == 2;}))
    return true;
  else
    return false;
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
	assert(is_closed());
	double sum = 0.;
	for (const auto & tri : m_triangles) 
	{
		std::vector<Point3D> pts;
		pts.reserve(3);
		pts.push_back(m_points[tri[0]]);
		pts.push_back(m_points[tri[1]]);
		pts.push_back(m_points[tri[2]]);
		Triangle triangle(pts);
		sum += SignedVolumeOfTriangle(triangle);
	}
	return std::abs(sum);
}

Point3D Mesh::near_with(const Point3D & pt) const
{
	Point3D near = m_points[m_triangles[0][0]];
	for (const auto & tri : m_triangles)
	{
		for (size_t i = 0; i < 3; i++)
		{
			if (pt.distance_with(near) < pt.distance_with(m_points[tri[i]]))
			{
				near = m_points[tri[i]];
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
			if (near.distance_with(m_points[tri[i]]) < Tolerance)
			{
				pts.push_back(pt);
			}
			else
			{
				pts.push_back(m_points[tri[i]]);
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
		std::vector<Point3D> pts;
		pts.reserve(3);
		pts.push_back(m_points[tri[0]]);
		pts.push_back(m_points[tri[1]]);
		pts.push_back(m_points[tri[2]]);
		Triangle triangle(pts);
		if (triangle.plan_support(pt))
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

/*
void Mesh::augmentation(const size_t & n_triangles)
{	
	assert((n_triangles - nb_triangles()) % 2 == 0);
	assert(n_triangles < nb_triangles()*3);
	std::vector<Triangle> tris;
	tris.reserve(n_triangles);
	std::vector<size_t> indexs;
	indexs.reserve(n_triangles - nb_triangles());
	for (size_t j = 0; j < n_triangles - nb_triangles(); j += 2)
	{
		double max = 0.;
		size_t index_max = 0;
		for (size_t i =0; i< nb_triangles(); i++)
		{
			double area = m_triangles[i].area();
			auto it = std::find(indexs.begin(), indexs.end(), i);
			if (area > max && it == indexs.end())
			{
				max = area;
				index_max = i;
			}
		}
		indexs.push_back(index_max);
		Triangle tri = m_triangles[index_max];
		Point3D center = tri.center();
		tris.push_back(Triangle({ center, tri.get_point(0), tri.get_point(1) }));
		tris.push_back(Triangle({ center, tri.get_point(0), tri.get_point(2) }));
		tris.push_back(Triangle({ center, tri.get_point(1), tri.get_point(2) }));
	}
	for (size_t i = 0; i < nb_triangles(); i++)
	{
		auto it = std::find(indexs.begin(), indexs.end(), i);
		if ( it == indexs.end())
		{
			tris.push_back(m_triangles[i]);
		}
	}
	m_triangles = tris;
}
*/


void Mesh::augmentation(const size_t & n_triangles)
{
	std::vector<Point3D> points;
	points.reserve(m_points.size() + (n_triangles - nb_triangles()) % 2);
	assert((n_triangles - nb_triangles()) % 2 == 0);
	std::vector<std::vector<size_t>> tris;
	tris.reserve(n_triangles);
	std::vector<std::vector<size_t>> tris2;
	tris2.reserve(n_triangles);
	for (const auto & p : m_points) 
	{
		points.push_back(p);
	}
	for (const auto & triangle : m_triangles)
	{
		tris.push_back(triangle);
	}
	size_t size = nb_triangles();
	for (size_t j = 0; j < n_triangles - nb_triangles(); j += 2)
	{
		double max = 0.;
		size_t index_max = 0;
		for (size_t i = 0; i < size; i++)
		{
			std::vector<Point3D> pts;
			pts.reserve(3);
			pts.push_back(m_points[tris[i][0]]);
			pts.push_back(m_points[tris[i][1]]);
			pts.push_back(m_points[tris[i][2]]);
			Triangle triangle(pts);
			double area = triangle.area();
			if (area > max)
			{
				max = area;
				index_max = i;
			}
		}
		
		for (size_t i = 0; i < size; i++)
		{
			if (i==index_max)
			{
				std::vector<Point3D> pts;
				pts.reserve(3);
				pts.push_back(m_points[tris[i][0]]);
				pts.push_back(m_points[tris[i][1]]);
				pts.push_back(m_points[tris[i][2]]);
				Triangle tri(pts);
				Point3D center = tri.center();
				points.push_back(center);
				size_t index_center = points.size() - 1;
				tris2.push_back(std::vector<size_t>({ index_center, tris[i][0], tris[i][1] }));
				tris2.push_back(std::vector<size_t>({ index_center, tris[i][1], tris[i][2] }));
				tris2.push_back(std::vector<size_t>({ index_center, tris[i][0], tris[i][2] }));
			}
			else
			{
				tris2.push_back(tris[i]);
			}
		}
		size += 2;
		tris = tris2;
		m_points = points;
		tris2.clear();
	}

	m_triangles = tris;
	
}
