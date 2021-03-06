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
		std::array<size_t,3> triangle;
		size_t index = 0;
		for (size_t i = 0; i < 3; i++)
		{
			Point3D point = tri.get_point(i);
			bool sign = false;
			for(size_t j=0; j < m_points.size(); j++)
			{
				if (point == m_points.at(j))
				{
					triangle[index++] = j;
					sign = true;
				}
			}
			if(sign == false)
			{
				m_points.push_back(point);
				triangle[index++] = m_points.size()-1;
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
  std::map<std::array<Point3D, 2>, size_t> EdgeMultiplicities;

  for (const auto & t : m_triangles)
  {
    for (size_t k = 0; k < 3; k++)
    {
      std::array<Point3D, 2> E{ m_points[ t[k]], m_points[t[(k + 1) % 3]] };
      if (E[1] < E[0])
        std::swap(E[0], E[1]);
      auto It = EdgeMultiplicities.insert(std::make_pair(E, 0));
      It.first->second++;
    }
  }

  if (std::all_of(EdgeMultiplicities.begin(), EdgeMultiplicities.end(), [](const std::pair<std::array<Point3D, 2>, size_t> & i) {return i.second == 2;}))
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
	std::vector<Point3D> pts;
	pts.reserve(3);
	double sum = 0.;
	for (const auto & tri : m_triangles) 
	{
		pts.push_back(m_points[tri[0]]);
		pts.push_back(m_points[tri[1]]);
		pts.push_back(m_points[tri[2]]);
		Triangle triangle(pts);
		sum += SignedVolumeOfTriangle(triangle);
		pts.clear();
	}
	return std::abs(sum);
}

Point3D Mesh::near_with(const Point3D & pt) const
{
  
  return *std::min_element(m_points.begin(), m_points.end(), [&](const Point3D & iP1, const Point3D & iP2)
  {
    return pt.distance_with(iP1) < pt.distance_with(iP2);
  });
  
}

Mesh Mesh::remeshing(const Point3D & near, const Point3D & pt) const
{
	std::vector<Triangle> new_triangles;
	new_triangles.reserve(m_triangles.size());
	std::vector<Point3D> pts;
	pts.reserve(3);
	for (const auto & tri : m_triangles)
	{
		for (size_t i = 0; i < 3; i++)
		{
			if (m_points[tri[i]]==near)
			{
				pts.push_back(pt);
			}
			else
			{
				pts.push_back(m_points[tri[i]]);
			}
		}
		new_triangles.push_back(Triangle(pts));
		pts.clear();
	}

	return Mesh(new_triangles);
}

// c'est pas la facon la plus simple

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
			return 0; // s'il n'est pas dans le triangle, retourne 0
		}
	}
	
  // ca marche pour n'importe quel point
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
	std::vector<std::array<size_t,3>> tris;
	tris.reserve(n_triangles);
	std::vector<std::array<size_t, 3>> tris2;
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
				tris2.push_back(std::array<size_t,3>({ index_center, tris[i][0], tris[i][1] }));
				tris2.push_back(std::array<size_t, 3>({ index_center, tris[i][1], tris[i][2] }));
				tris2.push_back(std::array<size_t, 3>({ index_center, tris[i][0], tris[i][2] }));
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













// tester si le point existe deja devrait etre optionnel, car ca peut etre couteux en temps
size_t Mesh::add_point(const Point3D & p, bool SearchForPoint)
{
	size_t index = 0;
  if (SearchForPoint)
  {
    for (const auto & pt : m_points)
    {
      if (p == pt)
      {
        return index;
      }
      index++;
    }
  }
	
	m_points.push_back(p);
	return m_points.size() - 1;
}

bool triangle_equal(const std::array<size_t, 3> & tri, const std::array<size_t, 3> & indices)
{
	/*if (tri == indices) return true;
	std::array<size_t, 3>  indices1 = { indices[0], indices[2], indices[1]};
	if (tri == indices1) return true;
	indices1 = { indices[1], indices[0], indices[1] };
	if (tri == indices1) return true;
	indices1 = { indices[1], indices[2], indices[0] };
	if (tri == indices1) return true;
	indices1 = { indices[2], indices[0], indices[1] };
	if (tri == indices1) return true;
	indices1 = { indices[2], indices[1], indices[0] };
	if (tri == indices1) return true;
	return false;*/
  auto s1 = tri;
  auto s2 = indices;
  std::sort(s1.begin(), s1.end());
  std::sort(s2.begin(), s2.end());
  return s1 == s2;
}

size_t Mesh::add_triangle(const std::array<size_t, 3> & indices, bool SearchForPoint )
{
	assert(indices[0] < m_points.size());
	assert(indices[1] < m_points.size());
	assert(indices[2] < m_points.size());
	size_t index = 0;

  if (SearchForPoint)
  {
    for ( size_t tri = 0; tri < m_triangles.size(); tri++)
    {
      if (triangle_equal(m_triangles[tri], indices))
      {
        return tri;
      }
    }
  }
	
	m_triangles.push_back(indices);
	return m_triangles.size() - 1;
}

void Mesh::remove_triangle(size_t iNumTriangle)
{
	assert(iNumTriangle < m_triangles.size());
  m_triangles.erase(m_triangles.begin() + iNumTriangle);
	/*for (size_t i = iNumTriangle; i < m_triangles.size()-1; i++)
	{
		m_triangles[i] = m_triangles[i + 1];
	}
	m_triangles.pop_back();*/
}
void Mesh::remove_vertex(size_t iNumVertex)
{
	assert(iNumVertex < m_points.size());
  
  // pourquoi dans cet ordre ? :)
	for (int i = m_triangles.size() - 1; i >= 0 ; i--)
	{

		if (m_triangles[i][0] == iNumVertex || m_triangles[i][1] == iNumVertex || m_triangles[i][2] == iNumVertex)
		{
			
			remove_triangle(i);
			
		}
	}

  m_points.erase(m_points.begin() + iNumVertex);

	/*m_points[iNumVertex] = m_points[m_points.size() - 1];
	m_points.pop_back();

	for (size_t i = 0; i < m_triangles.size(); i++)
	{
		if (m_triangles[i][0] == m_points.size())
		{
			m_triangles[i][0] = iNumVertex;
		}
		else if (m_triangles[i][1] == m_points.size())
		{
			m_triangles[i][1] = iNumVertex;
		}
		else if (m_triangles[i][2] == m_points.size())
		{
			m_triangles[i][2] = iNumVertex;
		}
	}*/
}

std::vector<size_t> Mesh::GetVerticesAroundVertex(size_t iNumVertex) const //renvoie la liste des triangles qui contiennent un vertex
{
	std::vector<size_t> res;
	res.push_back(iNumVertex);
	std::vector<size_t> triangles = GetTrianglesAroundVertex(iNumVertex);
	for (const auto & tri : triangles)
	{	
		for(size_t i = 0 ; i < 3 ; i++)
		{ 
			if (!std::binary_search(res.begin(), res.end(), m_triangles[tri][i]))
			{
				res.push_back(m_triangles[tri][i]);
			}
		}
	}
	res.erase(res.begin());
	return res;
}

std::vector<size_t> Mesh::GetTrianglesAroundVertex(size_t iNumVertex) const //renvoie la liste des triangles qui contiennent un vertex
{
	std::vector<size_t> res;
	for (size_t i = 0; i < m_triangles.size(); i++)
	{
		if (m_triangles[i][0] == iNumVertex || m_triangles[i][1] == iNumVertex || m_triangles[i][2] == iNumVertex)
		{
			res.push_back(i);
		}
	}
	return res;
}
std::vector<size_t> Mesh::GetTrianglesAroundTriangles(size_t itr) const
{
  std::vector<size_t> Result; Result.reserve(3);

  std::array<size_t, 3> trVertices(m_triangles[itr]);
  std::sort(trVertices.begin(), trVertices.end());
  for (size_t t = 0; t < m_triangles.size(); t++)
  {
    if (itr == t) continue;
    std::array<size_t, 3> currentTr(m_triangles[t]);
    std::sort(currentTr.begin(), currentTr.end());
    std::vector<size_t> intersection; intersection.reserve(2);
    std::set_intersection(trVertices.begin(), trVertices.end(), currentTr.begin(), currentTr.end(), intersection.end());
    if (intersection.size() == 2)
      Result.push_back(t);
  }
  return Result;
}

std::vector<size_t> Mesh::GetTrianglesAroundEdge(size_t Vertex1, size_t Vertex2) const //renvoie la liste des triangles qui partagent une edge donn�e (sans ordre) par 2 vertices
{

  auto R1 = GetTrianglesAroundVertex(Vertex1), R2 = GetTrianglesAroundVertex(Vertex2);
  std::vector<size_t> Result;
  std::set_intersection(R1.begin(), R1.end(), R2.begin(), R2.end(), std::back_inserter(Result));

  return Result;
}

void Mesh::flip_edge(size_t Vertex1, size_t Vertex2)
{
	std::vector<size_t> res = GetTrianglesAroundEdge(Vertex1,Vertex2);
	assert(res.size() == 2);
	const std::array<size_t, 3> &tri1 = m_triangles[res[0]];
	const std::array<size_t, 3> &tri2 = m_triangles[res[1]];
	size_t vertex3, vertex4;
	for(const size_t & vertex : tri1)
	{
		if (vertex != Vertex1 && vertex !=Vertex2)
		{
			vertex3 = vertex;
		}
	}
	for (const size_t & vertex : tri2)
	{
		if (vertex != Vertex1 && vertex != Vertex2)
		{
			vertex4 = vertex;
		}
	}
	/*std::vector<size_t> res1 = GetTrianglesAroundEdge(vertex3, vertex4);
	assert(res1.size() == 0);*/
	m_triangles[res[0]] = std::array<size_t, 3>({Vertex1, vertex3, vertex4  });
	m_triangles[res[1]] = std::array<size_t, 3>({Vertex2, vertex4, vertex3  });
}

void Mesh::SplitEdge(size_t Vertex1, size_t Vertex2, const Point3D & p) //coupe une edge en ins�rant un point 
{
	size_t index = add_point(p);
	std::vector<size_t> res = GetTrianglesAroundEdge(Vertex1, Vertex2);
	//assert(res.size() == 2); // pourquoi ? on peut splitter une edge sur le bord
	
  std::vector<size_t> OtherPoints;

  for (const auto & t : res)
  {
    for (const size_t & vertex : m_triangles[t])
    {
      if (vertex != Vertex1 && vertex != Vertex2)
      {
        OtherPoints.push_back(vertex);
      }
    }
  }
  assert(OtherPoints.size() == res.size());
  for (size_t k=0; k < OtherPoints.size(); k++)
  {
    m_triangles[res[k]] = std::array<size_t, 3>{Vertex1, OtherPoints[k], index  };
    add_triangle(std::array<size_t, 3>{Vertex2, index, OtherPoints[k]  });
  }	
}
void Mesh::CollapseEdge(size_t Vertex1, size_t Vertex2)
{

	std::vector<size_t> res = GetTrianglesAroundEdge(Vertex1, Vertex2);
	double x = (m_points[Vertex1].get_x() + m_points[Vertex2].get_x()) / 2.;
	double y = (m_points[Vertex1].get_y() + m_points[Vertex2].get_y()) / 2.;
	double z = (m_points[Vertex1].get_z() + m_points[Vertex2].get_z()) / 2.;

	if (res[0] > res[1])
	{
		remove_triangle(res[0]);
		remove_triangle(res[1]);
	}
	else
	{
		remove_triangle(res[1]);
		remove_triangle(res[0]);
	}

	m_points[Vertex1] = Point3D({ x,y,z });
	for (size_t i = 0; i < m_triangles.size(); i++)
	{
		for (size_t k = 0; k < 3; k++)
		{
			if (m_triangles[i][k] == Vertex2)
			{
				m_triangles[i][k] = Vertex1;
				break;
			}
		}
	}

	remove_vertex(Vertex2);
}


Point3D operator * (const double & alpha , const Point3D & pt)
{
	return Point3D(pt.get_x() * alpha, pt.get_y() * alpha, pt.get_z() * alpha);
}


void Mesh::LaplacianSmoothing(size_t itr, double alpha)
{
	for (size_t i = 0; i < itr; i++)
	{
		std::vector<Point3D> points = m_points;
		for (size_t j = 0; j < m_points.size(); j++)
		{
			std::vector<size_t> triangles = GetTrianglesAroundVertex(j);
			std::vector<size_t> vertices = GetVerticesAroundVertex(j);
			std::array<double, 3> new_position = { 0., 0., 0. };
			for (const size_t & vertex : vertices)
			{
				new_position[0] -= m_points[vertex].get_x() / vertices.size();
				new_position[1] -= m_points[vertex].get_y() / vertices.size();
				new_position[2] -= m_points[vertex].get_z() / vertices.size();
			}
			Point3D new_point(new_position);
			points[j] = m_points[j] +  alpha * ( m_points[j] + new_point);
		}
		m_points = points;
	}
}