#include"MeshLoader.h"
#include "Mesh.h"
#include "Point3D.h"
#include <assert.h>
#include <iostream>  
#include <fstream>  
#include <cmath>
#include<map>


MeshLoader::MeshLoader(std::string filename)
{
	m_filename=filename;
}


Mesh MeshLoader::load() const
{
	
	std::ifstream in(m_filename.c_str());
	assert(in.good());


	size_t nb_points = 0;
	size_t nb_triangles = 0;
	std::string line;

	while (std::getline(in, line))
	{
		if (line.at(0) == 'v')
		{
			nb_points++;
		}
		else if (line.at(0) == 'f')
		{
			nb_triangles++;
		}
	}
	
	in.clear();
	in.seekg(0, in.beg);

	std::vector<Point3D> pts;
	pts.reserve(nb_points);
	std::vector<Triangle> tris;
	tris.reserve(nb_triangles+10);

	size_t iy =0;
	while (!std::getline(in, line).eof())
	{

		// reading a vertex  
		if (line.at(0) == 'v' && line.at(1) == ' ' )
		{
			double f1, f2, f3;
			if (sscanf_s(line.c_str(), "v %lf %lf %lf", &f1, &f2, &f3) == 3)
			{
				pts.push_back(Point3D(f1, f2, f3));
			}
			else
			{
				throw "ERROR: vertex not in wanted format";
			}
		}else if (line.at(0) == 'f' && line.at(1) == ' ' )
		{
			int index1 ,index2, index3;
			if (sscanf_s(line.c_str(), "f %d %d %d", &index1, &index2, &index3) == 3)
			{
				tris.push_back(Triangle({ pts[index1-1], pts[index2-1], pts[index3-1] }));
			}
			else
			{
				throw "ERROR: Face not in wanted format";
			}

		}


	}

	Mesh mesh(tris);
	return mesh;

}


MeshWriter::MeshWriter(std::string filename)
{
	m_filename = filename;
}


void MeshWriter::write(const Mesh & mesh) const
{
	std::ofstream out(m_filename.c_str());
	assert(out.is_open());

	std::map<std::array<double, 3>, int> pts;

	for (size_t i = 0; i < mesh.nb_triangles(); i++)
	{
		auto triangle = mesh.get_triangle(i);
		std::array<double, 3> pt{ triangle.get_point(0).get_x(),triangle.get_point(0).get_y(),triangle.get_point(0).get_z() };
		pts.insert(std::make_pair(pt, 0));
		pt = std::array<double, 3>{ triangle.get_point(1).get_x(), triangle.get_point(1).get_y(), triangle.get_point(1).get_z() };
		pts.insert(std::make_pair(pt, 0));
		pt = std::array<double, 3>{ triangle.get_point(2).get_x(), triangle.get_point(2).get_y(), triangle.get_point(2).get_z() };
		pts.insert(std::make_pair(pt, 0));
	}

	for (std::map<std::array<double, 3>, int>::iterator it = pts.begin(); it != pts.end(); ++it)
		out << "v " << it->first[0] << ' ' << it->first[1] << ' ' << it->first[2] << std::endl;

	for (size_t i = 0; i < mesh.nb_triangles(); i++)
	{
		auto triangle = mesh.get_triangle(i);
		std::array<double, 3> pt1{ triangle.get_point(0).get_x(),triangle.get_point(0).get_y(),triangle.get_point(0).get_z() };


		std::array<double, 3> pt2{ triangle.get_point(1).get_x(), triangle.get_point(1).get_y(), triangle.get_point(1).get_z() };
		

		std::array<double, 3> pt3{ triangle.get_point(2).get_x(), triangle.get_point(2).get_y(), triangle.get_point(2).get_z() };
		
		out<< "f " << distance(pts.begin(), pts.find(pt1))+1 << ' ' << distance(pts.begin(), pts.find(pt2)) +1<< ' ' << distance(pts.begin(), pts.find(pt3))+1 << std::endl;
	}

	out.close();

}