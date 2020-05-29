#include"SimpleShapeGenerator.h"
#include "Mesh.h"
#include "constant.h"
#include "Point3D.h"
#include "Triangle.h"
#include <vector>
#include<iostream>

CubeGenerator::CubeGenerator(double l) {
	L = l;
}

Mesh CubeGenerator::run() const{
	double x = L / 2.;
	std::vector<Point3D> pts;
	pts.reserve(3);
	std::vector<Triangle> tris;
	tris.reserve(12);

	pts = { Point3D(x,x,-x),Point3D(-x,x,-x), Point3D(x,-x,-x) };
	tris.push_back(Triangle(pts));
	pts.clear();
	
	pts = { Point3D(-x,-x,-x),Point3D(-x,x,-x), Point3D(x,-x,-x) };
	tris.push_back(Triangle(pts));
	pts.clear();
	
	pts = { Point3D(x,x,-x),Point3D(x,x,x), Point3D(x,-x,-x) };
	tris.push_back(Triangle(pts));
	pts.clear();
	
	pts = { Point3D(x,-x,x),Point3D(x,x,x), Point3D(x,-x,-x) };
	tris.push_back(Triangle(pts));
	pts.clear();
	
	pts = { Point3D(x,x,-x),Point3D(x,x,x), Point3D(-x,x,-x) };
	tris.push_back(Triangle(pts));
	pts.clear();

	pts = { Point3D(-x,x,x),Point3D(x,x,x), Point3D(-x,x,-x) };
	tris.push_back(Triangle(pts));
	pts.clear();

	pts = { Point3D(-x,x,x),Point3D(-x,-x,-x), Point3D(-x,x,-x) };
	tris.push_back(Triangle(pts));
	pts.clear();

	pts = { Point3D(-x,x,x),Point3D(-x,-x,-x), Point3D(-x,-x,x) };
	tris.push_back(Triangle(pts));
	pts.clear();


	pts = { Point3D(x,-x,-x),Point3D(-x,-x,-x), Point3D(-x,-x,x) };
	tris.push_back(Triangle(pts));
	pts.clear();

	pts = { Point3D(x,-x,-x),Point3D(x,-x,x), Point3D(-x,-x,x) };
	tris.push_back(Triangle(pts));
	pts.clear();

	pts = { Point3D(x,x,x),Point3D(x,-x,x), Point3D(-x,-x,x) };
	tris.push_back(Triangle(pts));
	pts.clear();

	pts = { Point3D(x,x,x),Point3D(-x,x,x), Point3D(-x,-x,x) };
	tris.push_back(Triangle(pts));
	return Mesh(tris);

}

SphereGenerator::SphereGenerator(double radius, size_t nb_Parallels, size_t nb_Meridians) {
	Radius = radius;
	NbMeridians = nb_Meridians;
	NbParallels = nb_Parallels;
}


Mesh SphereGenerator::run() const {
	std::vector<Point3D> pts;
	std::vector<Triangle> tris;
	pts.reserve((NbParallels-1) * NbMeridians+2);
	tris.reserve(NbParallels  * NbMeridians);
	pts.push_back(Point3D(0.0, Radius, 0.0));
	for (size_t j = 0; j < NbParallels - 1; j++) {
		double const polar = Pi * double(j + 1) / double(NbParallels);
		double const sp = std::sin(polar);
		double const cp = std::cos(polar);
		for (size_t i = 0; i < NbMeridians; i++) {
			double const azimuth = 2.0 * Pi * double(i) / double(NbMeridians);
			double const sa = std::sin(azimuth);
			double const ca = std::cos(azimuth);
			double const x = sp * ca *Radius;
			double const y = cp * Radius;
			double const z = sp * sa * Radius;
			pts.push_back(Point3D(x, y, z));
		}
	}
	pts.push_back(Point3D(0.0, -Radius, 0.0));

	//std::cout << pts.size() << std::endl;
		
	std::vector<Point3D> construct_points;
	construct_points.reserve(3);

	for (size_t i = 0; i < NbMeridians; ++i)
	{
		size_t const a = i + 1;
		size_t const b = (i + 1) % NbMeridians + 1;
		construct_points.push_back(pts[0]);
		construct_points.push_back(pts[b]);
		construct_points.push_back(pts[a]);
		tris.push_back(Triangle( construct_points) );
		construct_points.clear();
	}

	for (size_t j = 0; j < NbParallels - 2; ++j)
	{
		size_t aStart = j * NbMeridians + 1;
		size_t bStart = (j + 1) * NbMeridians + 1;
		for (size_t i = 0; i < NbMeridians; ++i)
		{
			const size_t a = aStart + i;
			const size_t a1 = aStart + (i + 1) % NbMeridians;
			const size_t b = bStart + i;
			const size_t b1 = bStart + (i + 1) % NbMeridians;
			construct_points.push_back(pts[a]);
			construct_points.push_back(pts[a1]);
			construct_points.push_back(pts[b]);
			tris.push_back(Triangle(construct_points));
			construct_points.clear();
			construct_points.push_back(pts[b1]);
			construct_points.push_back(pts[a1]);
			construct_points.push_back(pts[b]);
			tris.push_back(Triangle(construct_points));
			construct_points.clear();
		}
	}

	for (size_t i = 0; i < NbMeridians; ++i)
	{
		size_t const a = i + NbMeridians * (NbParallels - 2) + 1;
		size_t const b = (i + 1) % NbMeridians + NbMeridians * (NbParallels - 2) + 1;
		construct_points.push_back(pts[pts.size() - 1]);
		construct_points.push_back(pts[a]);
		construct_points.push_back(pts[b]);
		tris.push_back(Triangle(construct_points));
		construct_points.clear();

	}

	//std::cout << tris.size() << std::endl;

	return Mesh(tris);
}