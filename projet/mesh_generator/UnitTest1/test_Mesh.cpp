#include "stdafx.h"
#include "targetver.h"
#include "./../mesh_generator/Mesh.h"
#include <vector>
#include "./../mesh_generator/Triangle.h"
#include "./../mesh_generator/Point3D.h"


namespace UnitTest1 
{


	TEST_CLASS(mesh_test)
	{
	public:
		Mesh generate_mesh()
		{
			Point3D a(1., 0., 0.);
			Point3D b(0., 1., 0.);
			Point3D c(0., 0., 1.);
			Point3D d(0., 0., 0.);

			std::vector<Point3D> pts;
			std::vector<Triangle> tris;
			pts.reserve(3);
			pts.push_back(a);
			pts.push_back(b);
			pts.push_back(c);
			Triangle tri(pts);
			tris.push_back(tri);
			pts.clear();

			pts.push_back(d);
			pts.push_back(b);
			pts.push_back(c);
			tri = Triangle(pts);
			tris.push_back(tri);
			pts.clear();

			pts.push_back(a);
			pts.push_back(b);
			pts.push_back(d);
			tri = Triangle(pts);
			tris.push_back(tri);
			pts.clear();

			pts.push_back(a);
			pts.push_back(d);
			pts.push_back(c);
			tri = Triangle(pts);
			tris.push_back(tri);

			return Mesh(tris);
		}
		TEST_METHOD(is_closed)
		{
			Mesh mesh = generate_mesh();

			Microsoft::VisualStudio::CppUnitTestFramework::Assert::IsTrue(mesh.is_closed());

		};
		TEST_METHOD(is_closed2)
		{
			Mesh mesh = generate_mesh();

			Microsoft::VisualStudio::CppUnitTestFramework::Assert::IsTrue(mesh.is_closed_2());
		}
		TEST_METHOD(volume)
		{
			Mesh mesh = generate_mesh();

			Microsoft::VisualStudio::CppUnitTestFramework::Assert::AreEqual(mesh.volume(), 0.16666);

		}
		TEST_METHOD(point_position)
		{
			Mesh mesh = generate_mesh();
			Point3D d(0., 0., 0.);
			Point3D in(0.1, 0.1, 0.1);
			Point3D out(-1, -1, -1);
			Microsoft::VisualStudio::CppUnitTestFramework::Assert::AreEqual(mesh.point_position(in), 1);
			Microsoft::VisualStudio::CppUnitTestFramework::Assert::AreEqual(mesh.point_position(out), -1);
			Microsoft::VisualStudio::CppUnitTestFramework::Assert::AreEqual(mesh.point_position(d), 0);
		}
	};

}



