#include "Mesh.h"
#include <vector>
#include "Triangle.h"
#include "Point3D.h"
#include "test_MeshUtils.h"

#include "constant.h"

TEST_CASE("mesh_test_is_closed")
{
  Mesh mesh = generate_mesh();

  REQUIRE(true == mesh.is_closed());
}

TEST_CASE("mesh_test_is_closed_2")
{
  Mesh mesh = generate_mesh();

  REQUIRE(true == mesh.is_closed_2());
}

TEST_CASE("mesh_test_volume")
{
  Mesh mesh = generate_mesh();

  REQUIRE(std::abs(mesh.volume() - 0.16666) < tol_for_volume_cmp);
}

TEST_CASE("mesh_test_point_position")
{
  Mesh mesh = generate_mesh();
  Point3D d(0., 0., 0.);
  Point3D in(0.1, 0.1, 0.1);
  Point3D out(-1, -1, -1);
  REQUIRE(mesh.point_position(in) == 1);
  REQUIRE(mesh.point_position(out) == -1);
  REQUIRE(mesh.point_position(d) == 0);
}

TEST_CASE("mesh_test_augmentation")
{
	Mesh mesh = generate_mesh();
	mesh.augmentation(6);
	REQUIRE(mesh.nb_triangles() == 6);
	REQUIRE(true == mesh.is_closed_2());

	Mesh mesh2 = generate_mesh();
	mesh2.augmentation(14);
	REQUIRE(mesh2.nb_triangles() == 14);
	REQUIRE(true == mesh2.is_closed_2());

}

TEST_CASE("mesh_add_point")
{
	Mesh mesh = generate_mesh();
	Point3D x({ 0.,0.,0. });
	size_t res = mesh.add_point(x);
	REQUIRE(res == 3);

	Point3D y({ 0.,5.,0. });
	res = mesh.add_point(y);
	REQUIRE(res == 4);
	mesh.remove_vertex(res);
	REQUIRE(mesh.nb_triangles() == 4);
	mesh.remove_vertex(3);
	REQUIRE(mesh.nb_triangles() == 1);
}

TEST_CASE("mesh_add_triangle")
{
	Mesh mesh = generate_mesh();

	Point3D y({ 5.,5.,5. });
	size_t index = mesh.add_point(y);
	size_t res = mesh.add_triangle({ 0,1,index });
	REQUIRE(res == 4);

	mesh.remove_triangle(res);
	REQUIRE(mesh.nb_triangles() == 4);

	mesh.remove_triangle(0);
	REQUIRE(mesh.nb_triangles() == 3);

}

TEST_CASE("mesh_get_triangles")
{
	Mesh mesh = generate_mesh();

	std::vector<size_t> res = mesh.GetTrianglesAroundVertex(0);
	REQUIRE(res.size() == 3);
	REQUIRE(res == std::vector<size_t>({0,2,3}));
	res = mesh.GetTrianglesAroundEdge(0, 1);
	REQUIRE(res.size() == 2);
	REQUIRE(res == std::vector<size_t>({0,2 }));
	res = mesh.GetTrianglesAroundTriangles({ 0,1,2 });
	REQUIRE(res.size() == 3);
	REQUIRE(res == std::vector<size_t>({ 1,2,3 }));
}


TEST_CASE("mesh_split_collapse")
{
	Mesh mesh = generate_mesh();
	mesh.SplitEdge(0, 1, Point3D({ 0.5,0.5,0. }));
	REQUIRE(mesh.nb_triangles() == 6);
	REQUIRE(mesh.is_closed_2());
	mesh.CollapseEdge(0, 4);
	REQUIRE(mesh.nb_triangles() == 4);
	REQUIRE(mesh.is_closed_2());

}

TEST_CASE("mesh_flip_edge")
{
	Mesh mesh = generate_mesh();
	mesh.remove_triangle(2);
	mesh.remove_triangle(2);
	REQUIRE(mesh.nb_triangles() == 2);
	auto res = mesh.GetTrianglesAroundEdge(1, 2);
	REQUIRE(res.size() == 2);
	mesh.flip_edge(1, 2);
	res = mesh.GetTrianglesAroundEdge(1, 2);
	REQUIRE(res.size() == 0);
	res = mesh.GetTrianglesAroundEdge(0, 3);
	REQUIRE(res.size() == 2);
}