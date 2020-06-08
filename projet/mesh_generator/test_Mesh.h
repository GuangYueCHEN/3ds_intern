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

