#pragma once

class Mesh;

// cube centré en 0, de côté L
class CubeGenerator{
public:
	CubeGenerator(double L);
	Mesh run() const;
private:
	double L;
};
// sphere centrée en 0, de rayon 
// https://medium.com/game-dev-daily/four-ways-to-create-a-mesh-for-a-sphere-d7956b825db4

class SphereGenerator {
public:
	SphereGenerator(double radius, size_t nb_Parallels, size_t nb_Meridians);
	Mesh run() const;
private:
	double Radius;
	size_t NbParallels, NbMeridians;
};

