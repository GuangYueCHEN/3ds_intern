#pragma once

class Mesh;

// cube centr� en 0, de c�t� L
Mesh GenerateCube(double L);

// sphere centr�e en 0, de rayon 
// https://medium.com/game-dev-daily/four-ways-to-create-a-mesh-for-a-sphere-d7956b825db4
Mesh GenerateSphere(double Radius, size_t NbParallels, size_t NbMeridians);

