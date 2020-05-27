#pragma once

class Mesh;

// cube centré en 0, de côté L
Mesh GenerateCube(double L);

// sphere centrée en 0, de rayon 
// https://medium.com/game-dev-daily/four-ways-to-create-a-mesh-for-a-sphere-d7956b825db4
Mesh GenerateSphere(double Radius, size_t NbParallels, size_t NbMeridians);

