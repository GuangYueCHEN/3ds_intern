#include "pch.h"
#include <cmath>
	
double compute_pi(int itr) {
	int i;
	double pi = 0.;
	for (i = 0; i < 100; i++) {
		pi += 1 / pow(i * 2 + 1, 2);

	}
	return sqrt(pi * 8);
}
