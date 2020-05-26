#include "pch.h"
#include "computor.h"


double computor::run() const {
	int i;
	double pi = 0.;
	for (i = 0; i < nbIteration; i++) {
		pi += 1 / pow(i * 2 + 1, 2);

	}
	return std::sqrt(pi * 8);
}



