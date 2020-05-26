
#include <cmath>

class computor
{
public:
	computor(size_t itr): nbIteration(itr){}

	double run() const;

private:
	size_t nbIteration;
};

