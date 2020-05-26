
#include "pch.h"
#include <iostream>
#include "computor.h"

using namespace std;
  

int main()
{
	computor cmt(100);
	double res = cmt.run();
    cout << "Pi with 100 iteration is " <<res <<endl; 
	system("pause");
	return 0;
}
