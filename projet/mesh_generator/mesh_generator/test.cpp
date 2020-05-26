#include <iostream>

using namespace std;
const int MAX = 3;

int main()
{
	int  var[MAX] = { 10, 100, 200 };

	for (int i = 0; i < MAX; i++)
	{
		*var = i;    // 这是正确的语法

	}
	return 0;
}