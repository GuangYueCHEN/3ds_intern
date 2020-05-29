#include "stdafx.h"
#include "targetver.h"
#include "./../mesh_generator/Plan.h"
#include <vector>
#include "./../mesh_generator/Vector.h"
#include "./../mesh_generator/Point3D.h"


namespace UnitTest1 
{


	TEST_CLASS(plan_test)
	{
	public:
		TEST_METHOD(TestClassInit)
		{
			int i = 1;
			Microsoft::VisualStudio::CppUnitTestFramework::Assert::AreEqual(i,1);
		}

		TEST_METHOD(is_in_plan)
		{	
			Point3D pt(1., 1., 1.);
			Point3D pt2(1., 1., 2.);
			Vector norm = pt2 - pt;
			Plan plan(pt, norm);
			Microsoft::VisualStudio::CppUnitTestFramework::Assert::IsTrue(plan.is_in_plane(pt));
			Microsoft::VisualStudio::CppUnitTestFramework::Assert::IsFalse(plan.is_in_plane(pt2));
		}
		
	};

}
