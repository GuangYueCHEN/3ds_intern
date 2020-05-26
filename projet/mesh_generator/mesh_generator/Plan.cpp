#include"Plan.h"

Plan::Plan(Point3D pt1, Point3D pt2, Point3D pt3) {
	Vector vec1(pt1, pt2);
	Vector vec2(pt3, pt2);
	Vector norm = vec1 ^ vec2;
	if (norm.norme_infini() == 0) {
		throw "error: can not built a plan";
	}else {
		d = -pt1.get_x()*norm.get_x() - pt1.get_y()*norm.get_y() - pt1.get_z()*norm.get_z();
		a = norm.get_x();
		b = norm.get_y();
		c = norm.get_z();
	}
}
Plan::Plan(Point3D pt1, Vector norm) {
	d = -pt1.get_x()*norm.get_x() - pt1.get_y()*norm.get_y() - pt1.get_z()*norm.get_z();
	a = norm.get_x();
	b = norm.get_y();
	c = norm.get_z();
}
bool Plan::plan_support(Point3D pt) {
	if (pt.get_x()*a + pt.get_y()*b + pt.get_z()*c + d == 0) {
		return true;
	}
	else {
		return false;
	}
}
Vector Plan::normal() {
	Vector norm(a, b, c);
	return norm;
}

/*x-x1,y-y1,z-z1*/
/*x2,y2,z2*/
/*x2x-x2x1+y2y-y2y1+z2z-z2z1*/