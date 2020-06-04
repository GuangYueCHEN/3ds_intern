#pragma once

class Mesh;
#include <string>

class MeshLoader{
public:
	MeshLoader(std::string filename);
	Mesh load() const;
private:
	std::string m_filename;
};

class MeshWriter{
public:
	MeshWriter(std::string filename);
	void write(const Mesh & mesh) const;
private:
	std::string m_filename;
};