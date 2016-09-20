CC=gcc
CXX=g++

CXXFLAGS=-std=c++11 -fopenmp -Wall

testreader: read_svmlight.cpp test_funcs.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^
