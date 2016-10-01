LIBS = -lpixelLab_linux -lpng -lz -lGL -lGLU
LFOLDER = -Llib/
IFOLDER = -Iinclude/
FLAGS = -O0 -g3 -std=c++11

all:
	mpic++ $(FLAGS) src/log-edges.cc -o bin/log-edges $(IFOLDER) $(LFOLDER) $(LIBS)