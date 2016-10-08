LIBS = -lpixelLab_linux -lpng -lz -lGL -lGLU
LIBF = lib/
INCLF = include/
SRCF = src/
BINF = bin/
FLAGS = -O0 -g3 -std=c++11

SEQS = $(SRCF)sequential
PARS = $(SRCF)parallel

SEQB = $(BINF)sequential
PARB = $(BINF)parallel

all:
	mpic++ $(FLAGS) $(SEQS)/log-edges.cc -o $(SEQB)/log-edges -I$(INCLF) -L$(LIBF) $(LIBS)
	mpic++ $(FLAGS) $(PARS)/log-edges.cc -o $(PARB)/log-edges -I$(INCLF) -L$(LIBF) $(LIBS)