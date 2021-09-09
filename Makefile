CXXFLAGS := -ggdb -std=c++17 -I lib/eigen -Wall -Wextra -O3 -march=native
CXX := g++

.PHONY: all clean profile

all: neuralnetwork

clean:
	rm -rf neuralnetwork neuralnetwork.prof