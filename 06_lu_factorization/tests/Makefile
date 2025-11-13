CXX      		:= g++
CXXFLAGS 		:= -O3 -march=native -std=c++20
THIRD_PARTY 	:= ../../third_party
EIGEN_INCLUDE 	:= $(THIRD_PARTY)/eigen
INC      		:= -I.. -I../../common/00_mx/include -I$(EIGEN_INCLUDE) -MMD -MP
SRCS	 		:= test_lu_factorization_1.cpp test_lu_factorization_2.cpp test_lu_solve.cpp
OBJS	 		:= $(patsubst %.cpp,%.o,$(SRCS))
DEPS	 		:= $(OBJS:.o=.d)
TARGETS	 		:= test_1 test_2 test_3

all: $(TARGETS)

test_1: test_lu_factorization_1.o
	$(CXX) $(CXXFLAGS) $< -o $@ 

test_2: test_lu_factorization_2.o
	$(CXX) $(CXXFLAGS) $< -o $@ 

test_3: test_lu_solve.o
	$(CXX) $(CXXFLAGS) $< -o $@ 

%.o: %.cpp 
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@

clean: 
	rm -f $(OBJS) $(DEPS) $(TARGETS)

.PHONY: all clean
-include $(DEPS)