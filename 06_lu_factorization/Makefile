CXX      := g++
CXXFLAGS := -O3 -march=native -std=c++20
INC      := -I. -I../common/00_mx/include -MMD -MP

SRCS      := test_lu_correctness.cpp
OBJS      := $(patsubst %.cpp,%.o,$(SRCS))
DEPS      := $(OBJS:.o=.d)
TARGET    := LU_test

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(INC) $< -o $@ 

%.o: %.cpp 
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@

clean: 
	rm -f $(OBJS) $(DEPS) $(TARGET)

.PHONY: all clean
-include $(DEPS)