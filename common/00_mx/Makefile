CXX      := g++
CXXFLAGS := -O3 -std=c++20 
INC      := -Iinclude

TEST_BIN := dense_basic_test

CPU_SRC := tests/dense_basic.test.cpp

CPU_OBJ := $(CPU_SRC:.cpp=.o)

all: $(TEST_BIN)

$(TEST_BIN): $(CPU_OBJ)
	$(CXX) $(CXXFLAGS) $(INC) $^ -o $@ 

tests/%.o: tests/%.cpp 
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@

clean: 
	rm -f $(TEST_BIN) $(CPU_OBJ)

.PHONY: all clean
