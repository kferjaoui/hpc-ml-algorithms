CXX := g++
CXXFLAGS := -O2 -march=native
TARGET := cpuid_cache
SRC := cache_info.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(TARGET)
