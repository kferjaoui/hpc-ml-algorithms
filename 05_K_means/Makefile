CXX = g++

CXXFLAGS = -std=c++20 -Wall -O3

TARGET = kmeans

# Source files
SOURCES = main.cpp utils.cpp kmeansSerial.cpp

# Object files (generated from source files)
OBJECTS = $(SOURCES:.cpp=.o)

# Default target
all: $(TARGET)

# Link object files to create the executable
$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $(TARGET)

# Compile source files to object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up generated files
clean:
	rm -f $(OBJECTS) $(TARGET) *.log *.png

# Phony targets (not actual files)
.PHONY: all clean