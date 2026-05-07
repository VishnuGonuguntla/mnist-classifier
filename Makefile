# ==============================================================================
# Makefile for MNIST Classification Project
# ==============================================================================

CXX       := g++
CXXFLAGS  := -std=c++20 -Wall -Wextra -O2
EIGEN_DIR := $(HOME)/packages/eigen
INCLUDES  := -I$(EIGEN_DIR) -Isrc

# ------------------------------------------------------------------------------
# Directories
# ------------------------------------------------------------------------------
SRC_DIR := src
OBJ_DIR := obj
BIN_DIR := bin

# ------------------------------------------------------------------------------
# Shared object files (used by multiple executables)
# ------------------------------------------------------------------------------
DATAREADER_OBJ := $(OBJ_DIR)/DataReader.o
HELPER_OBJ     := $(OBJ_DIR)/helper.o
NEURALNET_OBJ  := $(OBJ_DIR)/NeuralNet.o
MNIST_OBJ      := $(OBJ_DIR)/mnist.o

# Compile each shared source
$(OBJ_DIR)/DataReader.o: $(SRC_DIR)/DataReader.cpp $(SRC_DIR)/DataReader.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
	@echo "[Compiled] $<"

$(OBJ_DIR)/helper.o: $(SRC_DIR)/helper.cpp $(SRC_DIR)/helper.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
	@echo "[Compiled] $<"

$(OBJ_DIR)/NeuralNet.o: $(SRC_DIR)/NeuralNet.cpp $(SRC_DIR)/NeuralNet.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
	@echo "[Compiled] $<"

$(OBJ_DIR)/mnist.o: $(SRC_DIR)/mnist.cpp $(SRC_DIR)/mnist.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
	@echo "[Compiled] $<"

# ==============================================================================
# Executables
# ==============================================================================

.PHONY: all clean dirs

all: dirs \
	$(BIN_DIR)/read_dataset_images \
	$(BIN_DIR)/read_dataset_labels \
	$(BIN_DIR)/mnist

# Create output directories if they don't exist
dirs:
	@mkdir -p $(OBJ_DIR) $(BIN_DIR)

# ------------------------------------------------------------------------------
# read_dataset_images
# Needs DataReader + helper for parse_image()
# ------------------------------------------------------------------------------
$(OBJ_DIR)/read_dataset_images.o: $(SRC_DIR)/read_dataset_images.cpp $(SRC_DIR)/DataReader.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
	@echo "[Compiled] $<"

$(BIN_DIR)/read_dataset_images: $(OBJ_DIR)/read_dataset_images.o $(DATAREADER_OBJ) $(HELPER_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@
	@echo "[Linked]   $@"

# ------------------------------------------------------------------------------
# read_dataset_labels
# Needs DataReader + helper for parse_label()
# ------------------------------------------------------------------------------
$(OBJ_DIR)/read_dataset_labels.o: $(SRC_DIR)/read_dataset_labels.cpp $(SRC_DIR)/DataReader.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
	@echo "[Compiled] $<"

$(BIN_DIR)/read_dataset_labels: $(OBJ_DIR)/read_dataset_labels.o $(DATAREADER_OBJ) $(HELPER_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@
	@echo "[Linked]   $@"

# ------------------------------------------------------------------------------
# mnist (main classifier)
# Needs everything: NeuralNet, DataReader, mnist, helper
# ------------------------------------------------------------------------------
$(OBJ_DIR)/main.o: $(SRC_DIR)/main.cpp $(SRC_DIR)/NeuralNet.h $(SRC_DIR)/mnist.h $(SRC_DIR)/DataReader.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
	@echo "[Compiled] $<"

$(BIN_DIR)/mnist: $(OBJ_DIR)/main.o $(NEURALNET_OBJ) $(MNIST_OBJ) $(DATAREADER_OBJ) $(HELPER_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@
	@echo "[Linked]   $@"

# ------------------------------------------------------------------------------
# Clean build artifacts
# ------------------------------------------------------------------------------
clean:
	@rm -rf $(OBJ_DIR) $(BIN_DIR)
	@echo "[Cleaned]  obj/ and bin/ removed."