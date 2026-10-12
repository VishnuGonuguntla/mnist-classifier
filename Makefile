# ==============================================================================
# Makefile for MNIST Classification Project
# ==============================================================================

CXX       := g++
CXXFLAGS  := -std=c++20 -Wall -Wextra -O3
EIGEN_DIR := $(HOME)/packages/eigen-3.4.0
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
HELPER_OBJ     := $(OBJ_DIR)/Helper.o
NEURALNET_OBJ  := $(OBJ_DIR)/NeuralNet.o

# Compile each shared source
$(OBJ_DIR)/DataReader.o: $(SRC_DIR)/DataReader.cpp $(SRC_DIR)/DataReader.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
	@echo "[Compiled] $<"

$(OBJ_DIR)/Helper.o: $(SRC_DIR)/Helper.cpp $(SRC_DIR)/Helper.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
	@echo "[Compiled] $<"

$(OBJ_DIR)/NeuralNet.o: $(SRC_DIR)/NeuralNet.cpp $(SRC_DIR)/NeuralNet.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
	@echo "[Compiled] $<"

# ==============================================================================
# Executables
# ==============================================================================

.PHONY: all clean dirs

.DEFAULT_GOAL := all

all: dirs \
	$(BIN_DIR)/read_dataset_images \
	$(BIN_DIR)/read_dataset_labels \
	$(BIN_DIR)/mnist

# Create output directories if they don't exist
dirs:
	@mkdir -p $(OBJ_DIR) $(BIN_DIR)

# ------------------------------------------------------------------------------
# read_dataset_images
# Needs DataReader + Helper for parse_image()
# ------------------------------------------------------------------------------
$(OBJ_DIR)/read_dataset_images.o: $(SRC_DIR)/read_dataset_images.cpp $(SRC_DIR)/DataReader.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
	@echo "[Compiled] $<"

$(BIN_DIR)/read_dataset_images: $(OBJ_DIR)/read_dataset_images.o $(DATAREADER_OBJ) $(HELPER_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@
	@echo "[Linked]   $@"

# ------------------------------------------------------------------------------
# read_dataset_labels
# Needs DataReader + Helper for parse_label()
# ------------------------------------------------------------------------------
$(OBJ_DIR)/read_dataset_labels.o: $(SRC_DIR)/read_dataset_labels.cpp $(SRC_DIR)/DataReader.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
	@echo "[Compiled] $<"

$(BIN_DIR)/read_dataset_labels: $(OBJ_DIR)/read_dataset_labels.o $(DATAREADER_OBJ) $(HELPER_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@
	@echo "[Linked]   $@"

# ------------------------------------------------------------------------------
# mnist (main classifier)
# Needs everything: NeuralNet, DataReader, mnist, Helper
# ------------------------------------------------------------------------------
$(OBJ_DIR)/main.o: $(SRC_DIR)/main.cpp $(SRC_DIR)/NeuralNet.h $(SRC_DIR)/DataReader.h
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