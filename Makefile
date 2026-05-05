TARGET = train
BUILD_DIR = build/
SRC_DIR = src
CXX ?= g++
LDFLAGS = -lstdc++fs
LIBS = 
DEBUG ?= false

ifeq ($(CXX), g++)
	TEST_FLAGS = 
else
	TEST_FLAGS = -fp-model=precise
endif

EXTRA_FLAGS := 
ifeq ($(DEBUG), true)
	EXTRA_FLAGS := $(EXTRA_FLAGS) -DDEBUG
endif

ifeq ($(CXX), g++)
	FLAGS = -std=c++0x -Wall -Winline -Wshadow -W -O3 -fopenmp -march=native
else
	FLAGS = -std=c++0x -Wall -Winline -Wshadow -W -O3 -qopenmp -xHOST -Wno-tautological-constant-compare
endif

CXXFLAGS = $(FLAGS) $(EXTRA_FLAGS)
CXXFLAGS += -I./../../packages/eigen
-include $(OBJS:.o=.d)

SRCS = $(shell find $(SRC_DIR) -type f -name '*.cpp')

INCLUDES = $(shell find $(SRC_DIR) -type f -name '*.h')

OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/lib/%.o,$(SRCS))

all: $(OBJS)

$(BUILD_DIR)/lib/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%: $(SRC_DIR)/%.cpp $(OBJS)
	@mkdir -p $(@D)
	$()

# all: clean

# test:
# 	:

# perf: 
# 	:

# clean:
# 	$(info ==> CLEAN)
# 	@rm -rf $(BUILD_DIR)

# cleanall:
# 	$(info ==> CLEANALL)
# 	@rm -rf $(BUILD_DIR)



.PHONY: all perf clean cleanall