CXX ?= g++
NVCC = nvcc

BUILD_PATH = build
COMPILE_FLAGS = -O3 -std=c++14 -Wall -Wextra -g #-fsanitize=address
CUDA_FLAGS = -g -O3 -arch=sm_30
INCLUDES = -I/usr/include/GL -I include/ -I tests/ -I /usr/local/include  -I/usr/local/cuda/include
LIBS = -Llibs/ -lMathGeoLib -L/usr/local/cuda/lib64 -lcudart -lGL -lGLU -lglut -lglfw -lGLEW -lrt -lm -ldl #-lasan

TARGET      = main.out
TARGET_TEST = tests.out
SRC_DIR     = src
TEST_DIR    = tests
INCLUDE_DIR = include
OBJ_DIR     = build

CPP_FILES  = $(wildcard $(SRC_DIR)/*.cpp)
TEST_FILES = $(wildcard $(TEST_DIR)/*.cpp)
CU_FILES   = $(wildcard $(SRC_DIR)/*.cu)

H_FILES   = $(wildcard $(INCLUDE_DIR)/*.h)
CUH_FILES = $(wildcard $(INCLUDE_DIR)/*.cuh)

OBJS =  $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir $(CPP_FILES)))
OBJS += $(patsubst %.cu,$(OBJ_DIR)/%.cu.o,$(notdir $(CU_FILES)))

TESTS_OBJS =  $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir $(TEST_FILES)))
TESTS_OBJS += $(filter-out build/main.o, $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir $(CPP_FILES))))
TESTS_OBJS += $(patsubst %.cu,$(OBJ_DIR)/%.cu.o,$(notdir $(CU_FILES)))

all : $(TARGET) $(TARGET_TEST)
.PHONY: clean


$(TARGET) : $(OBJS)
		@echo "linking rule : " -o $@ $?
		$(CXX) $(COMPILE_FLAGS) $(OBJS) -o $@  $(LIBS)

$(OBJ_DIR)/%.cu.o : $(SRC_DIR)/%.cu $(CUH_FILES)
		@echo ".cu.o rule : " $@ $<
		$(NVCC) $(CUDA_FLAGS) $(INCLUDES) -c -o $@ $<

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp $(H_FILES)
		@echo ".o rule : " $@ $<
		$(CXX) $(COMPILE_FLAGS) $(INCLUDES) -c -o $@ $<

$(TARGET_TEST) : $(TESTS_OBJS)
		@echo "linking rule : " -o $@ $?
		$(CXX) $(COMPILE_FLAGS) $(TESTS_OBJS) -o $@ $(LIBS)
	  ./$(TARGET_TEST) --success

$(OBJ_DIR)/%.o : $(TEST_DIR)/%.cpp
		@echo ".o rule : " $@ $<
		$(CXX) $(COMPILE_FLAGS) $(INCLUDES) -c -o $@ $<



clean:
		@echo "CLEANING"
		@$(RM) -r $(BUILD_PATH)/*
		@$(RM)  $(TARGET_TEST) $(TARGET)
