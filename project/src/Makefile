include Make_linux.inc
#include Make_msys2.inc
#include Make_osx.inc

CXX = g++-15
CXXFLAGS = -std=c++17

CXXFLAGS += -I/opt/homebrew/include
LDFLAGS += -L/opt/homebrew/lib -lSDL2

ifdef DEBUG
CXXFLAGS += -g -O0 -Wall -fbounds-check -pedantic -D_GLIBCXX_DEBUG
CXXFLAGS2 = CXXFLAGS
else
CXXFLAGS2 = ${CXXFLAGS} -O2 -march=native -Wall 
CXXFLAGS += -O3 -march=native -Wall
endif

ALL= ant_simu.exe 
OUTPUT_DIR = ../output
ALL_BINS = $(addprefix $(OUTPUT_DIR)/,$(ALL))
OBJS = $(addprefix $(OUTPUT_DIR)/,ant.o fractal_land.o renderer.o window.o ant_simu.o)

default:	help

all: $(ALL_BINS)

clean:
	@rm -fr $(OUTPUT_DIR)/*.o $(OUTPUT_DIR)/*.exe *~

$(OUTPUT_DIR):
	@mkdir -p $(OUTPUT_DIR)

$(OUTPUT_DIR)/%.o: %.cpp | $(OUTPUT_DIR)
	$(CXX) $(CXXFLAGS2) -c $< -o $@

$(OUTPUT_DIR)/ant_simu.exe : $(OBJS)
	$(CXX) $(CXXFLAGS2) $^ -o $@ $(LDFLAGS) $(LIB)

help:
	@echo "Available targets : "
	@echo "    all            : compile all executables"
	@echo "Add DEBUG=yes to compile in debug"
	@echo "Configuration :"
	@echo "    CXX      :    $(CXX)"
	@echo "    CXXFLAGS :    $(CXXFLAGS)"

%.html: %.md
	pandoc -s --toc $< --css=./github-pandoc.css --metadata pagetitle="OS202 - TD1" -o $@
