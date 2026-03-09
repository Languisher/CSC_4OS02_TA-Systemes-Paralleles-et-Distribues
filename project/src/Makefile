include Make_linux.inc
#include Make_msys2.inc
#include Make_osx.inc

CXX = g++-15
CXXFLAGS = -std=c++17

CXXFLAGS += -I/opt/homebrew/include
LDFLAGS += -L/opt/homebrew/lib -lSDL2

ifdef DEBUG
CXXFLAGS += -g -O0 -Wall -fbounds-check -pedantic -D_GLIBCXX_DEBUG
CXXFLAGS2 = $(CXXFLAGS)
else
CXXFLAGS2 = ${CXXFLAGS} -O2 -march=native -Wall 
CXXFLAGS += -O3 -march=native -Wall
endif

ALL= ant_simu.exe 
OUTPUT_DIR = ../output
ALL_BINS = $(addprefix $(OUTPUT_DIR)/,$(ALL))
OBJS = $(addprefix $(OUTPUT_DIR)/,ant.o fractal_land.o renderer.o window.o ant_simu.o)
SCALABILITY_ANTS ?= 1000 2500 5000 10000 20000
SCALABILITY_ITER ?= 2000
SCALABILITY_RUNS ?= 5
SCALABILITY_CSV ?= $(OUTPUT_DIR)/timing_scalability.csv
SCALABILITY_TMP ?= $(OUTPUT_DIR)/timing_scalability_tmp.csv

default:	help

all: $(ALL_BINS)

benchmark: all
	$(OUTPUT_DIR)/ant_simu.exe --benchmark --runs 10 --iterations 2000 --ants 5000 --render 0 --csv $(OUTPUT_DIR)/timing_results.csv

benchmark_scalability: all | $(OUTPUT_DIR)
	@rm -f $(SCALABILITY_CSV) $(SCALABILITY_TMP)
	@first=1; \
	for ants in $(SCALABILITY_ANTS); do \
		echo "[scalability] ants=$$ants runs=$(SCALABILITY_RUNS) iterations=$(SCALABILITY_ITER)"; \
		$(OUTPUT_DIR)/ant_simu.exe --benchmark --runs $(SCALABILITY_RUNS) --iterations $(SCALABILITY_ITER) --ants $$ants --render 0 --csv $(SCALABILITY_TMP); \
		if [ $$first -eq 1 ]; then \
			head -n 1 $(SCALABILITY_TMP) > $(SCALABILITY_CSV); \
			first=0; \
		fi; \
		awk -F, 'NR==1{for(i=1;i<=NF;i++) if($$i=="run") run_col=i; next} $$run_col=="avg"{print}' $(SCALABILITY_TMP) >> $(SCALABILITY_CSV); \
	done; \
	rm -f $(SCALABILITY_TMP); \
	echo "Scalability CSV written to $(SCALABILITY_CSV)"

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
	@echo "    benchmark      : run benchmark and write $(OUTPUT_DIR)/timing_results.csv"
	@echo "    benchmark_scalability : run multiple workloads and write $(SCALABILITY_CSV)"
	@echo "Add DEBUG=yes to compile in debug"
	@echo "Configuration :"
	@echo "    CXX      :    $(CXX)"
	@echo "    CXXFLAGS :    $(CXXFLAGS)"

%.html: %.md
	pandoc -s --toc $< --css=./github-pandoc.css --metadata pagetitle="OS202 - TD1" -o $@
