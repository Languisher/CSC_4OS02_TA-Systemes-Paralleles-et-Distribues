include Make_linux.inc
#include Make_msys2.inc
#include Make_osx.inc

CXX = g++-15
CXXFLAGS = -std=c++17
CXXFLAGS += -fopenmp

CXXFLAGS += -I/opt/homebrew/include
LDFLAGS += -L/opt/homebrew/lib -lSDL2 -fopenmp

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
MPI_OBJS = $(addprefix $(OUTPUT_DIR)/,ant.o fractal_land.o renderer.o window.o ant_simu_mpi.o)
SCALABILITY_ANTS ?= 1000 2500 5000 10000 20000
SCALABILITY_ITER ?= 2000
SCALABILITY_RUNS ?= 5
SCALABILITY_CSV ?= $(OUTPUT_DIR)/timing_scalability.csv
SCALABILITY_TMP ?= $(OUTPUT_DIR)/timing_scalability_tmp.csv
MODES_RUNS ?= 5
MODES_ITER ?= 800
MODES_ANTS ?= 5000
CSV_NONVEC ?= $(OUTPUT_DIR)/timing_non_vectorized.csv
CSV_VEC ?= $(OUTPUT_DIR)/timing_vectorized.csv
CSV_VEC_OMP ?= $(OUTPUT_DIR)/timing_vectorized_openmp.csv
CSV_VEC_OMP_SUBMAP ?= $(OUTPUT_DIR)/timing_vectorized_openmp_submap.csv
SECOND_RUNS ?= 5
SECOND_ITER ?= 800
SECOND_ANTS ?= 5000
MPI_RUN ?= mpirun
MPI_CXX ?= $(CXX)
MPI_CFLAGS ?= $(shell mpic++ --showme:compile 2>/dev/null)
MPI_LINK_FLAGS ?= $(shell mpic++ --showme:link 2>/dev/null)
MPI_NP ?= 4
MPI_PROCS ?= 1 2 4 8
MPI_RUNS ?= 5
MPI_ITER ?= 1000
MPI_ANTS ?= 5000
MPI_CSV ?= $(OUTPUT_DIR)/timing_mpi.csv
MPI_TMP_CSV ?= $(OUTPUT_DIR)/timing_mpi_tmp.csv
MPI_SPEEDUP_CSV ?= $(OUTPUT_DIR)/timing_mpi_speedup.csv
MPI_COMP_NP ?= 4
MPI_COMP_ANTS ?= 1000 2500 5000 10000 20000
MPI_COMP_ITER ?= 800
MPI_COMP_RUNS ?= 3
MPI_COMP_CSV ?= $(OUTPUT_DIR)/timing_mpi_compute_scalability.csv
MPI_COMP_TMP_CSV ?= $(OUTPUT_DIR)/timing_mpi_compute_tmp.csv

default:	help

all: $(ALL_BINS)

benchmark: all
	$(OUTPUT_DIR)/ant_simu.exe --benchmark --runs 10 --iterations 2000 --ants 5000 --render 0 --csv $(OUTPUT_DIR)/timing_results.csv

benchmark_non_vectorized: all
	$(OUTPUT_DIR)/ant_simu.exe --benchmark --runs $(MODES_RUNS) --iterations $(MODES_ITER) --ants $(MODES_ANTS) --vectorized 0 --simd 0 --parallel 0 --render 0 --csv $(CSV_NONVEC)

benchmark_vectorized: all
	$(OUTPUT_DIR)/ant_simu.exe --benchmark --runs $(MODES_RUNS) --iterations $(MODES_ITER) --ants $(MODES_ANTS) --vectorized 1 --simd 1 --parallel 0 --render 0 --csv $(CSV_VEC)

benchmark_vectorized_openmp: all
	$(OUTPUT_DIR)/ant_simu.exe --benchmark --runs $(MODES_RUNS) --iterations $(MODES_ITER) --ants $(MODES_ANTS) --vectorized 1 --simd 1 --parallel 1 --render 0 --csv $(CSV_VEC_OMP)

benchmark_vectorized_openmp_submap: all
	$(OUTPUT_DIR)/ant_simu.exe --benchmark --runs $(SECOND_RUNS) --iterations $(SECOND_ITER) --ants $(SECOND_ANTS) --vectorized 1 --simd 1 --parallel 1 --submap 1 --render 0 --csv $(CSV_VEC_OMP_SUBMAP)

benchmark_second_amelioration: benchmark_vectorized_openmp_submap
	@echo "Second amelioration CSV written:"
	@echo "  $(CSV_VEC_OMP_SUBMAP)"

benchmark_modes: benchmark_non_vectorized benchmark_vectorized benchmark_vectorized_openmp
	@echo "Mode CSVs written:"
	@echo "  $(CSV_NONVEC)"
	@echo "  $(CSV_VEC)"
	@echo "  $(CSV_VEC_OMP)"

mpi: $(OUTPUT_DIR)/ant_simu_mpi.exe

benchmark_mpi: mpi
	OMP_NUM_THREADS=1 $(MPI_RUN) -np $(MPI_NP) $(OUTPUT_DIR)/ant_simu_mpi.exe --benchmark --runs $(MPI_RUNS) --iterations $(MPI_ITER) --ants $(MPI_ANTS) --parallel 0 --vectorized 0 --simd 1 --threads 1 --render 0 --mpi 1 --csv $(MPI_CSV)
	@echo "MPI benchmark CSV written to $(MPI_CSV)"

benchmark_mpi_scalability: mpi | $(OUTPUT_DIR)
	@rm -f $(MPI_SPEEDUP_CSV) $(MPI_TMP_CSV)
	@echo "processes,run_total_ms,speedup,advance_time_ms,ants_ms,evaporation_ms,mpi_mark_sync_ms,mpi_evap_sync_ms" > $(MPI_SPEEDUP_CSV)
	@baseline=""; \
	for np in $(MPI_PROCS); do \
		echo "[mpi-scalability] np=$$np runs=$(MPI_RUNS) iterations=$(MPI_ITER) ants=$(MPI_ANTS)"; \
		OMP_NUM_THREADS=1 $(MPI_RUN) -np $$np $(OUTPUT_DIR)/ant_simu_mpi.exe --benchmark --runs $(MPI_RUNS) --iterations $(MPI_ITER) --ants $(MPI_ANTS) --parallel 0 --vectorized 0 --simd 1 --threads 1 --render 0 --mpi 1 --csv $(MPI_TMP_CSV); \
		line=$$(awk -F, 'NR==1{for(i=1;i<=NF;i++){if($$i=="run_total_ms") rt=i; if($$i=="advance_time_ms") adv=i; if($$i=="ants_ms") ants=i; if($$i=="evaporation_ms") evap=i; if($$i=="mpi_mark_sync_ms") ms=i; if($$i=="mpi_evap_sync_ms") es=i;} next} $$1=="avg"{print $$rt","$$adv","$$ants","$$evap","$$ms","$$es}' $(MPI_TMP_CSV)); \
		rt=$$(echo "$$line" | cut -d, -f1); \
		adv=$$(echo "$$line" | cut -d, -f2); \
		ants=$$(echo "$$line" | cut -d, -f3); \
		evap=$$(echo "$$line" | cut -d, -f4); \
		ms=$$(echo "$$line" | cut -d, -f5); \
		es=$$(echo "$$line" | cut -d, -f6); \
		if [ -z "$$baseline" ]; then baseline=$$rt; fi; \
		speedup=$$(awk -v b="$$baseline" -v t="$$rt" 'BEGIN{ if (t>0) printf "%.6f", b/t; else print "0.000000" }'); \
		echo "$$np,$$rt,$$speedup,$$adv,$$ants,$$evap,$$ms,$$es" >> $(MPI_SPEEDUP_CSV); \
	done; \
	rm -f $(MPI_TMP_CSV); \
	echo "MPI scalability CSV written to $(MPI_SPEEDUP_CSV)"

benchmark_mpi_compute_scalability: mpi | $(OUTPUT_DIR)
	@rm -f $(MPI_COMP_CSV) $(MPI_COMP_TMP_CSV)
	@echo "nb_ants,run_total_ms,speedup_vs_first,advance_time_ms,ants_ms,evaporation_ms,mpi_mark_sync_ms,mpi_evap_sync_ms,delivered_food,processes,threads" > $(MPI_COMP_CSV)
	@baseline=""; \
	for ants in $(MPI_COMP_ANTS); do \
		echo "[mpi-compute-scalability] np=$(MPI_COMP_NP) ants=$$ants runs=$(MPI_COMP_RUNS) iterations=$(MPI_COMP_ITER)"; \
		OMP_NUM_THREADS=1 $(MPI_RUN) -np $(MPI_COMP_NP) $(OUTPUT_DIR)/ant_simu_mpi.exe --benchmark --runs $(MPI_COMP_RUNS) --iterations $(MPI_COMP_ITER) --ants $$ants --parallel 0 --vectorized 0 --simd 1 --threads 1 --render 0 --mpi 1 --csv $(MPI_COMP_TMP_CSV); \
		line=$$(awk -F, 'NR==1{for(i=1;i<=NF;i++){if($$i=="nb_ants") ants=i; if($$i=="run_total_ms") rt=i; if($$i=="advance_time_ms") adv=i; if($$i=="ants_ms") am=i; if($$i=="evaporation_ms") evap=i; if($$i=="mpi_mark_sync_ms") ms=i; if($$i=="mpi_evap_sync_ms") es=i; if($$i=="delivered_food") food=i; if($$i=="mpi_processes") p=i; if($$i=="threads") th=i;} next} $$1=="avg"{print $$ants","$$rt","$$adv","$$am","$$evap","$$ms","$$es","$$food","$$p","$$th}' $(MPI_COMP_TMP_CSV)); \
		rt=$$(echo "$$line" | cut -d, -f2); \
		if [ -z "$$baseline" ]; then baseline=$$rt; fi; \
		speedup=$$(awk -v b="$$baseline" -v t="$$rt" 'BEGIN{ if (t>0) printf "%.6f", b/t; else print "0.000000" }'); \
		echo "$$line" | awk -F, -v s="$$speedup" '{print $$1","$$2","s","$$3","$$4","$$5","$$6","$$7","$$8","$$9","$$10}' >> $(MPI_COMP_CSV); \
	done; \
	rm -f $(MPI_COMP_TMP_CSV); \
	echo "MPI compute-scalability CSV written to $(MPI_COMP_CSV)"

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

$(OUTPUT_DIR)/ant_simu_mpi.o: ant_simu.cpp | $(OUTPUT_DIR)
	$(MPI_CXX) $(CXXFLAGS2) $(MPI_CFLAGS) -DUSE_MPI -c $< -o $@

$(OUTPUT_DIR)/ant_simu_mpi.exe : $(MPI_OBJS)
	$(MPI_CXX) $(CXXFLAGS2) $^ -o $@ $(LDFLAGS) $(MPI_LINK_FLAGS) $(LIB)

help:
	@echo "Available targets : "
	@echo "    all            : compile all executables"
	@echo "    mpi            : compile MPI executable ($(OUTPUT_DIR)/ant_simu_mpi.exe)"
	@echo "    benchmark      : run benchmark and write $(OUTPUT_DIR)/timing_results.csv"
	@echo "    benchmark_non_vectorized : non-vectorized mode -> $(CSV_NONVEC)"
	@echo "    benchmark_vectorized : vectorized+SIMD mode -> $(CSV_VEC)"
	@echo "    benchmark_vectorized_openmp : vectorized+SIMD+OpenMP mode -> $(CSV_VEC_OMP)"
	@echo "    benchmark_vectorized_openmp_submap : vectorized+SIMD+OpenMP + sub-map ownership -> $(CSV_VEC_OMP_SUBMAP)"
	@echo "    benchmark_second_amelioration : alias of benchmark_vectorized_openmp_submap"
	@echo "    benchmark_mpi  : run MPI benchmark -> $(MPI_CSV)"
	@echo "    benchmark_mpi_scalability : run MPI scaling + speedup -> $(MPI_SPEEDUP_CSV)"
	@echo "    benchmark_mpi_compute_scalability : run MPI compute-size scaling -> $(MPI_COMP_CSV)"
	@echo "    benchmark_modes : run all 3 modes and write their CSV files"
	@echo "    benchmark_scalability : run multiple workloads and write $(SCALABILITY_CSV)"
	@echo "Add DEBUG=yes to compile in debug"
	@echo "Configuration :"
	@echo "    CXX      :    $(CXX)"
	@echo "    CXXFLAGS :    $(CXXFLAGS)"

%.html: %.md
	pandoc -s --toc $< --css=./github-pandoc.css --metadata pagetitle="OS202 - TD1" -o $@
