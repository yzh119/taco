FLAGS=-O3

all: sddmm

sddmm: sddmm_csr_gpu_taco.cu
	nvcc ${FLAGS} -L${TACO_HOME}/build/lib/ -I${TACO_HOME}/include -o sddmm_csr_gpu_taco sddmm_csr_gpu_taco.cu -ltaco -arch=sm_75

clean:
	rm -f sddmm_csr_gpu_taco
