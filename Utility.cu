#include "Utility.h"
#include <cassert>
#include <string>

/* n < 1024 */
__global__ void _add(const double* a, const double* b, double* out) {
	int i = threadIdx.x;
	out[i] = a[i] + b[i];
}
__global__ void _sub(const double* a, const double* b, double* out) {
	int i = threadIdx.x;
	out[i] = a[i] - b[i];
}
__global__ void _mul(const double* a, const double* b, double* out) {
	int i = threadIdx.x;
	out[i] = a[i] * b[i];
}
__global__ void _div(const double* a, const double* b, double* out) {
	int i = threadIdx.x;
	out[i] = a[i] / b[i];
}

__global__ void _add(const double* a, const double b, double* out) {
	int i = threadIdx.x;
	out[i] = a[i] + b;
}
__global__ void _sub(const double* a, const double b, double* out) {
	int i = threadIdx.x;
	out[i] = a[i] - b;
}
__global__ void _mul(const double* a, const double b, double* out) {
	int i = threadIdx.x;
	out[i] = a[i] * b;
}
__global__ void _div(const double* a, const double b, double* out) {
	int i = threadIdx.x;
	out[i] = a[i] / b;
}

__global__ void _abs(const double* in, double* out) { //what if in == out? well...
	int i = threadIdx.x;
	out[i] = in[i] > 0 ? in[i] : -in[i];
}

/* n >= 1024 */
__global__ void _add(const double* a, const double* b, double* out, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
		out[i] = a[i] + b[i];
}
__global__ void _sub(const double* a, const double* b, double* out, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
		out[i] = a[i] - b[i];
}
__global__ void _mul(const double* a, const double* b, double* out, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
		out[i] = a[i] * b[i];
}
__global__ void _div(const double* a, const double* b, double* out, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
		out[i] = a[i] / b[i];
}

__global__ void _add(const double* a, const double b, double* out, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
		out[i] = a[i] + b;
}
__global__ void _sub(const double* a, const double b, double* out, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
		out[i] = a[i] - b;
}
__global__ void _mul(const double* a, const double b, double* out, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
		out[i] = a[i] * b;
}
__global__ void _div(const double* a, const double b, double* out, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
		out[i] = a[i] / b;
}

void add(const double* a, const double* b, double* o, int n) {
	if (n < 1024) {
		_add<<<1,n>>>(a,b,o);
	} else {
		int nb = (n + 255) / 256; //# of blocks
		_add<<<nb,256>>>(a,b,o,n);
	}
}
void sub(const double* a, const double* b, double* o, int n) {
	if (n < 1024) {
		_sub<<<1,n>>>(a,b,o);
	} else {
		int nb = (n + 255) / 256; //# of blocks
		_sub<<<nb,256>>>(a,b,o,n);
	}
}
void mul(const double* a, const double* b, double* o, int n) {
	if (n < 1024) {
		_mul<<<1,n>>>(a,b,o);
	} else {
		int nb = (n + 255) / 256; //# of blocks
		_mul<<<nb,256>>>(a,b,o,n);
	}
}
void div(const double* a, const double* b, double* o, int n) {
	if (n < 1024) {
		_div<<<1,n>>>(a,b,o);
	} else {
		int nb = (n + 255) / 256; //# of blocks
		_div<<<nb,256>>>(a,b,o,n);
	}
}

void add(const double* a, const double b, double* o, int n) {
	if (n < 1024) {
		_add<<<1,n>>>(a,b,o);
	} else {
		int nb = (n + 255) / 256; //# of blocks
		_add<<<nb,256>>>(a,b,o,n);
	}
}
void sub(const double* a, const double b, double* o, int n) {
	if (n < 1024) {
		_sub<<<1,n>>>(a,b,o);
	} else {
		int nb = (n + 255) / 256; //# of blocks
		_sub<<<nb,256>>>(a,b,o,n);
	}
}
void mul(const double* a, const double b, double* o, int n) {
	if (n < 1024) {
		_mul<<<1,n>>>(a,b,o);
	} else {
		int nb = (n + 255) / 256; //# of blocks
		_mul<<<nb,256>>>(a,b,o,n);
	}
}
void div(const double* a, const double b, double* o, int n) {
	if (n < 1024) {
		_div<<<1,n>>>(a,b,o);
	} else {
		int nb = (n + 255) / 256; //# of blocks
		_div<<<nb,256>>>(a,b,o,n);
	}
}

void abs(const double* in, double* out, int n) {
	//TODO : reimplement for robustness
	_abs<<<1,n>>>(in,out);
}


__global__ void gridMax(const double* arr, int n, double* b_max) { //b_sum = block-sum
	extern __shared__ double s_arr[]; //blockDim.x;

	int start = blockIdx.x * blockDim.x;
	int i = start + threadIdx.x;
	int ti = threadIdx.x;

	if (i >= n) //o.o.b
		return;

	s_arr[ti] = arr[i]; // copy to shared memory
	__syncthreads(); //guarantee complete copy

	int nt = NearestPowerOf2(blockDim.x); //num threads in block
	//reduction within block...
	for (int half = (nt >> 1); half > 0; half >>= 1) {
		if (ti < half) {
			int ti_2 = ti + half;
			if (start + ti_2 < n) { //within bounds
				s_arr[ti] = max(s_arr[ti], s_arr[ti_2]);
			}
		}
		__syncthreads();
	}
	__syncthreads();

	if (ti == 0) { // 1 per block
		b_max[blockIdx.x] = s_arr[0];
	}
}

__device__ int NearestPowerOf2 (int n)
{
  if (!n) return n;  //(0 == 2^0)

  int x = 1;
  while(x < n)
    {
      x <<= 1;
    }
  return x;
}

__global__ void gridMin(const double* arr, int n, double* b_min) { //b_sum = block-sum
	extern __shared__ double s_arr[]; //blockDim.x;

	int start = blockIdx.x * blockDim.x;
	int i = start + threadIdx.x;
	int ti = threadIdx.x;

	if (i >= n) //o.o.b
		return;

	s_arr[ti] = arr[i]; // copy to shared memory
	__syncthreads(); //guarantee complete copy

	int nt = NearestPowerOf2(blockDim.x); //num threads in block
	//reduction within block...
	for (int half = (nt >> 1); half > 0; half >>= 1) {
		if (ti < half) {
			int ti_2 = ti + half;
			if (start + ti_2 < n) { //within bounds
				s_arr[ti] = min(s_arr[ti], s_arr[ti_2]);
			}
		}
		__syncthreads();
	}
	__syncthreads();

	if (ti == 0) { // 1 per block
		b_min[blockIdx.x] = s_arr[0];
	}
}

__global__ void gridSum(const double* arr, int n, double* b_sum) { //b_sum = block-sum
	extern __shared__ double s_arr[]; //blockDim.x;
	int start = blockIdx.x * blockDim.x;
	int i = start + threadIdx.x;
	int ti = threadIdx.x;

	if (i >= n) //o.o.b
		return;

	s_arr[ti] = arr[i]; // copy to shared memory
	__syncthreads(); //guarantee complete copy

	int nt = NearestPowerOf2(blockDim.x); //num threads in block
	//reduction within block...
	for (int half = (nt >> 1); half > 0; half >>= 1) {
		if (ti < half) {
			int ti_2 = ti + half;
			if (start + ti_2 < n) { //within bounds
				s_arr[ti] += s_arr[ti_2];
			}
		}
		__syncthreads();
	}
	__syncthreads();

	if (ti == 0) { // 1 per block
		b_sum[blockIdx.x] = s_arr[0];
	}
}

double reduce(const double* d_arr, int n, std::string op) {
	assert(n < 65536);

	double* d_tmp, *d_res;
	double res = 0;

	dim3 gridDims((n + 255) / 256);
	dim3 blockDims(256);

	cudaMalloc(&d_tmp, sizeof(double) * gridDims.x);
	cudaMalloc(&d_res, sizeof(double));

	if(op == "sum"){
		gridSum<<<gridDims,blockDims,sizeof(double)*256>>>(d_arr,n,d_tmp);
		gridSum<<<1,gridDims,sizeof(double)>>>(d_tmp,gridDims.x,d_res);
	}else if(op == "min"){
		gridMin<<<gridDims,blockDims,sizeof(double)*256>>>(d_arr,n,d_tmp);
		gridMin<<<1,gridDims,sizeof(double)>>>(d_tmp,gridDims.x,d_res);
	}else if(op == "max"){
		gridMax<<<gridDims,blockDims,sizeof(double)*256>>>(d_arr,n,d_tmp);
		gridMax<<<1,gridDims,sizeof(double)>>>(d_tmp,gridDims.x,d_res);
	}

	cudaMemcpy(&res, d_res, sizeof(double), cudaMemcpyDeviceToHost);

	return res;
}
