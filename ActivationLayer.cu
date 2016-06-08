#include "ActivationLayer.h"
#include "Utility.h"

double __device__ sigmoid(double x) {
	//can only be called from device
	return 1.0 / (1.0 + exp(-x));
}

double __device__ sigmoidPrime(double x) {
	x = sigmoid(x);
	return x * (1 - x);
}

double __device__ softplus(double x) {
	return log(1 + exp(x));
}

double __device__ softplusPrime(double x) {
	return sigmoid(x);
}
double __device__ ReLU(double x) {
	return x > 0 ? x : 0;
}
double __device__ ReLUPrime(double x) {
	return x > 0 ? 1 : 0;
}

double __device__ mytanh(double x) {
	//in order to enforce device function ptr.
	return tanh(x);
}

double __device__ tanhPrime(double x) {
	x = tanh(x);
	return 1 - x * x;
	//return x * (1-x);
}
void __global__ sigmoid(double* I, double* O){
	int i = threadIdx.x;
	O[i]  = 1.0 / (1.0 + exp(-I[i]));
}
void __global__ activate(double* I, double* O, dfun f) {
	//can be called from host
	int i = threadIdx.x;
	O[i] = f(I[i]);
}
void __global__ activate(double* I, double* O, dfun f, int lim) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i<lim)
		O[i] = f(I[i]);
}
void activate(Matrix& I, Matrix& O, dfun f) {

	int n_elem = I.size().wh;
	if(n_elem < 1024){
		activate<<<1, n_elem>>>
					(I.d_data(), O.d_data(), f);
	}else{
		activate<<< (n_elem+255) / 256, 256>>>
					(I.d_data(), O.d_data(), f, n_elem);
	}
}

__device__ dfun pf_sig = sigmoid;
__device__ dfun pf_sig_d = sigmoidPrime;
__device__ dfun pf_sp = softplus;
__device__ dfun pf_sp_d = softplusPrime;
__device__ dfun pf_relu = ReLU;
__device__ dfun pf_relu_d = ReLUPrime;
__device__ dfun pf_tanh = mytanh;
__device__ dfun pf_tanh_d = tanhPrime;

ActivationLayer::ActivationLayer(std::string _f) {
	for (auto& c : _f) {
		c = std::tolower(c);
	}

	if (_f == "sigmoid") {
		cudaMemcpyFromSymbol(&f, pf_sig, sizeof(dfun));
		cudaMemcpyFromSymbol(&f_d, pf_sig_d, sizeof(dfun));
	} else if (_f == "softplus") {
		cudaMemcpyFromSymbol(&f, pf_sp, sizeof(dfun));
		cudaMemcpyFromSymbol(&f_d, pf_sp_d, sizeof(dfun));
	} else if (_f == "relu") {
		cudaMemcpyFromSymbol(&f, pf_relu, sizeof(dfun));
		cudaMemcpyFromSymbol(&f_d, pf_relu_d, sizeof(dfun));
	} else if (_f == "tanh") {
		cudaMemcpyFromSymbol(&f, pf_tanh, sizeof(dfun));
		cudaMemcpyFromSymbol(&f_d, pf_tanh_d, sizeof(dfun));
	} else {
		throw "WRONG ACTIVATION FUNCTION!!";
	}

}

ActivationLayer::~ActivationLayer(){

}

void ActivationLayer::setup(Size& _s, int& _d, int _b) {
	s = _s;
	d = _d;
	b = _b;

	I_b.resize(b); //batch size
	G_b.resize(b);

	for(int b_i=0; b_i<b; ++b_i){
		auto& I = I_b[b_i];
		auto& G = G_b[b_i];
		I.resize(d); //depth
		G.resize(d);
		for (int i = 0; i < d; ++i) {
			I[i] = Matrix(s);
			G[i] = Matrix(s);
		}
	}

}

void ActivationLayer::FF(Single_t& I_s, Single_t& O_s, int b_i) {
	// I --> O
	auto& I = I_b[b_i];
	for (int i = 0; i < d; ++i) { //can be parallelized
		I_s[i].copyTo(I[i]);
		activate(I_s[i], O_s[i], f);
	}
	return;
}

void ActivationLayer::BP(Single_t& G_O, Single_t& G_I, int b_i) {
	// O --> I
	auto& I = I_b[b_i];
	for (int i = 0; i < d; ++i) {
		activate(I[i], G_I[i], f_d);
		G_I[i] %= G_O[i];
	}
	return;
}
