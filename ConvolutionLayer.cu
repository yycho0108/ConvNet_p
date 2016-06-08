/*
 * ConvolutionLayer.cpp
 *
 *  Created on: May 6, 2016
 *      Author: jamiecho
 */

#include "ConvolutionLayer.h"


__device__ int d_abs(int x) {
	return x > 0 ? x : -x;
}

__global__ void _convolve(const double* d_i, const double* d_k, double* d_o, int r) {


	int i = threadIdx.y;
	int j = threadIdx.x;

	int h = blockDim.y;
	int w = blockDim.x;

	extern __shared__ double s_i[];
	double* s_k = &s_i[w*h];
	s_i[idx(i,j,w)] = d_i[idx(i,j,w)];

	if(i < 2*r+1 && j < 2*r+1) //within kernel index
		s_k[idx(i,j,2*r+1)] = d_k[idx(i,j,2*r+1)]; // --> is this necessary?

	__syncthreads();

	double tmp = 0;

	for (int ki = -r; ki <= r; ++ki) {
		for (int kj = -r; kj <= r; ++kj) {
			if (inbound(i + ki, j + kj, h, w)) {
				tmp += s_i[idx(i + ki, j + kj, w)]
						* s_k[idx(r - ki, r - kj, 2 * r + 1)]; //flip here if correlation
			}
			//effectively zero-padding
			//may change to VALID convolution later

			//d_o[i][j] += d_i[i+ki][j+kj] * d_k[ki+r][kj+r]
		}
	}
	d_o[idx(i,j,w)] = tmp;
}

__global__ void _correlate(const double* d_i, const double* d_k, double* d_o, int r) {

	int i = threadIdx.y;
	int j = threadIdx.x;

	int h = blockDim.y;
	int w = blockDim.x;

	extern __shared__ double s_i[];
	double* s_k = &s_i[w*h];

	s_i[idx(i,j,w)] = d_i[idx(i,j,w)];

	if(i < 2*r+1 && j < 2*r+1) //within kernel index
		s_k[idx(i,j,2*r+1)] = d_k[idx(i,j,2*r+1)];
	__syncthreads();

	double tmp = 0;
	for (int ki = -r; ki <= r; ++ki) {
		for (int kj = -r; kj <= r; ++kj) {
			if (inbound(i + ki, j + kj, h, w)) {
				tmp += s_i[idx(i + ki, j + kj, w)]
						* s_k[idx(r + ki, r + kj, 2 * r + 1)]; //flipped here, for correlation
			}
			//effectively zero-padding
			//may change to VALID convolution later

			//d_o[i][j] += d_i[i+ki][j+kj] * d_k[ki+r][kj+r]
		}
	}
	d_o[idx(i,j,w)] = tmp;
}
void convolve_d(const double* d_i, const double* d_k, double* d_o,
//if all ptrs are in gpu
		int w, int h, int r, cudaStream_t* stream) {
	dim3 g(1, 1);
	dim3 b(w, h);
	int sMemSize = sizeof(double) * (w*h + (2*r+1)*(2*r+1));
	if (stream) {
		_convolve<<<g,b,sMemSize,*stream>>>(d_i,d_k,d_o,r);
	} else {
		_convolve<<<g,b,sMemSize>>>(d_i,d_k,d_o,r);
	}

}

void correlate_d(const double* d_i, const double* d_k, double* d_o,
//if all ptrs are in gpu
		int w, int h, int r, cudaStream_t* stream) {
	dim3 g(1, 1);
	dim3 b(w, h);
	int sMemSize = sizeof(double)* (w*h + (2*r+1)*(2*r+1));
	if (stream) {
		_correlate<<<g,b,sMemSize,*stream>>>(d_i,d_k,d_o,r);
	} else {
		_correlate<<<g,b,sMemSize>>>(d_i,d_k,d_o,r);
	}
}

void convolve(Matrix& I, Matrix& K, Matrix& O, cudaStream_t* stream=nullptr) {
	//TODO : support different modes
	int w = I.size().w;
	int h = I.size().h;
	int r = K.size().w / 2;

	convolve_d(I.d_data(), K.d_data(), O.d_data(), w, h, r, stream);
}

void correlate(Matrix& I, Matrix& K, Matrix& O, cudaStream_t* stream=nullptr) {
	//TODO : support different modes
	int w = I.size().w;
	int h = I.size().h;
	int r = K.size().w / 2;

	correlate_d(I.d_data(), K.d_data(), O.d_data(), w, h, r,stream);
}

/*__device__ void submat_mul(double* a, double* b, double* o,
 int asrci, int asrcj, int aw,
 int bsrci, int bsrcj, int bw){

 auto i = threadIdx.y;
 auto j = threadIdx.x;
 auto w = blockDim.x;

 auto a_i = idx(asrci + i, asrcj + j, aw);
 auto b_i = idx(bsrci + i, bsrcj + j, bw);

 o[idx(i,j,w)] = a[a_i] * b[b_i];
 }*/


__global__ void safe_deconvolve(double* I, double* _G, double* dW, int w, int h){
	//int kw = blockDim.x;
	//int kh = blockDim.y; //kernel dimensions
	int r = blockDim.x / 2; //radius of kernel
	int ki = threadIdx.y;
	int kj = threadIdx.x;

	auto i_start = max(0, r - ki);
	auto j_start = max(0, r - kj);

	auto i_end = min(h, h + r - ki);
	auto j_end = min(w, w + r - kj);

	auto index = idx(ki,kj,2*r+1);

	dW[index] = 0;
	for(int i=i_start; i < i_end; ++i){
		for(int j=j_start; j < j_end; ++j){
			dW[index] += I[idx(i,j,w)] * _G[idx(i+(ki-r),j+(kj-r),w)];
		}
	}

}

__global__ void rapid_deconvolve(double* I, double* _G, double* dW, int w, int h){
	extern __shared__ double ddW[]; //dims same as I

	int kw = gridDim.x;
	int kh = gridDim.y;

	int kj = blockIdx.x;
	int ki = blockIdx.y;
	int r = kw / 2;
	//each block takes dW[ki][kj]

	auto i_start = max(0, r - ki);
	auto j_start = max(0, r - kj);
	auto i_end = min(h, h + r - ki);
	auto j_end = min(w, w + r - kj);
	int j = threadIdx.x;
	int i = threadIdx.y;

	auto index = idx(i,j,w);

	if(i_start<i && i<i_end && j_start<j && j<j_end)
		ddW[index] = I[index] * _G[idx(i+(ki-r),j+(kj-r),w)];
	else
		ddW[index] = 0; //out_of_bounds

	__syncthreads();

	//now accumulate ddW...
	auto n = blockDim.x * blockDim.y;
	int nTotalThreads = NearestPowerOf2(n);	// Total number of threads, rounded up to the next power of two

	while(nTotalThreads > 1)
	{
	  int halfPoint = (nTotalThreads >> 1);	// divide by two
	  // only the first half of the threads will be active.
	  if (index < halfPoint)
	  {
	   int index2 = index + halfPoint;
	   if (index2 < n)
	      ddW[index] += ddW[index2];
	  }
	  __syncthreads();
	  // Reducing the binary tree size by two:
	  nTotalThreads = halfPoint;
	}

	if(index == 0){
		//only 1 thread will write to dW
		dW[idx(ki,kj,kw)] = ddW[0]; //0 = final accumulated index
	}

}
void deconvolve(Matrix& I, Matrix& _G, Matrix& dW) {

	auto s = dW.size().w;
	auto r = dW.size().w / 2; //assume square kernel, odd-size

	if(I.size().wh > 1024){
		throw "TOO MANY THREADS!!";
	}

	//TRYING
	dim3 gridDims(s,s);
	dim3 blockDims(I.size().w, I.size().h);
	rapid_deconvolve<<<gridDims,blockDims,I.size().wh * sizeof(double)>>>(I.d_data(),_G.d_data(),dW.d_data(),I.size().w,I.size().h);
}

ConvolutionLayer::ConvolutionLayer(int d_out) : //TODO : accept kernel size
		d_out(d_out) {
	connection = nullptr;
	d_in = 0;
}

ConvolutionLayer::~ConvolutionLayer() {
	for (int i = 0; i < d_in; ++i) {
		delete connection[i];
	}
	delete[] connection;

}

void ConvolutionLayer::setup(Size& _s, int& _d, int _b) {
	//_d = depth of input
	s = _s;
	d_in = _d;
	b = _b; //batch size


	I_b.resize(b);
	G_b.resize(b);
	dW_b.resize(b);
	dB_b.resize(b);

	for(int b_i=0; b_i<b; ++b_i){
		auto& I = I_b[b_i];
		auto& G = G_b[b_i];
		auto& dW = dW_b[b_i];
		auto& dB = dB_b[b_i];

		I.resize(d_in);
		G.resize(d_out);

		for (int i = 0; i < d_in; ++i) {
			I[i] = Matrix(s);
			G[i] = Matrix::zeros(s);

		}

		dW.resize(d_out);
		dB.resize(d_out);
		for(int o = 0; o < d_out; ++o){
			dW[o] = Matrix::zeros(5,5);
			dB[o] = Matrix::zeros(s);
		}


	}

	for (int o = 0; o < d_out; ++o) {
		//weight
		W.push_back(Matrix::rand(5, 5)); //5,5 = kernel size
		dW_p.push_back(Matrix::zeros(5, 5)); //previous dW
		dW_t.push_back(Matrix::zeros(5,5));

		//bias
		B.push_back(Matrix::zeros(s));
		dB_p.push_back(Matrix::zeros(s)); //previous dB
		dB_t.push_back(Matrix::zeros(s));
	}

	connection = new bool*[d_out];

	for (int o = 0; o < d_out; ++o) {
		connection[o] = new bool[d_in];
		for (int i = 0; i < d_in; ++i) {
			//connection[o][i] = true;
			connection[o][i] = ((o % 3) == (i % 3));
			//partial connection
		}
	}
	_s = s; //same size, at least for current convolution function.
	_d = d_out;


}

void ConvolutionLayer::FF(Single_t& I_s, Single_t& O_s, int b_i) {

	for (int i = 0; i < d_in; ++i) {
		I_s[i].copyTo(I_s[i]);
	}

	Matrix tmp = Matrix(O_s[0].size());

	for (int o = 0; o < d_out; ++o) {
		O_s[o].zero(); //set to zero
		for (int i = 0; i < d_in; ++i) {
			if (connection[o][i]) { //TODO : parallelize
				convolve(I_s[i], W[o], tmp);
				O_s[o] += tmp;
			}
		}
		O_s[o] += B[o]; //add bias
	}
}

void ConvolutionLayer::BP(Single_t& G_O, Single_t& G_I, int b_i) {

	auto& I = I_b[b_i];
	auto& dW = dW_b[b_i];
	auto& dB = dB_b[b_i];

	auto iw = s.w;
	auto ih = s.h;

	auto fw = W[0].size().w; //kernel size
	auto fh = W[0].size().h;

	auto fwr = fw / 2; //kernel size
	auto fhr = fh / 2;

	for (int i = 0; i < d_in; ++i) {
		G_I[i].zero(); //reset to 0
	}

	Matrix dG(G_I[0].size()); //TODO : make this static?
	Matrix ddW(dW[0].size()); //there are acculumants.

	for (int o = 0; o < d_out; ++o) { //for each output channel(depth):
		dW[o].zero();
	}

	for (int o = 0; o < d_out; ++o) { //for each output channel(depth):
		correlate(G_O[o], W[o], dG);
		for (int i = 0; i < d_in; ++i) { //for each input channel
			if (connection[o][i]) { //if the channels are related..
				G_I[i] += dG;
				deconvolve(I[i], G_O[o], ddW);
				dW[o] += ddW; //accum
			}
		}
	}
}

void ConvolutionLayer::update() {
	//accum dW, dB...
	//TODO : optimize this addition
	for(int o=0; o<d_out; ++o){
		dW_t[o].zero();
		dB_t[o].zero();
		for(int b_i=0;b_i<b;++b_i){
			dW_t[o] += dW_b[b_i][o];
			dB_t[o] += dB_b[b_i][o];
		}
	}

	//minibatch-like
	for (int o = 0; o < d_out; ++o) {
		//dW_t[o] /= 128.0;
		//dB_t[o] /= 128.0; //batch size

		W[o] += (dW_p[o] * MOMENTUM) + \
				(dW_t[o] * ETA) - \
				(W[o] * DECAY);
		B[o] += (dB_p[o] * MOMENTUM) + \
				(dB_t[o] * ETA);
		dW_t[o].copyTo(dW_p[o]);
		dB_t[o].copyTo(dB_p[o]);

		dW_t[o].zero();
		dB_t[o].zero();
	}
}

void ConvolutionLayer::debug(){
	for(int o=0;o<d_out;++o){
		W[o].print(std::cout);
	}
}
