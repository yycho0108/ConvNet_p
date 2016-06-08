#include "SoftMaxLayer.h"

__global__ void softMax_half(double* I, double* O, double max_in) {
	int i = threadIdx.x;
	O[i] = exp(I[i] - max_in);
}

void softMax(Matrix& I, Matrix& O) {
	double max_in = I.max();
	softMax_half<<<1,I.size().wh>>>(I.d_data(), O.d_data(), max_in);
	O /= O.sum();
}

SoftMaxLayer::SoftMaxLayer() {

}

SoftMaxLayer::~SoftMaxLayer() {

}
void SoftMaxLayer::setup(Size& s, int& d, int b) {
	this->s = s;
	this->d = d;
	this->b = b;

	I_b.resize(b);
	G_b.resize(b);

	for(int b_i=0;b_i<b;++b_i){
		I_b[b_i].resize(d);
		G_b[b_i].resize(d);
		for(int d_i=0;d_i<d;++d_i){
			I_b[b_i][d_i] = Matrix(s);
			G_b[b_i][d_i] = Matrix(s);
		}

	}
}

void SoftMaxLayer::FF(Single_t& I_s, Single_t& O_s, int b_i) {
	for (int i = 0; i < d; ++i) {
		softMax(I_s[i], O_s[i]);
	}
}

void SoftMaxLayer::BP(Single_t&,Single_t&,int) {
	throw "SoftMAX : BP Not Implemented";
}

void SoftMaxLayer::update() {

}
