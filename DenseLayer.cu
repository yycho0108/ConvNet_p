#include "DenseLayer.h"

DenseLayer::DenseLayer(int s_o) :
		s_o(s_o) {
}

DenseLayer::~DenseLayer() {

}

void DenseLayer::setup(Size& s, int& d, int b) {

	//assert(s.w == 1);
	//assert(d == 1);

	s_i = s.h; //column vector input
	this->d = d; // == 0
	this->b = b;

	//CAUTION:: do not use s directly for the size!
	//it is not the dimension of the kernel.

	W = Matrix::rand(s_i, s_o); //width = s_i, height = s_o
	B = Matrix::zeros(1, s_o);

	Wt = Matrix(s_o, s_i); //allocate space

	dW_p = Matrix::zeros(s_i, s_o);
	dB_p = Matrix::zeros(1, s_o);

	dW = Matrix::zeros(s_i, s_o);
	dB = Matrix::zeros(1, s_o);

	//placeholders
	I_b.resize(b);
	G_b.resize(b);

	dW_b.resize(b);
	dB_b.resize(b);

	for(int b_i=0;b_i<b;++b_i){
		I_b[b_i].push_back(Matrix(1,s_i));
		G_b[b_i].push_back(Matrix(1,s_i));

		dW_b[b_i] = Matrix(s_i, s_o);
		dB_b[b_i] = Matrix(1,s_o);

	}

	s = Size(1,s_o);
	d = 1;
}

void DenseLayer::FF(Single_t& I_s, Single_t& O_s, int b_i) {
	auto& I = I_b[b_i];
	I_s[0].copyTo(I[0]);
	O_s[0] = W * I_s[0] + B;
}

void DenseLayer::BP(Single_t& G_O, Single_t& G_I, int b_i) {
	auto& I = I_b[b_i];
	auto& dW = dW_b[b_i];

	//TODO : implement fancy optimizations
	G_I[0] = Wt * G_O[0];
	dW = G_O[0] * Matrix::transpose(I[0]);
}

void DenseLayer::update(){

	dW.zero();
	dB.zero();
	for(int b_i=0;b_i<b;++b_i){
		dW += dW_b[b_i];
		dB += dB_b[b_i];
	}
	//dW /= 128.0;
	//dB /= 128.0;

	W += (dW_p * MOMENTUM) + \
		 (dW * ETA) - \
		 (W * DECAY);
	B += (dB_p * MOMENTUM) + \
		 (dW * ETA);

	dW.copyTo(dW_p);
	dB.copyTo(dB_p);


	//update transpose...
	Wt = Matrix::transpose(W);
}
