#include "DropoutLayer.h"

bool DropoutLayer::enabled = true;

DropoutLayer::DropoutLayer(double p):p(p){

}

DropoutLayer::~DropoutLayer(){

}
void DropoutLayer::setup(Size& _s, int& _d, int _b) {
	s = _s;
	d = _d;
	b = _b;

	//do not need to save I

	Mask_b.resize(b);
	I_b.resize(b);
	G_b.resize(b);

	for(int b_i=0; b_i<b; ++b_i){

		Mask_b[b_i].resize(d);
		G_b[b_i].resize(d);
		I_b[b_i].resize(d);

		for(int d_i=0; d_i<d; ++d_i){
			Mask_b[b_i][d_i] = Matrix(s);
			G_b[b_i][d_i] = Matrix(s);
			I_b[b_i][d_i] = Matrix(s);
		}
	}
}

void DropoutLayer::FF(Single_t& I_s, Single_t& O_s, int b_i) {
	auto& Mask = Mask_b[b_i];

	if(enabled){
		for (int i = 0; i < d; ++i) {
				Mask[i].randu(0.0,1.0);
				Mask[i] = (Mask[i] > p); //binary threshold
				O_s[i] = (I_s[i] % Mask[i]) * 1.0/(1.0-p); //reinforced to have consistent mean
			}
	}else{
		//disabled -- just propagate.
		for(int i=0; i<d; ++i){
			I_s[i].copyTo(O_s[i]);
		}
	}
}

void  DropoutLayer::BP(Single_t& G_O, Single_t& G_I, int b_i) {
	auto& Mask = Mask_b[b_i];
	if(enabled){
		for (int i = 0; i < d; ++i) {
			G_O[i] = G_I[i] % Mask[i];
		}
	}else{
		for(int i=0;i<d;++i){
			G_I[i].copyTo(G_O[i]);
		}
	}
}

void DropoutLayer::enable(bool d){
	enabled = d;
}
