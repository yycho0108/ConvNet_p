#include "FlattenLayer.h"

FlattenLayer::FlattenLayer() {
	d_in=0;
}

FlattenLayer::~FlattenLayer() {

}

void FlattenLayer::setup(Size& s, int& d, int _b) {
	d_in = d;
	s_in = s;
	b = _b;
	s_out = Size(1, d * s.wh);

	stream_b.resize(b);

	I_b.resize(b);
	G_b.resize(b);

	for(int b_i=0; b_i<b; ++b_i){
		auto& stream = stream_b[b_i];
		stream.resize(d_in);
		I_b[b_i].resize(d_in);
		G_b[b_i].resize(d_in);

		for (int d_i = 0; d_i < d_in; ++d_i) {
			I_b[b_i][d_i] = Matrix(s);
			G_b[b_i][d_i] = Matrix(s);
			//cudaStreamCreate(&stream[d_i]);
		}
	}

	s = s_out;
	d = 1;
}

void FlattenLayer::FF(Single_t& I_s, Single_t& O_s, int b_i) {
	auto& stream = stream_b[b_i];
	double* o_ptr = O_s[0].d_data();

	auto sz = s_in.wh * sizeof(double);

	for (int i = 0; i < d_in; ++i) {
		double* i_ptr = I_s[i].d_data();
		cudaMemcpy(o_ptr + i*s_in.wh, i_ptr, sz, cudaMemcpyDeviceToDevice);
		//cudaMemcpyAsync(o_ptr + i*s_in.wh, i_ptr, sz,
		//		cudaMemcpyDeviceToDevice,stream[i]);
	}

	/*for(int d_i=0;d_i<d_in;++d_i){
		cudaStreamSynchronize(stream[d_i]);
	}*/
}

void FlattenLayer::BP(Single_t& G_O, Single_t& G_I, int b_i) {
	auto& stream = stream_b[b_i];
	double* g_o_ptr = G_O[0].d_data();
	auto sz = s_in.wh * sizeof(double);

	for (int i = 0; i < d_in; ++i) {
		double* g_i_ptr = G_I[i].d_data();
		cudaMemcpy(g_i_ptr, g_o_ptr + i*s_in.wh, sz,
						cudaMemcpyDeviceToDevice);
		//cudaMemcpyAsync(g_i_ptr, g_o_ptr + i*s_in.wh, sz,
		//		cudaMemcpyDeviceToDevice,stream[i]);
	}

	/*for(int d_i=0;d_i<d_in;++d_i){
		cudaStreamSynchronize(stream[d_i]);
	}*/
}

void FlattenLayer::update(){

}
