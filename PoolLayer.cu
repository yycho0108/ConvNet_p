#include "PoolLayer.h"

__global__ void pool(double* I, double* O, int* SW, //Switch
		int iw, int ih, //width of input matrix
		int s_w, int s_h,  //stride dims
		int p_w, int p_h){ //pool dims

	//TODO: is 'max' pooling in terms of magnitude? or positive-max only?

	int h = blockDim.y;
	int w = blockDim.x;

	int i = threadIdx.y;
	int j = threadIdx.x;

	double maxVal = -99999.0;// reasonably small value, anyways.
	int maxIdx = 0;

	int index = idx(i,j,w);
	//TODO : fix all these arbitrary numbers

	for(int ii=0;ii<p_h && s_h*i+ii < ih;++ii){ //check i+ii for bounds
		for(int jj=0;jj<p_w && s_w*j+jj < iw;++jj){ //check j+jj for bounds

			int index_i = idx(s_h*i+ii,s_w*j+jj,iw);
			double val = I[index_i];

			if(val > maxVal){
				maxIdx = index_i; //switches, stored in flattened index
				maxVal = val;
			}
		}
	}

	O[index] = maxVal;
	SW[index] = maxIdx;
}

__global__ void invert_pool(double* G_o, double* G_i, int* SW){

	int i = threadIdx.x;
	G_i[SW[i]] = G_o[i];
}

PoolLayer::PoolLayer(Size s_s, Size s_p):s_s(s_s),s_p(s_p){

}

PoolLayer::~PoolLayer(){
	for(int b_i=0;b_i<b;++b_i){
		auto& SW = SW_b[b_i];
		for(int i=0;i<d;++i){
			cudaFree(SW[i]);
		}
	}

}

void PoolLayer::setup(Size& s, int& d, int _b){
	s_in = s;
	this->d = d;
	this->b = _b;

	int w = s_in.w / s_s.w; //(s_in.w-s_p.w+s_s.w-1)/s_s.w;
	int h = s_in.h / s_s.h; //(s_in.h-s_p.h+s_s.h-1)/s_s.h;
	s_out = Size(w,h);

	G_b.resize(b);
	I_b.resize(b);
	SW_b.resize(b);

	for(int b_i=0; b_i<b; ++b_i){
		auto& SW = SW_b[b_i];
		SW.resize(d);
		I_b[b_i].resize(d);
		G_b[b_i].resize(d);

		for(int i=0;i<d;++i){
			cudaMalloc(&SW[i],sizeof(int) * w*h);
			I_b[b_i][i] = Matrix(s_in);
			G_b[b_i][i] = Matrix(s_in);

			//G.push_back(Matrix(s_in));
			//O.push_back(Matrix(s_out));

		}
	}


	s = s_out;
	//no change for d
}

void PoolLayer::FF(Single_t& I_s, Single_t& O_s, int b_i){
	auto& SW = SW_b[b_i];
	dim3 blockDims(s_out.w, s_out.h);

	for(int i=0;i<d;++i){
		pool<<<1, blockDims>>>(I_s[i].d_data(),O_s[i].d_data(),SW[i],
				s_in.w, s_in.h,
				s_s.w, s_s.h,
				s_p.w, s_p.h
				);
	}
}


void PoolLayer::BP(Single_t& G_O, Single_t& G_I, int b_i){
	auto& SW = SW_b[b_i];
	for(int i=0;i<d;++i){
		G_I[i].zero();
		invert_pool<<<1,s_out.wh>>>(G_O[i].d_data(),G_I[i].d_data(),SW[i]);
	}
}

void PoolLayer::update(){

}
