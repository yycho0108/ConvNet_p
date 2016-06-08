#include "ConvNet.h"

//RMS error
double RMS(Matrix& m){
	return sqrt((m%m).avg());
	//square -> mean -> root
}

ConvNet::ConvNet(){

}

ConvNet::~ConvNet(){
	for(auto& l : L){
		delete l;
	}
}

struct ff_info{
	int b_i; //batch id
	int s_i; //sample id
	std::vector<Layer*> *L;
	Batch_t *I;
	Batch_t *T;
};

void* FF_wrap(void* args){

	ff_info* info = (ff_info*) args;
	int b_i = info->b_i; //batch id
	int s_i = info->s_i; //sample id
	auto& I = (*info->I)[s_i];
	auto& T = (*info->T)[s_i]; //put output here
	//numeric parameters

	int nL = info->L->size();

	int d_in = I.size();
	int d_out = T.size();

	std::vector<Layer*>& L = *(info->L);

	//copy input

	for(int d_i=0;d_i<d_in;++d_i){
		I[d_i].copyTo(L[0]->I_b[b_i][d_i]);
	}

	// FF...
	for(int l=0;l<nL;++l){
		if(l+1 < nL){ // = has next layer
			Single_t& I = L[l]->I_b[b_i];
			Single_t& O = L[l+1]->I_b[b_i];
			L[l]->FF(I,O,b_i); //here use local O
		}else{ //last layer
			Single_t& I = L[l]->I_b[b_i];
			L[l]->FF(I,T,b_i); //place result inside T (which is actually output)
		}
	}

	cudaStreamSynchronize(0);
	pthread_exit(NULL);
}

void* FFBP_wrap(void* args){
	ff_info* info = (ff_info*) args;
	int b_i = info->b_i; //batch id
	int s_i = info->s_i; //sample id
	auto& I = (*info->I)[s_i];
	auto& T = (*info->T)[s_i];
	//numeric parameters

	int nL = info->L->size();

	int d_in = I.size();
	int d_out = T.size();

	//dummy objects
	Single_t O = T; //assign T simply to allocating memory

	std::vector<Layer*>& L = *(info->L);

	//copy input

	for(int d_i=0;d_i<d_in;++d_i){
		I[d_i].copyTo(L[0]->I_b[b_i][d_i]);
	}

	// FF...
	for(int l=0;l<nL;++l){
		if(l+1 < nL){ // = has next layer
			Single_t& I = L[l]->I_b[b_i];
			Single_t& O = L[l+1]->I_b[b_i];
			L[l]->FF(I,O,b_i); //here use local O
		}else{ //last layer
			Single_t& I = L[l]->I_b[b_i];
			L[l]->FF(I,O,b_i); //here, use global O
		}
	}

	// BP...
	for(int d_i=0; d_i<d_out; ++d_i){
		L[nL-1]->G_b[b_i][d_i] = T[d_i] - O[d_i];
	}

	for(auto l = nL-2; l >= 0; --l){
		auto& G_O = L[l+1]->G_b[b_i];
		auto& G_I = L[l]->G_b[b_i]; //here use local G_I
		L[l]->BP(G_O,G_I,b_i);
	}

	cudaStreamSynchronize(0);
	pthread_exit(NULL);
}

void ConvNet::FFBP(Batch_t& _I, Batch_t& _T, std::vector<int>& indices){
	int n = indices.size(); // == batch_size

	pthread_t* threads = new pthread_t[n];
	ff_info* info = new ff_info[n];

	for(int i=0;i<n;++i){
		info[i] = {i,indices[i],&L,&_I,&_T};
		pthread_create(&threads[i],nullptr,FFBP_wrap,(void*) &info[i]);
	}

	for(int i=0;i<n;++i){
		pthread_join(threads[i],NULL);
	}

	delete[] threads;
	delete[] info;
}

void ConvNet::FF(Batch_t& _I, Batch_t& _O, std::vector<int>& indices){
	int n = indices.size();

	pthread_t* threads = new pthread_t[n];
	ff_info* info = new ff_info[n];

	for(int i=0;i<n;++i){
		info[i] = {i,indices[i],&L,&_I,&_O};
		pthread_create(&threads[i],nullptr,FF_wrap,(void*) &info[i]);
	}

	for(int i=0;i<n;++i){
		pthread_join(threads[i],NULL);
	}

	delete[] threads;
	delete[] info;

}


void ConvNet::push_back(Layer*&& l){
	L.push_back(l);
	//take ownership
	l = nullptr;
}

void ConvNet::setup(Size s, int d, int b){
	for(auto& l : L){
		l->setup(s,d,b);
	}
}
void ConvNet::update(){
	for(auto& l : L){
		l->update();
	}
}
double ConvNet::error(){
	return loss;
}

void ConvNet::debug(){
	L[0]->debug();
}
