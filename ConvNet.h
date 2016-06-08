#ifndef __CONVNET_H__
#define __CONVNET_H__

#include "Layer.h"
#include "ActivationLayer.h"
#include "ConvolutionLayer.h"
#include "DenseLayer.h"
#include "DropoutLayer.h"
#include "FlattenLayer.h"
#include "PoolLayer.h"
#include "SoftMaxLayer.h"
#include <vector>

class ConvNet{

private:
	std::vector<Layer*> L; //layers. set as ptr, to avoid copying-syntax when pushing
	double loss; //most recent loss
public:
	ConvNet();
	~ConvNet();

	void FF(Batch_t& _I, Batch_t& _O, std::vector<int>& indices);
	void FFBP(Batch_t& _I, Batch_t& _T, std::vector<int>& indices);

	void setup(Size s, int d, int b); //size & depth & batchsize of input
	void push_back(Layer*&& l);
	void update();
	double error();
	void debug();
	//void save(std::string dir);//save directory
	//void load(std::string dir);
};
#endif
