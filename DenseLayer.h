#ifndef __DENSE_LAYER_H__
#define __DENSE_LAYER_H__

#include "Layer.h"

class DenseLayer : public Layer{
private:
	int s_i,s_o; //no depth. just no.
	Matrix W, B,
		   dW, dB,
		   dW_p, dB_p;
	Matrix Wt;
	Single_t dW_b, dB_b;
public:
	DenseLayer(int s_out); //and possibly also optimization as arg.
	~DenseLayer();
	virtual void setup(Size& s, int& d, int b);

	virtual void FF(Single_t& I, Single_t& O, int b_i);
	virtual void BP(Single_t& G_O, Single_t& G_I, int b_i);
	virtual void update();

	//virtual void save(FileStorage& f, int i);
	//virtual void load(FileStorage& f, int i);
};

#endif
