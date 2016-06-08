#ifndef __DROPOUT_LAYER_H__
#define __DROPOUT_LAYER_H__

//TODO : Implement

#include "Layer.h"


class DropoutLayer : public Layer{
private:
	double p; //dropout probability
	static bool enabled;
	Batch_t Mask_b;
public:
	DropoutLayer(double p=0.5);
	~DropoutLayer();
	virtual void setup(Size& s, int& d, int b);
	virtual void FF(Single_t& I, Single_t& O, int b_i);
	virtual void BP(Single_t& G_O, Single_t& G_I, int b_i);
	static void enable(bool);
	//no need to update since to trainable parameter
	//virtual void save(FileStorage& f, int i);
	//virtual void load(FileStorage& f, int i);
};
#endif
