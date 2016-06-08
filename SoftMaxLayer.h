#ifndef __SOFTMAX_LAYER_H__
#define __SOFTMAX_LAYER_H__

#include "Layer.h"


class SoftMaxLayer: public Layer{
private:
	Size s;
	int d;
public:
	SoftMaxLayer();
	~SoftMaxLayer();
	virtual void setup(Size& s, int& d, int b);
	virtual void FF(Single_t& I, Single_t& O, int b_i);
	virtual void BP(Single_t& G_O, Single_t& G_I, int b_i);
	virtual void update();

	//virtual void save(FileStorage& f, int i);
	//virtual void load(FileStorage& f, int i);
};

#endif
