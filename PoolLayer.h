#ifndef __POOL_LAYER_H__
#define __POOL_LAYER_H__

#include "Layer.h"

class PoolLayer : public Layer{
private:
	Size s_in, s_out;
	Size s_s,s_p; //pooling size, stride size

	std::vector<std::vector<int*>> SW_b;

public:
	PoolLayer(Size s_s, Size s_p);
	~PoolLayer();
	virtual void setup(Size& s, int& d, int b);

	virtual void FF(Single_t& I, Single_t& O, int b_i);
	virtual void BP(Single_t& G_O, Single_t& G_I, int b_i);
	virtual void update();

	//virtual void save(FileStorage& f, int i);
	//virtual void load(FileStorage& f, int i);
};

#endif
