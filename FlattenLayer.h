#ifndef __FLATTEN_LAYER_H__
#define __FLATTEN_LAYER_H__


#include "Layer.h"

class FlattenLayer : public Layer {
private:
	int d_in; //d_out = 1
	Size s_in,s_out;
	std::vector<std::vector<cudaStream_t>> stream_b;
public:
	FlattenLayer();
	~FlattenLayer();
	virtual void setup(Size&,int&, int);//int for "depth" of previous.
	virtual void FF(Single_t& I, Single_t& O, int b_i);
	virtual void BP(Single_t& G_O, Single_t& G_I, int b_i);
	virtual void update();

	//virtual void save(FileStorage& f, int i)=0;
	//virtual void load(FileStorage& f, int i)=0;
};

#endif
