#ifndef __ACTIVATION_LAYER_H__
#define __ACTIVATION_LAYER_H__

#include "Layer.h"
#include <string>

typedef double (*dfun)(double);

class ActivationLayer : public Layer{
private:
	dfun f;
	dfun f_d;
public:
	ActivationLayer(std::string _f);
	~ActivationLayer();
	virtual void setup(Size& s, int& d, int);

	virtual void FF(Single_t& I, Single_t& O, int i);
	virtual void BP(Single_t& G_I, Single_t& G_O, int i);

	//no need to update since to trainable parameter
	//virtual void save(FileStorage& f, int i);
	//virtual void load(FileStorage& f, int i);
};
#endif
