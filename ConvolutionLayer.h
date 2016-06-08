/*
 * ConvolutionLayer.h
 *
 *  Created on: May 6, 2016
 *      Author: jamiecho
 */

#ifndef __CONVOLUTION_LAYER_H__
#define __CONVOLUTION_LAYER_H__


#include "Layer.h"
#include "Matrix.h"

#include <vector>

class ConvolutionLayer: public Layer {
private:
	int d_in, d_out;
	bool** connection;
	Single_t W, B, dW_p, dB_p;
	Batch_t dW_b, dB_b;
	Single_t dW_t, dB_t;
public:
	ConvolutionLayer(int d_out=1); //# kernels
	~ConvolutionLayer();
	virtual void setup(Size&,int&,int);

	virtual void FF(Single_t& I, Single_t& O, int b_i);
	virtual void BP(Single_t& G_O, Single_t& G_I, int b_i);

	virtual void update();
	virtual void debug();
};

#endif /* CONVOLUTIONLAYER_H_ */
