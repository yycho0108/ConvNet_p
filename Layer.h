/*
 * Layer.h
 *
 *  Created on: May 6, 2016
 *      Author: jamiecho
 */

#ifndef __LAYER_H_
#define __LAYER_H_

#include <vector>
#include <pthread.h>

#include "Size.h"
#include "Utility.h"
#include "Params.h"
#include "Matrix.h"

using Single_t = std::vector<Matrix>;
using Batch_t = std::vector<Single_t>;

class Layer {
	friend class ConvNet;
	friend void* FFBP_wrap(void*);
	friend void* FF_wrap(void*);
	friend void* BP_wrap(void*);
protected:
	Size s; //size
	int d; //depth
	int b; //batch size
	Batch_t I_b; //inputs
	Batch_t G_b; //gradients
public:
	Layer(); //no need for constructor
	virtual ~Layer();

	virtual void setup(Size&,int&,int)=0;//int for "depth" of previous.

	virtual void FF(Single_t&, Single_t&, int)=0; // --> for batch-learn
	virtual void BP(Single_t&, Single_t&, int)=0; // --> for batch-learn

	virtual void update();
	virtual void debug();
	//TODO : implement save-load logic
	//virtual void save(FileStorage& f, int i)=0;
	//virtual void load(FileStorage& f, int i)=0;
};

#endif /* LAYER_H_ */
