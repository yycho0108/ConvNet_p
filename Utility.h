/*
 * Utility.h
 *
 *  Created on: May 6, 2016
 *      Author: jamiecho
 */

#ifndef UTILITY_H_
#define UTILITY_H_

#include <cuda_runtime.h>
#include <string>
typedef cudaStream_t cudaStream_t;


#define idx(i,j,w) ((i)*(w)+(j))
#define inbound(i,j,n,m) (0<=(i) && 0<=(j) && (i)<(n) && (j)<(m))
#define LOOP(start,end,content) \
	for(int _it = start; _it < end; ++_it){ \
		content; \
	} \

#define namedPrint(x) \
	std::cout << #x << " : " << x << std::endl;

#define hline() \
	std::cout << "------------------" << std::endl;

extern void add(const double* a, const double* b, double* o, int n);
extern void sub(const double* a, const double* b, double* o, int n);
extern void mul(const double* a, const double* b, double* o, int n);
extern void div(const double* a, const double* b, double* o, int n);
extern void add(const double* a, const double b, double* o, int n);
extern void sub(const double* a, const double b, double* o, int n);
extern void mul(const double* a, const double b, double* o, int n);
extern void div(const double* a, const double b, double* o, int n);
extern void abs(const double* in, double* out, int n);

extern __device__ int NearestPowerOf2 (int n);

extern double reduce(const double* d_arr, int n, std::string op);

#endif /* UTILITY_H_ */
