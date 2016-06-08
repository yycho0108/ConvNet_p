#include <stdio.h>
#include <stdlib.h>
#include <string>
//#include <cuda.h>
#include <ctime>

#include "ConvNet.h"
#include "Sampler.h"
#include "Parser.h"


//static volatile bool keepTraining = true;
//static volatile bool keepTesting = true;
static bool keepTraining = true;
static bool keepTesting = true;

void intHandler(int){
	if(keepTraining){
		keepTraining = false;
	}else{
		keepTesting = false;
	}
}


void setup(ConvNet& net){
	/* ** CONV LAYER TEST ** */
	net.push_back(new ConvolutionLayer(12));
	net.push_back(new ActivationLayer("relu"));
	net.push_back(new DropoutLayer(0.5));
	net.push_back(new PoolLayer(Size(2,2),Size(2,2)));

	net.push_back(new ConvolutionLayer(16));
	net.push_back(new ActivationLayer("relu"));
	net.push_back(new DropoutLayer(0.5));
	net.push_back(new PoolLayer(Size(2,2),Size(2,2)));

	net.push_back(new FlattenLayer());
	net.push_back(new DenseLayer(84));
	net.push_back(new ActivationLayer("sigmoid"));
	net.push_back(new DenseLayer(10));
	net.push_back(new ActivationLayer("sigmoid"));
	net.push_back(new SoftMaxLayer());

	net.setup(Size(28,28), 1, 128);
}

void train_batch(ConvNet& net,
		std::vector<std::vector<Matrix>>& train_data,
		std::vector<std::vector<Matrix>>& train_labels,
		int batchSize,
		int lim){
	DropoutLayer::enable(true);
	keepTraining = true;

	std::ofstream ferr("error.csv");
	std::cout << "TRAINING FOR : " << lim << std::endl;

	std::ofstream fdebug("debug.txt");

	int len = train_data.size();
	Sampler sampler(time(0));
	for(int i=0;i<lim;++i){
		cout << "TRAINING ... " << i << endl;
		std::vector<int> indices = sampler(train_data.size(),batchSize);
		net.FFBP(train_data,train_labels,indices);
		net.update();
	}

	//ferr << std::endl;
	ferr.flush();
	ferr.close();

	keepTraining = false;
}

void test(
		ConvNet& net,
		std::vector<std::vector<Matrix>>& test_data,
		std::vector<std::vector<Matrix>>& test_labels,
		int batch_size){

	DropoutLayer::enable(false);
	keepTesting = true;

	std::vector<int> indices(batch_size);
	std::vector<std::vector<Matrix>> test_predict = test_labels; //placeholder

	int n = test_labels.size();

	for(int i=0; i*batch_size < n; i += batch_size){
		std::iota(indices.begin(),indices.end(),i);
		net.FF(test_data,test_predict,indices); //FF test batch
		int cor = 0;
		int inc = 0;

		for(int j=0;j<batch_size;++j){
			Size y;
			Size t;
			test_predict[j][0].set_sync(false); //--> to force sync
			namedPrint(test_predict[j][0]);

			//namedPrint(Y[0]);
			test_predict[j][0].max(&y);
			test_labels[j][0].max(&t);
			//namedPrint(y);
			//namedPrint(t);
			(y==t)?(++cor):(++inc);
			cout << "y[" << y.h << "]:T[" << t.h <<"]"<<endl;
			printf("%d cor, %d inc\n", cor,inc);
		}
	}

	keepTesting = false;
}

void build_dataset(
		std::vector<std::vector<Matrix>>& train_data,
		std::vector<std::vector<Matrix>>& train_labels,
		std::vector<std::vector<Matrix>>& test_data,
		std::vector<std::vector<Matrix>>& test_labels){

	train_data.clear();
	train_labels.clear();
	test_data.clear();
	test_labels.clear();

	Parser trainer("data/trainData","data/trainLabel");
	do{
		train_data.push_back(std::vector<Matrix>(1));
		train_labels.push_back(std::vector<Matrix>(1));
	}while(trainer.read(train_data.back()[0], train_labels.back()[0]));
	train_data.pop_back();
	train_labels.pop_back();

	Parser tester("data/testData","data/testLabel");
	do{
		test_data.push_back(std::vector<Matrix>(1));
		test_labels.push_back(std::vector<Matrix>(1));
	}while(tester.read(test_data.back()[0], test_labels.back()[0]));

	test_data.pop_back();
	test_labels.pop_back();

}
int main(int argc, char* argv[]){

	/*
	std::ofstream f_check("check.txt"); //checking working directory
	f_check << "CHECK " << std::endl;
	f_check.flush();
	f_check.close();
	*/
	int lim = 60000;

	if(argc > 1){
		lim = std::atoi(argv[1]);
	}

	std::vector<std::vector<Matrix>> train_data, train_labels, \
									 test_data, test_labels;

	build_dataset(train_data, train_labels, test_data, test_labels);

	ConvNet net; //128 = batch size
	setup(net);
	auto start = std::clock();
	int batch_size = 128;
	//train_batch(net,train_data,train_labels,batch_size,lim);// --> SGD
	auto end = std::clock();

	std::cout << (end-start) /CLOCKS_PER_SEC << "SECONDS ELAPSED" << std::endl;
	test(net,test_data,test_labels,batch_size);
}
