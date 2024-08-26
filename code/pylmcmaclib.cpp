#include "pylmcma.h"

extern "C" {
	MyLMCMA* createMyLMCMA(int N, int lambda, int inseed, double sigma, double* xmean) {
			return new MyLMCMA(N, lambda, inseed, sigma, xmean);
	}
	void deleteMyLMCMA(MyLMCMA* mylmcma) {
		delete mylmcma;
	}
	void getarx(MyLMCMA* mylmcma, double* output) {
		mylmcma->get_arx(output);
	}
	void setarf(MyLMCMA* mylmcma, double* arf) {
		mylmcma->set_arfitness(arf);
	}
	void update(MyLMCMA* mylmcma) {
		mylmcma->update();
	}
};


