#pragma once
#include <vector>
#include <string>

namespace NNUE {
    const int featureNum   = 32;
    const int featureBatchInt8 = featureNum / 32;
    const int featureBatchInt16 = featureNum / 16;

    const int mlpChannel = 16;
    const int mlpBatchFloat = mlpChannel / 8;

    struct ModelWeight
    {
      int8_t mapping[6][19683][featureNum];

      int16_t prelu1_w[featureNum];

      // mlp
      float mlp_w1[featureNum][mlpChannel];  
      float mlp_b1[mlpChannel];
      float mlp_w2[mlpChannel][mlpChannel];
      float mlp_b2[mlpChannel];
      float mlpfinal_w[ mlpChannel][3];
      float mlpfinal_w_for_safety[5];  // mlp_w3在read的时候一次read
                                   // 8个，会read到后续内存mlp_w3[valueNum-1][2]+5，
      float mlpfinal_b[3];
      float mlpfinal_b_for_safety[5];  // mlp_b3在read的时候一次read
                                   // 8个，会read到后续内存mlp_b3[2]+5，

      bool loadParam(std::string filepath);
      ModelWeight();
      ModelWeight(std::string filepath);
    };



    class Evaluator
    {
    public:
      const NNUE::ModelWeight* weights;
      Evaluator() = delete;
      Evaluator(const NNUE::ModelWeight* weights);
      ~Evaluator();

      float eval(const int* board);//board=int[49]

    };

}  // namespace NNUE
