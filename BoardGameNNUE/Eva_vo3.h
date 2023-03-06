#pragma once
//#include "NNUEglobal.h"
#include <vector>
#include <string>

namespace NNUE_VO3 {
    const int featureNum   = 32;
    const int featureBatch = featureNum / 16;

    const int mlpChannel = 16;
    const int mlpBatch32 = mlpChannel / 8;

    struct ModelWeight
    {
      int16_t mapping[25][19683][featureNum];

      int16_t prelu1_w[featureNum];

      // 14  mlp
      float mlp_w1[featureNum][mlpChannel];  // shape=(inc，outc)，相同的inc对应权重相邻
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
      const NNUE_VO3::ModelWeight* weights;
      Evaluator() = delete;
      Evaluator(const NNUE_VO3::ModelWeight* weights);
      ~Evaluator();

      float eval(const int* board);//board=int[49]

    };

}  // namespace NNUE_VO3
