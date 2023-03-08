#pragma once
#include <vector>
#include <string>

namespace NNUE {
    const int featureNum   = 128;
    const int featureBatchInt8 = featureNum / 32;
    const int featureBatchInt16 = featureNum / 16;

    const int mlpChannel1 = 8;
    const int mlpBatchFloat1 = mlpChannel1 / 8;
    const int mlpChannel2 = 64;
    const int mlpBatchFloat2 = mlpChannel2 / 8;

    struct ModelWeight
    {
      int8_t mapping[25][19683][featureNum];

      int16_t prelu1_w[featureNum];

      // mlp
      float mlp_w1[featureNum][mlpChannel1];  
      float mlp_b1[mlpChannel1];
      float mlp_w2[mlpChannel1][mlpChannel2];
      float mlp_b2[mlpChannel2];
      float mlpfinal_w[mlpChannel2];
      float mlpfinal_b;

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
