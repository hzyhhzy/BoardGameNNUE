#include "model.h"

#include <cmath>
#include "external/simde_avx2.h"
#include "external/simde_fma.h"

#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
using namespace NNUE;

float Evaluator::eval(const int* board)
{
  simde__m256i sum[featureBatchInt16];
  for (int batch = 0; batch < featureBatchInt16; batch++)
    sum[batch] = simde_mm256_setzero_si256();

  static const int featureGroupIds[25] = {
    0,1,2,1,0,
    1,3,4,3,1,
    2,4,5,4,2,
    1,3,4,3,1,
    0,1,2,1,0 };
  static const int featureSymIds[25] =
  { 3,7,5,5,1,
    3,3,5,1,1,
    2,2,0,0,0,
    2,2,4,0,0,
    2,6,4,4,0
  };
  static const int featureSymKernels[8][9] =
  {
    {
         1,    3,    9,
        27,   81,  243,
       729, 2187, 6561,
    },
    {
       729, 2187, 6561,
        27,   81,  243,
         1,    3,    9,
    },
    {
         9,    3,    1,
       243,   81,   27,
      6561, 2187,  729,
    },
    {
      6561, 2187,  729,
       243,   81,   27,
         9,    3,    1,
    },
    {
         1,   27,  729,
         3,   81, 2187,
         9,  243, 6561,
    },
    {
         9,  243, 6561,
         3,   81, 2187,
         1,   27,  729,
    },
    {
       729,   27,    1,
      2187,   81,    3,
      6561,  243,    9,
    },
    {
      6561,  243,    9,
      2187,   81,    3,
       729,   27,    1,
    },
  };

  for (int y = 0; y < 5; y++)
    for (int x = 0; x < 5; x++)
    {
      int loc = y * 7 + x;
      int feature_loc = y * 5 + x;
      int sym = featureSymIds[feature_loc];
      int feature_group = featureGroupIds[feature_loc];
      int feature_id = 
        featureSymKernels[sym][0] * board[loc + 0] + 
        featureSymKernels[sym][1] * board[loc + 1] + 
        featureSymKernels[sym][2] * board[loc + 2] + 
        featureSymKernels[sym][3] * board[loc + 7] +
        featureSymKernels[sym][4] * board[loc + 8] +
        featureSymKernels[sym][5] * board[loc + 9] +
        featureSymKernels[sym][6] * board[loc + 14] +
        featureSymKernels[sym][7] * board[loc + 15] +
        featureSymKernels[sym][8] * board[loc + 16];


      for (int batch = 0; batch < featureBatchInt8; batch++)
      {
        auto f = simde_mm256_loadu_si256((const simde__m256i*)(weights->mapping[feature_group][feature_id] + batch * 32));
        sum[2 * batch] = simde_mm256_add_epi16(sum[2 * batch], simde_mm256_cvtepi8_epi16(simde_mm256_extractf128_si256(f, 0)));
        sum[2 * batch + 1] = simde_mm256_add_epi16(sum[2 * batch + 1], simde_mm256_cvtepi8_epi16(simde_mm256_extractf128_si256(f, 1)));
      }
    }


  float layer0[featureNum];
  for (int batch = 0; batch < featureBatchInt16; batch++)
  {
    auto prelu1_w = simde_mm256_loadu_si256((const simde__m256i*)(weights->prelu1_w + batch * 16));
    auto x = sum[batch];
    x = simde_mm256_max_epi16(x, simde_mm256_mulhrs_epi16(x, prelu1_w));
    simde_mm256_storeu_ps(layer0 + batch * 16, simde_mm256_cvtepi32_ps(simde_mm256_cvtepi16_epi32(simde_mm256_extractf128_si256(x, 0))));
    simde_mm256_storeu_ps(layer0 + batch * 16 + 8, simde_mm256_cvtepi32_ps(simde_mm256_cvtepi16_epi32(simde_mm256_extractf128_si256(x, 1))));
  }
  for (int i = 0; i < featureNum; i++)std::cout << layer0[i] << " ";


  // linear 1
  float layer1[mlpChannel];
  for (int i = 0; i < mlpBatchFloat; i++) {
    auto sum = simde_mm256_loadu_ps(weights->mlp_b1 + i * 8);
    for (int j = 0; j < featureNum; j++) {
      auto x = simde_mm256_set1_ps(layer0[j]);
      auto w = simde_mm256_loadu_ps(weights->mlp_w1[j] + i * 8);
      sum = simde_mm256_fmadd_ps(w, x, sum);
    }
    sum = simde_mm256_max_ps(simde_mm256_setzero_ps(), sum);  // relu
    simde_mm256_storeu_ps(layer1 + i * 8, sum);
  }

  // linear 2
  float layer2[mlpChannel];
  for (int i = 0; i < mlpBatchFloat; i++) {
    auto sum = simde_mm256_loadu_ps(weights->mlp_b2 + i * 8);
    for (int j = 0; j < mlpChannel; j++) {
      auto x = simde_mm256_set1_ps(layer1[j]);
      auto w = simde_mm256_loadu_ps(weights->mlp_w2[j] + i * 8);
      sum = simde_mm256_fmadd_ps(w, x, sum);
    }
    sum = simde_mm256_max_ps(simde_mm256_setzero_ps(), sum);  // relu
    simde_mm256_storeu_ps(layer2 + i * 8, sum);
  }
  
  // final linear
  auto v = simde_mm256_loadu_ps(weights->mlpfinal_b);
  for (int inc = 0; inc < mlpChannel; inc++) {
    auto x = simde_mm256_set1_ps(layer2[inc]);
    auto w = simde_mm256_loadu_ps(weights->mlpfinal_w[inc]);
    v = simde_mm256_fmadd_ps(w, x, v);
  }
  float value[8];
  simde_mm256_storeu_ps(value, v);
  std::cout << "\n" << value[0] << " " << value[1] << " " << value[2] << "\n";
  return value[0]-value[1];
}


bool ModelWeight::loadParam(std::string filepath)
{
  using namespace std::filesystem;
  path ext = path(filepath).extension();
  if (ext.string() == ".bin") {
    std::ifstream cacheStream(path(filepath), std::ios::binary);
    cacheStream.read(reinterpret_cast<char*>(this), sizeof(ModelWeight));
    if (cacheStream.good()) {
      return true;
    }
    else
      return false;
  }

  path cachePath = path(filepath).replace_extension("bin");
  // Read parameter cache if exists
  if (exists(cachePath)) {
    std::ifstream cacheStream(cachePath, std::ios::binary);
    cacheStream.read(reinterpret_cast<char*>(this), sizeof(ModelWeight));
    if (cacheStream.good()) {
      return true;
    }
  }


  using namespace std;
  ifstream fs(filepath);

  string modelname;
  fs >> modelname;
  if (modelname != "vo8") {
    cout << "Wrong model type:" << modelname << endl;
    return false;
  }

  int param;
  fs >> param;
  if (param != featureNum) {
    cout << "Wrong feature num:" << param << endl;
    return false;
  }
  fs >> param;
  if (param != mlpChannel) {
    cout << "Wrong mlp channel:" << param << endl;
    return false;
  }



  string varname;

  // mapping
  fs >> varname;
  if (varname != "mapping") {
    cout << "Wrong parameter name:" << varname << endl;
    return false;
  }
  for (int i = 0; i < 6; i++)
    for (int j = 0; j < 19683; j++)
      for (int k = 0; k < featureNum; k++)
      {
        int t;
        fs >> t;
        mapping[i][j][k] = t;
      }
  

  //prelu1_w
  fs >> varname;
  if (varname != "prelu1_w") {
      cout << "Wrong parameter name:" << varname << endl;
      return false;
  }
  for (int j = 0; j < featureNum; j++)
      fs >> prelu1_w[j];

  // mlp_w1
  fs >> varname;
  if (varname != "mlp_w1") {
    cout << "Wrong parameter name:" << varname << endl;
    return false;
  }
  for (int j = 0; j < featureNum; j++)
    for (int i = 0; i < mlpChannel; i++)
        fs >> mlp_w1[j][i];

  // mlp_b1
  fs >> varname;
  if (varname != "mlp_b1") {
    cout << "Wrong parameter name:" << varname << endl;
    return false;
  }
  for (int i = 0; i < mlpChannel; i++)
    fs >> mlp_b1[i];

  // mlp_w2
  fs >> varname;
  if (varname != "mlp_w2") {
    cout << "Wrong parameter name:" << varname << endl;
    return false;
  }
  for (int j = 0; j < mlpChannel; j++)
    for (int i = 0; i < mlpChannel; i++)
      fs >> mlp_w2[j][i];

  // mlp_b2
  fs >> varname;
  if (varname != "mlp_b2") {
    cout << "Wrong parameter name:" << varname << endl;
    return false;
  }
  for (int i = 0; i < mlpChannel; i++)
    fs >> mlp_b2[i];

  // mlpfinal_w
  fs >> varname;
  if (varname != "mlpfinal_w") {
    cout << "Wrong parameter name:" << varname << endl;
    return false;
  }
  for (int j = 0; j < mlpChannel; j++)
    for (int i = 0; i < 3; i++)
      fs >> mlpfinal_w[j][i];

  // mlpfinal_b
  fs >> varname;
  if (varname != "mlpfinal_b") {
    cout << "Wrong parameter name:" << varname << endl;
    return false;
  }
  for (int i = 0; i < 3; i++)
    fs >> mlpfinal_b[i];

  for (int i = 0; i < 5; i++) {
    mlpfinal_w_for_safety[i] = 0;
    mlpfinal_b_for_safety[i] = 0;
  }

  //save bin model
  std::ofstream cacheStream(cachePath, std::ios::binary);
  cacheStream.write(reinterpret_cast<char*>(this), sizeof(ModelWeight));
  
  return true;
}

NNUE::ModelWeight::ModelWeight()
{
}

NNUE::ModelWeight::ModelWeight(std::string filepath)
{
  loadParam(filepath);
}

Evaluator::Evaluator(const ModelWeight* weights):weights(weights)
{
}

Evaluator::~Evaluator()
{
}
