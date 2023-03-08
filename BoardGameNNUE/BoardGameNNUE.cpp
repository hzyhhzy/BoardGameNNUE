﻿
#include <iostream>
#include <chrono>
#include <random>
#include "model.h"
using namespace std;


inline int64_t now_ms()
{
  auto dur = std::chrono::steady_clock::now().time_since_epoch();
  return std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
}

void benchmark()
{
  NNUE::ModelWeight* model = new NNUE::ModelWeight("vov1_128c.txt");
  NNUE::Evaluator eva(model);

  int64_t testnum = 1000000;

  std::mt19937_64 prng{ uint64_t(now_ms()) };
  prng();
  prng();

  int board[50];//at least 49
  int64_t time_start = now_ms();
  float tmp = 0;

  for (int64_t i = 0; i < testnum; i++) {
    //board[prng() % 49] = prng() % 3;
    // 
    //generate random board
    for (int a = 0; a < 2; a++)
    {
      uint64_t r = prng();
      for (int b = 0; b < 25; b++)
      {
        board[a * 25 + b] = r % 3;
        r = r / 3;
      }
    }

    auto v = eva.eval(board);
    tmp += v;
  }
  if (tmp > 0)
    cout << "";//保证eva.eval不会被优化掉
  int64_t time_end = now_ms();
  double  time_used = time_end - time_start;
  cout << "NNevals = " << testnum << " Time = " << time_used / 1000.0 << " s" << endl;
  cout << "Speed = " << testnum / time_used * 1000.0 << " eval/s" << endl;

  delete model;
}




void debugeval()
{
  
  string boardstr =
    "oooooxx"
    "oooxxxx"
    ".ooxxxx"
    "o.ooxx."
    "..oooo."
    ".xo..o."
    "x.....o"
    ;

  //string boardstr = "oxxx.x.oooxoo.xxxxxoooxoo.xo.ooox..xx.oxxo.oxoox.";
  //string boardstr = ".................................................";

  NNUE::ModelWeight* model = new NNUE::ModelWeight("vov1_128c.txt");
  NNUE::Evaluator eva(model);


  int board[49];
  for (int i = 0; i < 49; i++)
  {
    char c = boardstr[i];
    if (c == '.')
      board[i] = 0;
    if (c == 'x')
      board[i] = 1;
    if (c == 'o')
      board[i] = 2;
  }

  float score = eva.eval(board);
  cout << score;
  delete model;
}

void checkeval()
{
  const int toTestNum = 10;

  NNUE::ModelWeight* model = new NNUE::ModelWeight("vov1_128c.txt");
  NNUE::Evaluator eva(model);


  string boardstrs[toTestNum] = {
    "x.o..x.ooxoxxoxo.oox..o...oxx.xxxxo.ox.xo.o.o.oo.",
    "oxoxox.x.xoxxoxoxx.xxxo.xxx.xxoxo.xoooox.xxoxoxoo",
    "oxo..xxx.ooooo.xxxo.ooo.ooox.oo.x..xox.oo.oooxo..",
    ".xxxox.ox...o...ox..oo.....oo...o.xx.xoox.ooo...o",
    "o.o.xoxx.xxx.x..xxxx.oxxoo...x.o.o.xx..xx.xo..xxx",
    "xo.xxoxx...xxox.o....ooooxo.xxoo.xxoo.xxx.xxoxx.o",
    "xxx.xxoo.o.x.oooxx...o..o...oxoo..x....o.oxoxoo.x",
    ".x..oxox.oooo...oxx.oxxoo.x..x...xooooxoo..o.xxx.",
    "o.oooxo.x...ox.xoo.o.oo.oo.oxoooxo.o.....xxxoxo.x",
    "ox.xoox.xoxoxoxox.oo.x.x.x...o.ox...xxxxooxxx.xox",
  };


  int board[49];

  for (int n = 0; n < toTestNum; n++)
  {
    string boardstr = boardstrs[n];
    for (int i = 0; i < 49; i++)
    {
      char c = boardstr[i];
      if (c == '.')
        board[i] = 0;
      if (c == 'x')
        board[i] = 1;
      if (c == 'o')
        board[i] = 2;
    }

    float score = eva.eval(board);
    cout << score << endl;
  }
  delete model;
}









int main()
{
  checkeval();
  //benchmark();
}
