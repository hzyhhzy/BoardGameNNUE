// BoardGameNNUE.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <chrono>
#include <random>
#include "Eva_vo3.h"
using namespace std;


inline int64_t now_ms()
{
  auto dur = std::chrono::steady_clock::now().time_since_epoch();
  return std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
}

void benchmark()
{
  Eva_vo3 eva;
  eva.loadParam("vo3_3.txt");

  int64_t testnum = 5000000;

  std::mt19937_64 prng{ uint64_t(now_ms()) };
  prng();
  prng();

  int board[50];//at least 49
  int64_t time_start = now_ms();
  int tmp = 0;

  // 平均每play和undo两次，然后eval一次
  for (int64_t i = 0; i < testnum; i++) {

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
  }
  int64_t time_end = now_ms();
  double  time_used = time_end - time_start;
  cout << "NNevals = " << testnum << " Time = " << time_used / 1000.0 << " s" << endl;
  cout << "Speed = " << testnum / time_used * 1000.0 << " eval/s" << endl;
}




void testeval()
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
    


  Eva_vo3 eva;
  eva.loadParam("vo3_3.txt");


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

}







int main()
{
  benchmark();
}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
