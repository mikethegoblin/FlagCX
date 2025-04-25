#ifndef FLAGCX_COST_MODEL_H
#define FLAGCX_COST_MODEL_H

#include "c2c_algo.h"
#include "flagcx.h"
#include <vector>

// typedef enum {
//     FullyConnected,
//     Ring,
// } flagcxHomoSimTopo;

// struct flagcxHomoSimInfo {
//     flagcxHomoSimTopo topo[2];
//     int npusCount[2];
//     float bandwidth[2];
//     float latency[2];
// };

class FlagCXAlgoTimeEstimator {
public:
  FlagCXAlgoTimeEstimator(flagcxDataType_t dt, int preHomoFuncLoops,
                          int heteroAndPostHomoFuncLoops,
                          std::vector<flagcxC2cHomoFunc> &preHomoFuncs,
                          std::vector<flagcxC2cHeteroFunc> &heteroFuncs,
                          std::vector<flagcxC2cHomoFunc> &postHomoFuncs,
                          struct flagcxHeteroComm *comm)
      : datatype_(dt), preHomoFuncs_(preHomoFuncs), heteroFuncs_(heteroFuncs),
        postHomoFuncs_(postHomoFuncs), comm_(comm) {}

  float GetAlgoTime();

private:
  float GetHomoAlgoTime();

  float GetHeteroAlgoTime();

  flagcxDataType_t datatype_;
  std::vector<flagcxC2cHomoFunc> &preHomoFuncs_;
  std::vector<flagcxC2cHeteroFunc> &heteroFuncs_;
  std::vector<flagcxC2cHomoFunc> &postHomoFuncs_;
  struct flagcxHeteroComm *comm_;
};

#endif