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

constexpr int FLAGCX_INTRA_LAT_IDX = 0;
constexpr int FLAGCX_INTER_LAT_IDX = 1;

#define FLAGCX_VENDOR_NUM 4

class FlagCXAlgoTimeEstimator {
public:
  FlagCXAlgoTimeEstimator(flagcxC2cPlanner &planner, flagcxDataType_t dtype)
      : planner_(planner), datatype(dtype) {}

  flagcxResult_t GetAlgoTime(float *time);

private:
  flagcxResult_t GetPreHomoAlgoTime(float *time);

  flagcxResult_t GetPostHomoAlgoTime(float *time);

  flagcxResult_t GetHomoAlgoTime(flagcxC2cHomoFunc &homoFunc, int rankSize,
                                 int vendor, float *time);

  flagcxResult_t GetHeteroAlgoTime(float *time);

  flagcxResult_t GetHomoInterAlgoTime(int loop, float *time);

  void GenerateHeteroFuncForMultiNic(int rank, int loop,
                                     flagcxC2cHeteroFunc &heteroFunc);

  void GenerateHeteroFuncForSingleNic(int rank,
                                      flagcxC2cHeteroFunc &heteroFunc);

  float GetP2pTimePerNic(
      uint64_t netGuid,
      std::unordered_map<uint64_t, std::vector<int>> &nicRankMap,
      std::unordered_map<int, std::vector<flagcxC2cHeteroFunc>> &heteroFuncMap);

  float GetRefreshTime();

  float GetSendRecvTime(float curClusterLat, float remoteClusterLat, float bw,
                        int totalCount, size_t chunkSize);

  flagcxC2cPlanner &planner_;
  flagcxDataType_t datatype;
};

#endif