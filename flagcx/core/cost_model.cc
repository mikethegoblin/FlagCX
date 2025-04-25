#include "cost_model.h"

constexpr uint64_t CHUNK_SIZE = 4ULL * 1024 * 1024;

float FlagCXAlgoTimeEstimator::GetAlgoTime() {
  float preHomoTime = GetHomoAlgoTime();
  float sendRecvTime = GetHeteroAlgoTime();
  float postHomoTime = GetHomoAlgoTime();
  return preHomoTime + sendRecvTime + postHomoTime;
}

float FlagCXAlgoTimeEstimator::GetHomoAlgoTime() {
  float maxTime = 0.0;
  // compute the time of homo collop for all clusters
  // for (int c = 0; c < comm_->clusterCount; c++) {
  //     // get vendor info
  //     int vendor = comm_->clusterVendorData[c].vendor;
  //     float time = 0.0;
  //     for (auto& func : preHomoFuncs_) {
  //         // get execution time for current homo func
  //     }
  //     maxTime = std::max(maxTime, time);
  // }
  return maxTime;
}

float FlagCXAlgoTimeEstimator::GetHeteroAlgoTime() {
  // assume 1 QP and fixed chunksize for now
  // filter out all the send operations
  // std::unordered_map<int, std::vector<flagcxC2cP2pOp&>> sendOps;
  // for (auto& func : heteroFuncs_) {
  //     std::vector<flagcxC2cP2pOp>& p2pOps = func.getP2pOps();
  //     for (auto& op : p2pOps) {
  //         if (op.isRecv_ == 0) {
  //             int rank = op.rank_;
  //             sendOps[rank].push_back(op);
  //         }
  //     }
  // }
  // for (auto it = sendOps.begin(); it != sendOps.end(); it++) {
  //     // get clusterId of this rank

  // }
}