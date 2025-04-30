#include "cost_model.h"
#include "topo.h"

constexpr size_t CHUNK_SIZE = 4ULL * 1024 * 1024;
const float flagcxLatMap[FLAGCX_VENDOR_NUM][2] = {
    {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};

flagcxResult_t FlagCXAlgoTimeEstimator::GetAlgoTime(float *time) {
  float preHomoTime, heteroTime, postHomoTime;
  FLAGCXCHECK(GetPreHomoAlgoTime(&preHomoTime));
  FLAGCXCHECK(GetHeteroAlgoTime(&heteroTime));
  FLAGCXCHECK(GetPostHomoAlgoTime(&postHomoTime));
  *time = preHomoTime + heteroTime + postHomoTime;
  return flagcxSuccess;
}

flagcxResult_t FlagCXAlgoTimeEstimator::GetPreHomoAlgoTime(float *time) {
  flagcxComm_t comm = planner_.getComm();
  auto &preHomoFuncs =
      planner_.getPreHomoFuncs(); // all clusters perform the same algo
  float totalPreHomoTime = 0.0;
  // compute the execution time for all clusters
  // use the max time for all clusters
  for (int i = 0; i < comm->nclusters; i++) {
    int vendor = comm->clusterVendorMap[i];
    int clusterRankSize =
        comm->cluster_sizes[i]; // get how many ranks are in this cluster
    float preHomoTimeForCluster = 0.0;
    for (auto &func : preHomoFuncs) {
      float algoTime = 0.0;
      FLAGCXCHECK(GetHomoAlgoTime(func, clusterRankSize, vendor, &algoTime));
      preHomoTimeForCluster += algoTime;
    }
    // get the max time for all clusters
    totalPreHomoTime = std::max(totalPreHomoTime, preHomoTimeForCluster);
  }
  *time = totalPreHomoTime;
  return flagcxSuccess;
}

flagcxResult_t FlagCXAlgoTimeEstimator::GetPostHomoAlgoTime(float *time) {
  flagcxComm_t comm = planner_.getComm();
  auto &postHomoFuncs = planner_.getPostHomoFuncs();
  float totalPostHomoTime = 0.0;
  // compute the execution time for all clusters
  // use the max time for all clusters
  for (int i = 0; i < comm->nclusters; i++) {
    int vendor = comm->clusterVendorMap[i];
    int clusterRankSize =
        comm->cluster_sizes[i]; // get how many ranks are in this cluster
    float postHomoTimeForCluster = 0.0;
    for (auto &func : postHomoFuncs) {
      float algoTime = 0.0;
      FLAGCXCHECK(GetHomoAlgoTime(func, clusterRankSize, vendor, &algoTime));
      postHomoTimeForCluster += algoTime;
    }
    // get the max time for all clusters
    totalPostHomoTime = std::max(totalPostHomoTime, postHomoTimeForCluster);
  }
  *time = totalPostHomoTime;
  return flagcxSuccess;
}

flagcxResult_t FlagCXAlgoTimeEstimator::GetHomoAlgoTime(
    flagcxC2cHomoFunc &homoFunc, int rankSize, int vendor, float *time) {
  float defaultTime = 0.0;
  *time = defaultTime;
  return flagcxSuccess;
}

flagcxResult_t FlagCXAlgoTimeEstimator::GetHomoInterAlgoTime(int loop,
                                                             float *time) {
  flagcxComm_t comm = planner_.getComm();
  auto &homoFunc = planner_.getHomoInterFuncs()[loop];
  // getHomoAlgoTime
  float totalHomoInterTime = 0.0;
  // compute the execution time for all clusters
  // use the max time for all clusters
  for (int i = 0; i < comm->nclusters; i++) {
    int vendor = comm->clusterVendorMap[i];
    int clusterInterRankSize = planner_.getClusterInterRankList()[i].size();
    float homoInterTimeForCluster = 0.0;
    FLAGCXCHECK(GetHomoAlgoTime(homoFunc, clusterInterRankSize, vendor,
                                &homoInterTimeForCluster));
    totalHomoInterTime = std::max(totalHomoInterTime, homoInterTimeForCluster);
  }
  *time = 0.0;
  return flagcxSuccess;
}

float FlagCXAlgoTimeEstimator::GetRefreshTime() {
  return 0.0; // return fixed time for now
}

flagcxResult_t FlagCXAlgoTimeEstimator::GetHeteroAlgoTime(float *time) {
  flagcxComm_t comm = planner_.getComm();
  flagcxHeteroComm_t heteroComm = comm->hetero_comm;
  // filter out hetero funcs for each rank
  std::unordered_map<int, std::vector<flagcxC2cHeteroFunc>> heteroFuncMap;
  int heteroFuncLoops = planner_.getHeteroAndHomoInterFuncLoops();
  auto &clusterInterRankList = planner_.getClusterInterRankList();
  auto &interRankBufferInfoManager = planner_.getInterRankBufferInfoManager();
  // get all interRanks
  std::vector<int> interRanks;
  std::unordered_map<uint64_t, std::vector<int>>
      nicRankMap; // {nicGuid: vector<rankId>} record the ranks that share the
                  // same nic
  for (size_t j = 0; j < clusterInterRankList.size(); j++) {
    for (size_t z = 0; z < clusterInterRankList[j].size(); z++) {
      int rank = clusterInterRankList[j][z];
      interRanks.push_back(rank);
      struct flagcxTopoServer *server;
      struct flagcxTopoNode *net;
      // get server of current rank
      FLAGCXCHECK(flagcxTopoGetServerFromRank(rank, heteroComm->interServerTopo,
                                              heteroComm->topoServer, &server));
      // get local nic used by current rank
      FLAGCXCHECK(flagcxTopoGetLocalNetNode(server, rank, &net));
      nicRankMap[net->net.guid].push_back(rank);
    }
  }
  for (int &rank : interRanks) {
    heteroFuncMap[rank].resize(heteroFuncLoops);
    for (int i = 0; i < heteroFuncLoops; i++) {
      flagcxC2cHeteroFunc &heteroFunc = heteroFuncMap[rank][i];
      for (size_t j = 0; j < clusterInterRankList.size(); j++) {
        for (size_t z = 0; z < clusterInterRankList[j].size(); z++) {
          if (rank == clusterInterRankList[j][z]) {
            auto &rankList =
                interRankBufferInfoManager.getBufferInfoList(j, rank);
            for (auto it = rankList.begin(); it != rankList.end(); it++) {
              if (it->loopId_ == i) {
                heteroFunc.addP2pOp(rank, it->peerRank_, it->offset_,
                                    it->count_, it->isRecv_);
              }
            }
          }
        }
      }
    }
  }
  float totalTime = 0.0;
  for (int i = 0; i < heteroFuncLoops; i++) {
    // get total send/recv time for each nic in case multiple gpus share a nic
    float timePerLoop = 0.0;
    timePerLoop += GetRefreshTime();
    float sendRecvTime = 0.0;
    for (auto it = nicRankMap.begin(); it != nicRankMap.end(); it++) {
      uint64_t netGuid = it->first;
      // total p2p time of a nic
      float p2pTime = GetP2pTimePerNic(netGuid, nicRankMap, heteroFuncMap);
      sendRecvTime = std::max(sendRecvTime, p2pTime);
    }
    timePerLoop += sendRecvTime;
    float homoInterTime = 0.0;
    FLAGCXCHECK(GetHomoInterAlgoTime(i, &homoInterTime));
    timePerLoop += homoInterTime;
    totalTime += timePerLoop;
  }

  *time = totalTime;

  return flagcxSuccess;
}

float FlagCXAlgoTimeEstimator::GetP2pTimePerNic(
    uint64_t netGuid,
    std::unordered_map<uint64_t, std::vector<int>> &nicRankMap,
    std::unordered_map<int, std::vector<flagcxC2cHeteroFunc>> &heteroFuncMap) {
  flagcxComm_t comm = planner_.getComm();
  flagcxHeteroComm_t heteroComm = comm->hetero_comm;
  auto &rankList = nicRankMap[netGuid];
  float sendTime = 0.0;
  float recvTime = 0.0;
  for (int &rank : rankList) {
    auto &funcList = heteroFuncMap[rank];
    // get clusterId of current rank
    int clusterId = comm->cluster_ids[rank];        // {rank: clusterId}
    int vendor = comm->clusterVendorMap[clusterId]; // {clusterId: vendor}
    // get cluster lat and bw
    float curClusterLat =
        flagcxLatMap[vendor][FLAGCX_INTER_LAT_IDX]; // {clusterId: lat}
    for (auto &func : funcList) {
      for (auto &p2pOp : func.getP2pOps()) {
        int remoteRank = p2pOp.peerRank_;
        int remoteClusterId = comm->cluster_ids[remoteRank];
        int remoteVendor = comm->clusterVendorMap[remoteClusterId];
        float remoteClusterLat =
            flagcxLatMap[remoteVendor][FLAGCX_INTER_LAT_IDX];
        // get nic of remote rank
        struct flagcxTopoServer *server;
        struct flagcxTopoNode *net;
        // get server of current rank
        FLAGCXCHECK(
            flagcxTopoGetServerFromRank(rank, heteroComm->interServerTopo,
                                        heteroComm->topoServer, &server));
        // get local nic used by current rank
        FLAGCXCHECK(flagcxTopoGetLocalNetNode(server, rank, &net));
        float routeBw =
            heteroComm->interServerTopo->routeMap[netGuid][net->net.guid]
                ->interBw; // we haven't recorded all route for all servers yet
        if (p2pOp.isRecv_) {
          recvTime += GetSendRecvTime(curClusterLat, remoteClusterLat, routeBw,
                                      p2pOp.count_, CHUNK_SIZE);
        } else {
          sendTime += GetSendRecvTime(curClusterLat, remoteClusterLat, routeBw,
                                      p2pOp.count_, CHUNK_SIZE);
        }
      }
    }
  }
  return std::max(sendTime, recvTime);
}

float FlagCXAlgoTimeEstimator::GetSendRecvTime(float curClusterLat,
                                               float remoteClusterLat, float bw,
                                               int totalCount,
                                               size_t chunkSize) {
  // in the current implementation, chunks are sent in serial order
  float lat =
      std::max(curClusterLat,
               remoteClusterLat); // use the higher latency between two clusters
  size_t bytes = totalCount * getFlagcxDataTypeSize(datatype);
  int steps = (bytes + chunkSize - 1) / chunkSize;
  float time = 0.0;
  int sizeSent = 0;
  for (int s = 0; s < steps; s++) {
    size_t sendSize = std::min(chunkSize, bytes - sizeSent);
    time += lat + sendSize / (1000 * bw); // convert to us (bw in GB/s)
    sizeSent += sendSize;
  }
  return time;
}