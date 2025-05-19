#include "cost_model.h"
#include "topo.h"

constexpr size_t MB_SIZE = 1024 * 1024;
constexpr size_t GB_SIZE = 1024 * 1024 * 1024;
constexpr size_t CHUNK_SIZE = 4ULL * 1024 * 1024;

std::map<flagcxVendorType,
         std::map<flagcxCommOp_t, std::map<int, std::map<size_t, float>>>>
    flagcxAlgoTimeEstimator::homoTimeMap;
// 0: Nvidia, 1: Iluvatar, 2: MLU, 3: Metax
const float flagcxLatMap[FLAGCX_VENDOR_NUM][2] = {
    {0.0, 14.0},
    {0.0, 14.0},
    {0.0, 0.0},
    {0.0, 14.0}}; // assume that latency have a negligible impact on algo time

flagcxResult_t flagcxAlgoTimeEstimator::getAlgoTime(float *time) {
  const char *enableTopoDetect = flagcxGetEnv("FLAGCX_ENABLE_TOPO_DETECT");
  const char *interServerTopoFile =
      flagcxGetEnv("FLAGCX_INTERSERVER_ROUTE_FILE");
  if (enableTopoDetect && interServerTopoFile &&
      strcmp(enableTopoDetect, "TRUE") == 0) {
    // algo time estimator depends on cluster level topology detection
    float preHomoTime, heteroTime, postHomoTime;
    INFO(FLAGCX_GRAPH, "COST_MODEL: getting time for prehomo funcs");
    FLAGCXCHECK(getPreHomoAlgoTime(&preHomoTime));
    INFO(FLAGCX_GRAPH, "COST_MODEL: getting time for hetero funcs");
    FLAGCXCHECK(getHeteroAlgoTime(&heteroTime));
    INFO(FLAGCX_GRAPH, "COST_MODEL: getting time for posthomo funcs");
    FLAGCXCHECK(getPostHomoAlgoTime(&postHomoTime));
    INFO(FLAGCX_GRAPH,
         "COST_MODEL: preHomoTime = %f, heteroTime = %f, postHomoTime = %f",
         preHomoTime, heteroTime, postHomoTime);
    *time = preHomoTime + heteroTime + postHomoTime;
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxAlgoTimeEstimator::getPreHomoAlgoTime(float *time) {
  flagcxComm_t comm = planner_.comm_;
  auto &preHomoFuncs =
      planner_.preHomoFuncList_; // all clusters perform the same algo
  float totalPreHomoTime = 0.0;
  // compute the execution time for all clusters
  // use the max time for all clusters
  for (int i = 0; i < comm->nclusters; i++) {
    flagcxVendorType vendor = comm->clusterVendorMap[i];
    int clusterRankSize =
        comm->cluster_sizes[i]; // get how many ranks are in this cluster
    float preHomoTimeForCluster = 0.0;
    for (auto &func : preHomoFuncs) {
      float algoTime = 0.0;
      FLAGCXCHECK(getHomoAlgoTime(func, clusterRankSize, vendor, &algoTime));
      preHomoTimeForCluster += algoTime;
    }
    // get the max time for all clusters
    totalPreHomoTime = std::max(totalPreHomoTime, preHomoTimeForCluster);
  }
  *time = totalPreHomoTime;
  return flagcxSuccess;
}

flagcxResult_t flagcxAlgoTimeEstimator::getPostHomoAlgoTime(float *time) {
  flagcxComm_t comm = planner_.comm_;
  auto &postHomoFuncs = planner_.postHomoFuncList_;
  float totalPostHomoTime = 0.0;
  // compute the execution time for all clusters
  // use the max time for all clusters
  for (int i = 0; i < comm->nclusters; i++) {
    flagcxVendorType vendor = comm->clusterVendorMap[i];
    int clusterRankSize =
        comm->cluster_sizes[i]; // get how many ranks are in this cluster
    float postHomoTimeForCluster = 0.0;
    for (auto &func : postHomoFuncs) {
      float algoTime = 0.0;
      FLAGCXCHECK(getHomoAlgoTime(func, clusterRankSize, vendor, &algoTime));
      postHomoTimeForCluster += algoTime;
    }
    // get the max time for all clusters
    totalPostHomoTime = std::max(totalPostHomoTime, postHomoTimeForCluster);
  }
  *time = totalPostHomoTime;
  return flagcxSuccess;
}

flagcxResult_t
flagcxAlgoTimeEstimator::getHomoAlgoTime(flagcxC2cHomoFunc &homoFunc,
                                         int rankSize, flagcxVendorType vendor,
                                         float *time) {
  flagcxComm_t comm = planner_.comm_;
  size_t totalSize = homoFunc.count_ * getFlagcxDataTypeSize(datatype);
  if (!homoFunc.isHomoInterComm_) {
    rankSize = comm->homo_ranks;
  }
  INFO(FLAGCX_GRAPH,
       "COST_MODEL: getHomoAlgoTime: vendor = %d, commOp = %d, rankSize = %d, "
       "totalSize = %ld",
       vendor, homoFunc.commOp_, rankSize, totalSize);
  float defaultTime =
      homoTimeMap[vendor][homoFunc.commOp_][rankSize][totalSize];
  *time = defaultTime;
  return flagcxSuccess;
}

flagcxResult_t flagcxAlgoTimeEstimator::getHomoInterAlgoTime(int loop,
                                                             float *time) {
  flagcxComm_t comm = planner_.comm_;
  auto &homoFunc = planner_.homoInterFuncList_[loop];
  // getHomoAlgoTime
  float totalHomoInterTime = 0.0;
  // compute the execution time for all clusters
  // use the max time for all clusters
  for (int i = 0; i < comm->nclusters; i++) {
    flagcxVendorType vendor = comm->clusterVendorMap[i];
    int clusterInterRankSize = planner_.clusterInterRankList_[i].size();
    float homoInterTimeForCluster = 0.0;
    FLAGCXCHECK(getHomoAlgoTime(homoFunc, clusterInterRankSize, vendor,
                                &homoInterTimeForCluster));
    totalHomoInterTime = std::max(totalHomoInterTime, homoInterTimeForCluster);
  }
  *time = totalHomoInterTime;
  return flagcxSuccess;
}

float flagcxAlgoTimeEstimator::getRefreshTime() {
  return 0.0; // return fixed time for now
}

flagcxResult_t flagcxAlgoTimeEstimator::getHeteroAlgoTime(float *time) {
  flagcxComm_t comm = planner_.comm_;
  flagcxHeteroComm_t heteroComm = comm->hetero_comm;
  // filter out hetero funcs for each rank
  std::unordered_map<int, std::vector<flagcxC2cHeteroFunc>> heteroFuncMap;
  int heteroFuncLoops = planner_.heteroAndHomoInterFuncLoops_;
  auto &clusterInterRankList = planner_.clusterInterRankList_;
  // get all interRanks
  std::vector<int> interRanks;
  std::unordered_map<uint64_t, std::vector<int>>
      nicRankMap; // {nicGuid: vector<rankId>} record the ranks that share the
                  // same nic
  INFO(FLAGCX_GRAPH, "COST_MODEL: fill nicRankMap");
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
      INFO(FLAGCX_GRAPH, "COST_MODEL: nicRankMap[%lx] = %d", net->net.guid,
           rank);
      nicRankMap[net->net.guid].push_back(rank);
    }
  }
  INFO(FLAGCX_GRAPH, "COST_MODEL: interRanks size = %lu", interRanks.size());
  for (int &rank : interRanks) {
    INFO(FLAGCX_GRAPH, "COST_MODEL: generating heteroFunc for rank %d", rank);
    heteroFuncMap[rank].resize(heteroFuncLoops);
    for (int i = 0; i < heteroFuncLoops; i++) {
      INFO(FLAGCX_GRAPH, "COST_MODEL: heteroFunc generation loop %d", i);
      flagcxC2cHeteroFunc &heteroFunc = heteroFuncMap[rank][i];
      if (planner_.multiNic_) {
        generateHeteroFuncForMultiNic(rank, i, heteroFunc);
      } else {
        generateHeteroFuncForSingleNic(rank, heteroFunc);
      }
    }
  }
  float totalTime = 0.0;
  for (int i = 0; i < heteroFuncLoops; i++) {
    INFO(FLAGCX_GRAPH, "COST_MODEL: heteroFunc loop %d", i);
    // get total send/recv time for each nic in case multiple gpus share a nic
    float timePerLoop = 0.0;
    timePerLoop += getRefreshTime();
    float sendRecvTime = 0.0;
    for (auto it = nicRankMap.begin(); it != nicRankMap.end(); it++) {
      uint64_t netGuid = it->first;
      // total p2p time of a nic
      float p2pTime = getP2pTimePerNic(netGuid, nicRankMap, heteroFuncMap);
      sendRecvTime = std::max(sendRecvTime, p2pTime);
    }
    timePerLoop += sendRecvTime;
    float homoInterTime = 0.0;
    INFO(FLAGCX_GRAPH, "COST_MODEL: getting homoInter time for loop %d", i);
    FLAGCXCHECK(getHomoInterAlgoTime(i, &homoInterTime));
    INFO(FLAGCX_COLL, "COST_MODEL: homoInterTime = %f", homoInterTime);
    timePerLoop += homoInterTime;
    totalTime += timePerLoop;
  }

  *time = totalTime;

  return flagcxSuccess;
}

void flagcxAlgoTimeEstimator::generateHeteroFuncForMultiNic(
    int rank, int loop, flagcxC2cHeteroFunc &heteroFunc) {
  auto &clusterInterRankList = planner_.clusterInterRankList_;
  auto &interRankBufferInfoManager = planner_.interRankBufferInfoManager_;
  for (size_t j = 0; j < clusterInterRankList.size(); j++) {
    for (size_t z = 0; z < clusterInterRankList[j].size(); z++) {
      if (rank == clusterInterRankList[j][z]) {
        auto &rankList = interRankBufferInfoManager.getBufferInfoList(j, rank);
        INFO(FLAGCX_GRAPH, "COST_MODEL: rankList size = %lu", rankList.size());
        for (auto it = rankList.begin(); it != rankList.end(); it++) {
          if (it->loopId_ == loop) {
            INFO(FLAGCX_GRAPH, "COST_MODEL: heteroFunc addP2pOp");
            heteroFunc.addP2pOp(rank, it->peerRank_, it->offset_, it->count_,
                                it->isRecv_);
          }
        }
      }
    }
  }
}

void flagcxAlgoTimeEstimator::generateHeteroFuncForSingleNic(
    int rank, flagcxC2cHeteroFunc &heteroFunc) {
  flagcxComm_t comm = planner_.comm_;
  auto &clusterInterRankList = planner_.clusterInterRankList_;
  int cid = 0;
  int clusterId = comm->cluster_ids[rank];
  int homoMyRank = comm->globalrank2homorank[rank];
  int homoRanks = comm->cluster_sizes[clusterId];
  int totalCount = planner_.totalCount_;
  for (size_t j = 0; j < clusterInterRankList.size(); ++j) {
    if (clusterId == j) {
      continue;
    }
    int homoRankToRecvFromCluster =
        (comm->globalrank2homorank[clusterInterRankList[clusterId][0]] - cid -
         1 + homoRanks) %
        homoRanks;
    if (homoMyRank == homoRankToRecvFromCluster) {
      heteroFunc.addP2pOp(rank, clusterInterRankList[j][0], 0, totalCount, 1);
    }
    int homoRankToSendToCluster =
        (comm->globalrank2homorank[clusterInterRankList[j][0]] - cid - 1 +
         comm->cluster_sizes[j]) %
        comm->cluster_sizes[j];
    int globalRankToSendToCluster =
        homoRankToSendToCluster -
        comm->globalrank2homorank[clusterInterRankList[j][0]] +
        clusterInterRankList[j][0];
    if (homoMyRank ==
        comm->globalrank2homorank[clusterInterRankList[clusterId][0]]) {
      heteroFunc.addP2pOp(rank, globalRankToSendToCluster, 0, totalCount, 0);
    }
    cid += 1;
  }
}

float flagcxAlgoTimeEstimator::getP2pTimePerNic(
    uint64_t netGuid,
    std::unordered_map<uint64_t, std::vector<int>> &nicRankMap,
    std::unordered_map<int, std::vector<flagcxC2cHeteroFunc>> &heteroFuncMap) {
  flagcxComm_t comm = planner_.comm_;
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
      for (auto &p2pOp : func.p2pOps_) {
        int remoteRank = p2pOp.peerRank_;
        int remoteClusterId = comm->cluster_ids[remoteRank];
        int remoteVendor = comm->clusterVendorMap[remoteClusterId];
        float remoteClusterLat =
            flagcxLatMap[remoteVendor][FLAGCX_INTER_LAT_IDX];
        // get nic of remote rank
        struct flagcxTopoServer *remoteServer;
        struct flagcxTopoNode *remoteNet;
        // get server of current rank
        FLAGCXCHECK(
            flagcxTopoGetServerFromRank(remoteRank, heteroComm->interServerTopo,
                                        heteroComm->topoServer, &remoteServer));
        // get local nic used by current rank
        FLAGCXCHECK(
            flagcxTopoGetLocalNetNode(remoteServer, remoteRank, &remoteNet));
        INFO(FLAGCX_GRAPH, "COST_MODEL: localNet = %lx, remoteNet = %lx",
             remoteNet->net.guid, netGuid);
        float routeBw =
            heteroComm->interServerTopo->routeMap[netGuid][remoteNet->net.guid]
                ->interBw; // we haven't recorded all route for all servers yet
        if (p2pOp.isRecv_) {
          recvTime += getSendRecvTime(curClusterLat, remoteClusterLat, routeBw,
                                      p2pOp.count_, CHUNK_SIZE);
        } else {
          sendTime += getSendRecvTime(curClusterLat, remoteClusterLat, routeBw,
                                      p2pOp.count_, CHUNK_SIZE);
        }
      }
    }
  }
  return std::max(sendTime, recvTime);
}

float flagcxAlgoTimeEstimator::getSendRecvTime(float curClusterLat,
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

void flagcxAlgoTimeEstimator::initializeHomoTimeMap() {
  // initialize
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpReduce][2][16 * MB_SIZE] =
      129.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpReduce][2][32 * MB_SIZE] =
      229.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpReduce][2][64 * MB_SIZE] =
      418.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpReduce][2][128 * MB_SIZE] =
      798.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpReduce][2][256 * MB_SIZE] =
      1554.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpReduce][2][512 * MB_SIZE] =
      3073.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpReduce][2][1 * GB_SIZE] =
      6109.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpReduce][2][2 * GB_SIZE] =
      12177.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpReduce][2][4 * GB_SIZE] =
      24363.0;

  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpReduce][4][16 * MB_SIZE] =
      142.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpReduce][4][32 * MB_SIZE] =
      255.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpReduce][4][64 * MB_SIZE] =
      466.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpReduce][4][128 * MB_SIZE] =
      893.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpReduce][4][256 * MB_SIZE] =
      1750.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpReduce][4][512 * MB_SIZE] =
      3461.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpReduce][4][1 * GB_SIZE] =
      6882.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpReduce][4][2 * GB_SIZE] =
      13725.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpReduce][4][4 * GB_SIZE] =
      27423.0;

  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpReduce][8][16 * MB_SIZE] =
      146.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpReduce][8][32 * MB_SIZE] =
      254.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpReduce][8][64 * MB_SIZE] =
      469.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpReduce][8][128 * MB_SIZE] =
      897.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpReduce][8][256 * MB_SIZE] =
      1753.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpReduce][8][512 * MB_SIZE] =
      3466.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpReduce][8][1 * GB_SIZE] =
      6882.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpReduce][8][2 * GB_SIZE] =
      13722.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpReduce][8][4 * GB_SIZE] =
      27396.0;

  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpAllReduce][2][16 * MB_SIZE] =
      180.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpAllReduce][2][32 * MB_SIZE] =
      317.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpAllReduce][2][64 * MB_SIZE] =
      580.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpAllReduce][2][128 * MB_SIZE] =
      1100.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpAllReduce][2][256 * MB_SIZE] =
      2055.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpAllReduce][2][512 * MB_SIZE] =
      3983.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpAllReduce][2][1 * GB_SIZE] =
      7671.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpAllReduce][2][2 * GB_SIZE] =
      15012.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpAllReduce][2][4 * GB_SIZE] =
      29754.0;

  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpAllReduce][4][16 * MB_SIZE] =
      235.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpAllReduce][4][32 * MB_SIZE] =
      409.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpAllReduce][4][64 * MB_SIZE] =
      722.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpAllReduce][4][128 * MB_SIZE] =
      1421.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpAllReduce][4][256 * MB_SIZE] =
      2773.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpAllReduce][4][512 * MB_SIZE] =
      5399.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpAllReduce][4][1 * GB_SIZE] =
      10554.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpAllReduce][4][2 * GB_SIZE] =
      20878.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpAllReduce][4][4 * GB_SIZE] =
      41392.0;

  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpAllReduce][8][16 * MB_SIZE] =
      293.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpAllReduce][8][32 * MB_SIZE] =
      465.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpAllReduce][8][64 * MB_SIZE] =
      845.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpAllReduce][8][128 * MB_SIZE] =
      1597.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpAllReduce][8][256 * MB_SIZE] =
      3118.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpAllReduce][8][512 * MB_SIZE] =
      6136.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpAllReduce][8][1 * GB_SIZE] =
      12123.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpAllReduce][8][2 * GB_SIZE] =
      24124.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpAllReduce][8][4 * GB_SIZE] =
      48340.0;

  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpReduceScatter][2]
             [16 * MB_SIZE] = 116.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpReduceScatter][2]
             [32 * MB_SIZE] = 191.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpReduceScatter][2]
             [64 * MB_SIZE] = 340.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpReduceScatter][2]
             [128 * MB_SIZE] = 650.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpReduceScatter][2]
             [256 * MB_SIZE] = 1250.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpReduceScatter][2]
             [512 * MB_SIZE] = 2315.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpReduceScatter][2][1 * GB_SIZE] =
      4469.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpReduceScatter][2][2 * GB_SIZE] =
      8698.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpReduceScatter][2][4 * GB_SIZE] =
      17042.0;

  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpReduceScatter][4]
             [16 * MB_SIZE] = 138.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpReduceScatter][4]
             [32 * MB_SIZE] = 234.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpReduceScatter][4]
             [64 * MB_SIZE] = 424.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpReduceScatter][4]
             [128 * MB_SIZE] = 791.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpReduceScatter][4]
             [256 * MB_SIZE] = 1484.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpReduceScatter][4]
             [512 * MB_SIZE] = 2885.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpReduceScatter][4][1 * GB_SIZE] =
      5595.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpReduceScatter][4][2 * GB_SIZE] =
      10894.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpReduceScatter][4][4 * GB_SIZE] =
      21406.0;

  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpReduceScatter][8]
             [16 * MB_SIZE] = 164.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpReduceScatter][8]
             [32 * MB_SIZE] = 258.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpReduceScatter][8]
             [64 * MB_SIZE] = 457.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpReduceScatter][8]
             [128 * MB_SIZE] = 865.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpReduceScatter][8]
             [256 * MB_SIZE] = 1624.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpReduceScatter][8]
             [512 * MB_SIZE] = 3181.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpReduceScatter][8][1 * GB_SIZE] =
      6253.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpReduceScatter][8][2 * GB_SIZE] =
      12331.0;
  homoTimeMap[FLAGCX_VENDOR_NVIDIA][flagcxCommOpReduceScatter][8][4 * GB_SIZE] =
      24391.0;

  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpReduce][2]
             [16 * MB_SIZE] = 450.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpReduce][2]
             [32 * MB_SIZE] = 888.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpReduce][2]
             [64 * MB_SIZE] = 1763.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpReduce][2]
             [128 * MB_SIZE] = 3512.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpReduce][2]
             [256 * MB_SIZE] = 7011.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpReduce][2]
             [512 * MB_SIZE] = 14008.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpReduce][2]
             [1 * GB_SIZE] = 28001.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpReduce][2]
             [2 * GB_SIZE] = 55991.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpReduce][2]
             [4 * GB_SIZE] = 111970.0;

  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpReduce][4]
             [16 * MB_SIZE] = 755.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpReduce][4]
             [32 * MB_SIZE] = 1492.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpReduce][4]
             [64 * MB_SIZE] = 2964.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpReduce][4]
             [128 * MB_SIZE] = 5909.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpReduce][4]
             [256 * MB_SIZE] = 11813.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpReduce][4]
             [512 * MB_SIZE] = 23617.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpReduce][4]
             [1 * GB_SIZE] = 47228.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpReduce][4]
             [2 * GB_SIZE] = 94358.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpReduce][4]
             [4 * GB_SIZE] = 188604.0;

  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpReduce][8]
             [16 * MB_SIZE] = 2038.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpReduce][8]
             [32 * MB_SIZE] = 4033.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpReduce][8]
             [64 * MB_SIZE] = 7962.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpReduce][8]
             [128 * MB_SIZE] = 15766.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpReduce][8]
             [256 * MB_SIZE] = 31313.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpReduce][8]
             [512 * MB_SIZE] = 62342.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpReduce][8]
             [1 * GB_SIZE] = 124147.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpReduce][8]
             [2 * GB_SIZE] = 248075.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpReduce][8]
             [4 * GB_SIZE] = 495911.0;

  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpAllReduce][2]
             [16 * MB_SIZE] = 951.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpAllReduce][2]
             [32 * MB_SIZE] = 1826.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpAllReduce][2]
             [64 * MB_SIZE] = 3576.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpAllReduce][2]
             [128 * MB_SIZE] = 7074.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpAllReduce][2]
             [256 * MB_SIZE] = 14069.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpAllReduce][2]
             [512 * MB_SIZE] = 28062.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpAllReduce][2]
             [1 * GB_SIZE] = 56048.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpAllReduce][2]
             [2 * GB_SIZE] = 112015.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpAllReduce][2]
             [4 * GB_SIZE] = 223956.0;

  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpAllReduce][4]
             [16 * MB_SIZE] = 1401.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpAllReduce][4]
             [32 * MB_SIZE] = 2725.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpAllReduce][4]
             [64 * MB_SIZE] = 5371.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpAllReduce][4]
             [128 * MB_SIZE] = 10668.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpAllReduce][4]
             [256 * MB_SIZE] = 21254.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpAllReduce][4]
             [512 * MB_SIZE] = 42442.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpAllReduce][4]
             [1 * GB_SIZE] = 84757.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpAllReduce][4]
             [2 * GB_SIZE] = 169440.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpAllReduce][4]
             [4 * GB_SIZE] = 338744.0;

  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpAllReduce][8]
             [16 * MB_SIZE] = 3834.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpAllReduce][8]
             [32 * MB_SIZE] = 7620.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpAllReduce][8]
             [64 * MB_SIZE] = 15189.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpAllReduce][8]
             [128 * MB_SIZE] = 30239.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpAllReduce][8]
             [256 * MB_SIZE] = 60229.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpAllReduce][8]
             [512 * MB_SIZE] = 120152.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpAllReduce][8]
             [1 * GB_SIZE] = 238922.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpAllReduce][8]
             [2 * GB_SIZE] = 476359.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpAllReduce][8]
             [4 * GB_SIZE] = 954225.0;

  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpReduceScatter][2]
             [16 * MB_SIZE] = 516.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpReduceScatter][2]
             [32 * MB_SIZE] = 964.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpReduceScatter][2]
             [64 * MB_SIZE] = 1854.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpReduceScatter][2]
             [128 * MB_SIZE] = 3624.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpReduceScatter][2]
             [256 * MB_SIZE] = 7167.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpReduceScatter][2]
             [512 * MB_SIZE] = 14245.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpReduceScatter][2]
             [1 * GB_SIZE] = 28416.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpReduceScatter][2]
             [2 * GB_SIZE] = 56750.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpReduceScatter][2]
             [4 * GB_SIZE] = 113402.0;

  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpReduceScatter][4]
             [16 * MB_SIZE] = 744.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpReduceScatter][4]
             [32 * MB_SIZE] = 1409.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpReduceScatter][4]
             [64 * MB_SIZE] = 2732.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpReduceScatter][4]
             [128 * MB_SIZE] = 5376.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpReduceScatter][4]
             [256 * MB_SIZE] = 10666.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpReduceScatter][4]
             [512 * MB_SIZE] = 21246.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpReduceScatter][4]
             [1 * GB_SIZE] = 42415.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpReduceScatter][4]
             [2 * GB_SIZE] = 84752.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpReduceScatter][4]
             [4 * GB_SIZE] = 169423.0;

  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpReduceScatter][8]
             [16 * MB_SIZE] = 1920.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpReduceScatter][8]
             [32 * MB_SIZE] = 3805.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpReduceScatter][8]
             [64 * MB_SIZE] = 7589.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpReduceScatter][8]
             [128 * MB_SIZE] = 15152.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpReduceScatter][8]
             [256 * MB_SIZE] = 30080.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpReduceScatter][8]
             [512 * MB_SIZE] = 59751.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpReduceScatter][8]
             [1 * GB_SIZE] = 119258.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpReduceScatter][8]
             [2 * GB_SIZE] = 238926.0;
  homoTimeMap[FLAGCX_VENDOR_ILUVATAR_COREX][flagcxCommOpReduceScatter][8]
             [4 * GB_SIZE] = 476313.0;

  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpReduce][2][16 * MB_SIZE] = 476.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpReduce][2][32 * MB_SIZE] = 905.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpReduce][2][64 * MB_SIZE] =
      1785.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpReduce][2][128 * MB_SIZE] =
      2750.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpReduce][2][256 * MB_SIZE] =
      5458.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpReduce][2][512 * MB_SIZE] =
      10969.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpReduce][2][1 * GB_SIZE] =
      22133.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpReduce][2][2 * GB_SIZE] =
      44718.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpReduce][2][4 * GB_SIZE] =
      89973.0;

  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpReduce][4][16 * MB_SIZE] = 343.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpReduce][4][32 * MB_SIZE] = 693.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpReduce][4][64 * MB_SIZE] =
      1048.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpReduce][4][128 * MB_SIZE] =
      1624.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpReduce][4][256 * MB_SIZE] =
      2975.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpReduce][4][512 * MB_SIZE] =
      5683.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpReduce][4][1 * GB_SIZE] =
      11075.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpReduce][4][2 * GB_SIZE] =
      21683.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpReduce][4][4 * GB_SIZE] =
      42680.0;

  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpReduce][8][16 * MB_SIZE] = 504.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpReduce][8][32 * MB_SIZE] = 848.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpReduce][8][64 * MB_SIZE] = 996.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpReduce][8][128 * MB_SIZE] =
      1758.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpReduce][8][256 * MB_SIZE] =
      2588.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpReduce][8][512 * MB_SIZE] =
      4282.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpReduce][8][1 * GB_SIZE] = 7706.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpReduce][8][2 * GB_SIZE] =
      14384.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpReduce][8][4 * GB_SIZE] =
      27515.0;

  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpAllReduce][2][16 * MB_SIZE] =
      503.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpAllReduce][2][32 * MB_SIZE] =
      939.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpAllReduce][2][64 * MB_SIZE] =
      1837.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpAllReduce][2][128 * MB_SIZE] =
      2872.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpAllReduce][2][256 * MB_SIZE] =
      5692.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpAllReduce][2][512 * MB_SIZE] =
      11369.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpAllReduce][2][1 * GB_SIZE] =
      22684.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpAllReduce][2][2 * GB_SIZE] =
      45345.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpAllReduce][2][4 * GB_SIZE] =
      90608.0;

  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpAllReduce][4][16 * MB_SIZE] =
      233.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpAllReduce][4][32 * MB_SIZE] =
      441.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpAllReduce][4][64 * MB_SIZE] =
      790.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpAllReduce][4][128 * MB_SIZE] =
      1490.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpAllReduce][4][256 * MB_SIZE] =
      2926.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpAllReduce][4][512 * MB_SIZE] =
      5796.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpAllReduce][4][1 * GB_SIZE] =
      11510.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpAllReduce][4][2 * GB_SIZE] =
      22977.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpAllReduce][4][4 * GB_SIZE] =
      45930.0;

  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpAllReduce][8][16 * MB_SIZE] =
      170.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpAllReduce][8][32 * MB_SIZE] =
      281.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpAllReduce][8][64 * MB_SIZE] =
      493.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpAllReduce][8][128 * MB_SIZE] =
      975.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpAllReduce][8][256 * MB_SIZE] =
      1695.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpAllReduce][8][512 * MB_SIZE] =
      3307.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpAllReduce][8][1 * GB_SIZE] =
      6591.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpAllReduce][8][2 * GB_SIZE] =
      13045.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpAllReduce][8][4 * GB_SIZE] =
      25989.0;

  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpReduceScatter][2][16 * MB_SIZE] =
      279.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpReduceScatter][2][32 * MB_SIZE] =
      523.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpReduceScatter][2][64 * MB_SIZE] =
      980.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpReduceScatter][2]
             [128 * MB_SIZE] = 1581.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpReduceScatter][2]
             [256 * MB_SIZE] = 3057.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpReduceScatter][2]
             [512 * MB_SIZE] = 6109.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpReduceScatter][2][1 * GB_SIZE] =
      12174.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpReduceScatter][2][2 * GB_SIZE] =
      24294.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpReduceScatter][2][4 * GB_SIZE] =
      48504.0;

  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpReduceScatter][4][16 * MB_SIZE] =
      260.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpReduceScatter][4][32 * MB_SIZE] =
      350.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpReduceScatter][4][64 * MB_SIZE] =
      439.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpReduceScatter][4]
             [128 * MB_SIZE] = 773.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpReduceScatter][4]
             [256 * MB_SIZE] = 1493.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpReduceScatter][4]
             [512 * MB_SIZE] = 2969.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpReduceScatter][4][1 * GB_SIZE] =
      5838.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpReduceScatter][4][2 * GB_SIZE] =
      11650.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpReduceScatter][4][4 * GB_SIZE] =
      23253.0;

  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpReduceScatter][8][16 * MB_SIZE] =
      136.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpReduceScatter][8][32 * MB_SIZE] =
      217.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpReduceScatter][8][64 * MB_SIZE] =
      269.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpReduceScatter][8]
             [128 * MB_SIZE] = 593.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpReduceScatter][8]
             [256 * MB_SIZE] = 883.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpReduceScatter][8]
             [512 * MB_SIZE] = 1770.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpReduceScatter][8][1 * GB_SIZE] =
      3395.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpReduceScatter][8][2 * GB_SIZE] =
      6710.0;
  homoTimeMap[FLAGCX_VENDOR_METAX][flagcxCommOpReduceScatter][8][4 * GB_SIZE] =
      13325.0;
}