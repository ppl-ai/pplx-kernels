#include "all_to_all.h"

#include "core/cuda_utils.h"
#include "core/utils.h"

using namespace pplx;

AllToAll::AllToAll(
    size_t maxNumTokens,
    size_t numExperts,
    size_t expertsPerToken,
    unsigned rank,
    unsigned worldSize,
    unsigned dpSize,
    size_t hiddenDim,
    size_t hiddenDimBytes,
    size_t hiddenDimScaleBytes
)
    : maxNumTokens(maxNumTokens),
      numExperts(numExperts),
      numLocalExperts(ceil_div<uint32_t>(numExperts, worldSize)),
      numDPGroups(ceil_div<uint32_t>(worldSize, dpSize)),
      expertsPerToken(expertsPerToken),
      hiddenDim(hiddenDim),
      hiddenDimBytes(hiddenDimBytes),
      hiddenDimScaleBytes(hiddenDimScaleBytes),
      rank(rank),
      worldSize(worldSize),
      dpSize(dpSize),
      numSMs(get_sm_count()) {

  PPLX_ASSERT(hiddenDimBytes % 16 == 0, "invalid hidden dim bytes");
  PPLX_ASSERT(hiddenDimScaleBytes % 16 == 0, "invalid hidden dim scale bytes");
  const size_t perTokenBytes =
      round_up<size_t>(hiddenDimBytes + hiddenDimScaleBytes + sizeof(uint32_t), 16);

  PPLX_ASSERT(numLocalExperts != 0, "numLocalExperts is 0");
  PPLX_ASSERT(numDPGroups > 1, "at least 2 DP groups are required");
  PPLX_ASSERT(hiddenDimScaleBytes <= hiddenDimBytes, "invalid hidden dim bytes");
}

AllToAll::~AllToAll() {}
