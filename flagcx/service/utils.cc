/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "utils.h"
#include "adaptor.h"
#include "core.h"
#include "flagcx_common.h"
#include <stdlib.h>

flagcxResult_t int64ToBusId(int64_t id, char *busId) {
  sprintf(busId, "%04lx:%02lx:%02lx.%01lx", (id) >> 20, (id & 0xff000) >> 12,
          (id & 0xff0) >> 4, (id & 0xf));
  return flagcxSuccess;
}

flagcxResult_t busIdToInt64(const char *busId, int64_t *id) {
  char hexStr[17]; // Longest possible int64 hex string + null terminator.
  int hexOffset = 0;
  for (int i = 0; hexOffset < sizeof(hexStr) - 1; i++) {
    char c = busId[i];
    if (c == '.' || c == ':')
      continue;
    if ((c >= '0' && c <= '9') || (c >= 'A' && c <= 'F') ||
        (c >= 'a' && c <= 'f')) {
      hexStr[hexOffset++] = busId[i];
    } else
      break;
  }
  hexStr[hexOffset] = '\0';
  *id = strtol(hexStr, NULL, 16);
  return flagcxSuccess;
}

// Convert a logical cudaDev index to the NVML device minor number
flagcxResult_t getBusId(int cudaDev, int64_t *busId) {
  // On most systems, the PCI bus ID comes back as in the 0000:00:00.0
  // format. Still need to allocate proper space in case PCI domain goes
  // higher.
  char busIdStr[] = "00000000:00:00.0";
  /**
   * TODO: how to get bus id
   **/
  FLAGCXCHECK(
      deviceAdaptor->getDevicePciBusId(busIdStr, sizeof(busIdStr), cudaDev));
  TRACE(FLAGCX_INIT, "busId for cudaDev %d is %s", cudaDev, busIdStr);
  FLAGCXCHECK(busIdToInt64(busIdStr, busId));
  return flagcxSuccess;
}

flagcxResult_t getHostName(char *hostname, int maxlen, const char delim) {
  if (gethostname(hostname, maxlen) != 0) {
    strncpy(hostname, "unknown", maxlen);
    return flagcxSystemError;
  }
  int i = 0;
  while ((hostname[i] != delim) && (hostname[i] != '\0') && (i < maxlen - 1))
    i++;
  hostname[i] = '\0';
  return flagcxSuccess;
}

uint64_t getHash(const char *string, int n) {
  // Based on DJB2a, result = result * 33 ^ char
  uint64_t result = 5381;
  for (int c = 0; c < n; c++) {
    result = ((result << 5) + result) ^ string[c];
  }
  return result;
}

/* Generate a hash of the unique identifying string for this host
 * that will be unique for both bare-metal and container instances
 * Equivalent of a hash of;
 *
 * $(hostname)$(cat /proc/sys/kernel/random/boot_id)
 *
 * This string can be overridden by using the FLAGCX_HOSTID env var.
 */
#define HOSTID_FILE "/proc/sys/kernel/random/boot_id"
uint64_t getHostHash(void) {
  char hostHash[1024];
  const char *hostId;

  // Fall back is the full hostname if something fails
  (void)getHostName(hostHash, sizeof(hostHash), '\0');
  int offset = strlen(hostHash);

  if ((hostId = flagcxGetEnv("FLAGCX_HOSTID")) != NULL) {
    INFO(FLAGCX_ENV, "FLAGCX_HOSTID set by environment to %s", hostId);
    strncpy(hostHash, hostId, sizeof(hostHash));
  } else {
    FILE *file = fopen(HOSTID_FILE, "r");
    if (file != NULL) {
      char *p;
      if (fscanf(file, "%ms", &p) == 1) {
        strncpy(hostHash + offset, p, sizeof(hostHash) - offset - 1);
        free(p);
      }
    }
    fclose(file);
  }

  // Make sure the string is terminated
  hostHash[sizeof(hostHash) - 1] = '\0';

  TRACE(FLAGCX_INIT, "unique hostname '%s'", hostHash);

  return getHash(hostHash, strlen(hostHash));
}

/* Generate a hash of the unique identifying string for this process
 * that will be unique for both bare-metal and container instances
 * Equivalent of a hash of;
 *
 * $$ $(readlink /proc/self/ns/pid)
 */
uint64_t getPidHash(void) {
  char pname[1024];
  // Start off with our pid ($$)
  sprintf(pname, "%ld", (long)getpid());
  int plen = strlen(pname);
  int len =
      readlink("/proc/self/ns/pid", pname + plen, sizeof(pname) - 1 - plen);
  if (len < 0)
    len = 0;

  pname[plen + len] = '\0';
  TRACE(FLAGCX_INIT, "unique PID '%s'", pname);

  return getHash(pname, strlen(pname));
}

int parseStringList(const char *string, struct netIf *ifList, int maxList) {
  if (!string)
    return 0;

  const char *ptr = string;

  int ifNum = 0;
  int ifC = 0;
  char c;
  do {
    c = *ptr;
    if (c == ':') {
      if (ifC > 0) {
        ifList[ifNum].prefix[ifC] = '\0';
        ifList[ifNum].port = atoi(ptr + 1);
        ifNum++;
        ifC = 0;
      }
      while (c != ',' && c != '\0')
        c = *(++ptr);
    } else if (c == ',' || c == '\0') {
      if (ifC > 0) {
        ifList[ifNum].prefix[ifC] = '\0';
        ifList[ifNum].port = -1;
        ifNum++;
        ifC = 0;
      }
    } else {
      ifList[ifNum].prefix[ifC] = c;
      ifC++;
    }
    ptr++;
  } while (ifNum < maxList && c);
  return ifNum;
}

static bool matchIf(const char *string, const char *ref, bool matchExact) {
  // Make sure to include '\0' in the exact case
  int matchLen = matchExact ? strlen(string) + 1 : strlen(ref);
  return strncmp(string, ref, matchLen) == 0;
}

static bool matchPort(const int port1, const int port2) {
  if (port1 == -1)
    return true;
  if (port2 == -1)
    return true;
  if (port1 == port2)
    return true;
  return false;
}

bool matchIfList(const char *string, int port, struct netIf *ifList,
                 int listSize, bool matchExact) {
  // Make an exception for the case where no user list is defined
  if (listSize == 0)
    return true;

  for (int i = 0; i < listSize; i++) {
    if (matchIf(string, ifList[i].prefix, matchExact) &&
        matchPort(port, ifList[i].port)) {
      return true;
    }
  }
  return false;
}

__thread struct flagcxThreadSignal flagcxThreadSignalLocalInstance =
    flagcxThreadSignalStaticInitializer();

void *flagcxMemoryStack::allocateSpilled(struct flagcxMemoryStack *me,
                                         size_t size, size_t align) {
  // `me->hunks` points to the top of the stack non-empty hunks. Hunks above
  // this (reachable via `->above`) are empty.
  struct Hunk *top = me->topFrame.hunk;
  size_t mallocSize = 0;

  // If we have lots of space left in hunk but that wasn't enough then we'll
  // allocate the object unhunked.
  if (me->topFrame.end - me->topFrame.bumper >= 8 << 10)
    goto unhunked;

  // If we have another hunk (which must be empty) waiting above this one and
  // the object fits then use that.
  if (top && top->above) {
    struct Hunk *top1 = top->above;
    uintptr_t uobj =
        (reinterpret_cast<uintptr_t>(top1) + sizeof(struct Hunk) + align - 1) &
        -uintptr_t(align);
    if (uobj + size <= reinterpret_cast<uintptr_t>(top1) + top1->size) {
      me->topFrame.hunk = top1;
      me->topFrame.bumper = uobj + size;
      me->topFrame.end = reinterpret_cast<uintptr_t>(top1) + top1->size;
      return reinterpret_cast<void *>(uobj);
    }
  }

  { // If the next hunk we're going to allocate wouldn't be big enough but the
    // Unhunk proxy fits in the current hunk then go allocate as unhunked.
    size_t nextSize = (top ? top->size : 0) + (64 << 10);
    constexpr size_t maxAlign = 64;
    if (nextSize < sizeof(struct Hunk) + maxAlign + size) {
      uintptr_t uproxy = (me->topFrame.bumper + alignof(Unhunk) - 1) &
                         -uintptr_t(alignof(Unhunk));
      if (uproxy + sizeof(struct Unhunk) <= me->topFrame.end)
        goto unhunked;
    }

    // At this point we must need another hunk, either to fit the object
    // itself or its Unhunk proxy.
    mallocSize = nextSize;
    INFO(FLAGCX_ALLOC, "%s:%d memory stack hunk malloc(%llu)", __FILE__,
         __LINE__, (unsigned long long)mallocSize);
    struct Hunk *top1 = (struct Hunk *)malloc(mallocSize);
    if (top1 == nullptr)
      goto malloc_exhausted;
    top1->size = nextSize;
    top1->above = nullptr;
    if (top)
      top->above = top1;
    top = top1;
    me->topFrame.hunk = top;
    me->topFrame.end = reinterpret_cast<uintptr_t>(top) + nextSize;
    me->topFrame.bumper =
        reinterpret_cast<uintptr_t>(top) + sizeof(struct Hunk);
  }

  { // Try to fit object in the new top hunk.
    uintptr_t uobj = (me->topFrame.bumper + align - 1) & -uintptr_t(align);
    if (uobj + size <= me->topFrame.end) {
      me->topFrame.bumper = uobj + size;
      return reinterpret_cast<void *>(uobj);
    }
  }

unhunked : { // We need to allocate the object out-of-band and put an Unhunk
             // proxy in-band
  // to keep track of it.
  uintptr_t uproxy =
      (me->topFrame.bumper + alignof(Unhunk) - 1) & -uintptr_t(alignof(Unhunk));
  Unhunk *proxy = reinterpret_cast<Unhunk *>(uproxy);
  me->topFrame.bumper = uproxy + sizeof(Unhunk);
  proxy->next = me->topFrame.unhunks;
  me->topFrame.unhunks = proxy;
  mallocSize = size;
  proxy->obj = malloc(mallocSize);
  INFO(FLAGCX_ALLOC, "%s:%d memory stack non-hunk malloc(%llu)", __FILE__,
       __LINE__, (unsigned long long)mallocSize);
  if (proxy->obj == nullptr)
    goto malloc_exhausted;
  return proxy->obj;
}

malloc_exhausted:
  WARN("%s:%d Unrecoverable error detected: malloc(size=%llu) returned null.",
       __FILE__, __LINE__, (unsigned long long)mallocSize);
  abort();
}

void flagcxMemoryStackDestruct(struct flagcxMemoryStack *me) {
  // Free unhunks first because both the frames and unhunk proxies lie within
  // the hunks.
  struct flagcxMemoryStack::Frame *f = &me->topFrame;
  while (f != nullptr) {
    struct flagcxMemoryStack::Unhunk *u = f->unhunks;
    while (u != nullptr) {
      free(u->obj);
      u = u->next;
    }
    f = f->below;
  }
  // Free hunks
  struct flagcxMemoryStack::Hunk *h = me->stub.above;
  while (h != nullptr) {
    struct flagcxMemoryStack::Hunk *h1 = h->above;
    free(h);
    h = h1;
  }
}

const char *flagcxOpToString(flagcxRedOp_t op) {
  switch (op) {
    case flagcxSum:
      return "flagcxSum";
    case flagcxProd:
      return "flagcxProd";
    case flagcxMax:
      return "flagcxMax";
    case flagcxMin:
      return "flagcxMin";
    case flagcxAvg:
      return "flagcxAvg";
    default:
      return "Unknown";
  }
}

const char *flagcxDatatypeToString(flagcxDataType_t type) {
  switch (type) {
    case flagcxInt8: // flagcxChar
      return "flagcxInt8";
    case flagcxInt32: // flagcxInt
      return "flagcxInt32";
    case flagcxUint32:
      return "flagcxUint32";
    case flagcxInt64:
      return "flagcxInt64";
    case flagcxUint64:
      return "flagcxUint64";
    case flagcxFloat16: // flagcxHalf
      return "flagcxFloat16";
    case flagcxFloat32: // flagcxFloat
      return "flagcxFloat32";
    case flagcxFloat64: // flagcxDouble
      return "flagcxFloat64";
#if defined(__CUDA_BF16_TYPES_EXIST__)
    case flagcxBfloat16:
      return "flagcxBfloat16";
#endif
    default:
      return "Unknown";
  }
}

const char *flagcxAlgoToString(int algo) {
  switch (algo) {
    case FLAGCX_ALGO_TREE:
      return "TREE";
    case FLAGCX_ALGO_RING:
      return "RING";
    case FLAGCX_ALGO_COLLNET_DIRECT:
      return "COLLNET_DIRECT";
    case FLAGCX_ALGO_COLLNET_CHAIN:
      return "COLLNET_CHAIN";
    case FLAGCX_ALGO_NVLS:
      return "NVLS";
    case FLAGCX_ALGO_NVLS_TREE:
      return "NVLS_TREE";
    default:
      return "Unknown";
  }
}

const char *flagcxProtoToString(int proto) {
  switch (proto) {
    case FLAGCX_PROTO_LL:
      return "LL";
    case FLAGCX_PROTO_LL128:
      return "LL128";
    case FLAGCX_PROTO_SIMPLE:
      return "SIMPLE";
    default:
      return "Unknown";
  }
}
