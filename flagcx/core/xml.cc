/*************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "xml.h"
#include "core.h"
#include "flagcx_common.h"
#include <ctype.h>
#include <fcntl.h>
#include <float.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#if defined(__x86_64__)
#include <cpuid.h>
#endif

#define BUSID_SIZE (sizeof("0000:00:00.0"))
#define BUSID_REDUCED_SIZE (sizeof("0000:00"))
static void memcpylower(char *dst, const char *src, const size_t size) {
  for (int i = 0; i < size; i++)
    dst[i] = tolower(src[i]);
}
static flagcxResult_t getPciPath(const char *busId, char **path) {
  char busPath[] = "/sys/class/pci_bus/0000:00/../../0000:00:00.0";
  memcpylower(busPath + sizeof("/sys/class/pci_bus/") - 1, busId,
              BUSID_REDUCED_SIZE - 1);
  memcpylower(busPath + sizeof("/sys/class/pci_bus/0000:00/../../") - 1, busId,
              BUSID_SIZE - 1);
  *path = realpath(busPath, NULL);
  if (*path == NULL) {
    WARN("Could not find real path of %s", busPath);
    return flagcxSystemError;
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxTopoGetStrFromSys(const char *path, const char *fileName,
                                       char *strValue) {
  char filePath[PATH_MAX];
  sprintf(filePath, "%s/%s", path, fileName);
  int offset = 0;
  FILE *file;
  if ((file = fopen(filePath, "r")) != NULL) {
    while (feof(file) == 0 && ferror(file) == 0 && offset < MAX_STR_LEN) {
      int len = fread(strValue + offset, 1, MAX_STR_LEN - offset, file);
      offset += len;
    }
    fclose(file);
  }
  if (offset == 0) {
    strValue[0] = '\0';
    INFO(FLAGCX_GRAPH, "Topology detection : could not read %s, ignoring",
         filePath);
  } else {
    strValue[offset - 1] = '\0';
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxTopoSetAttrFromSys(struct flagcxXmlNode *pciNode,
                                        const char *path, const char *fileName,
                                        const char *attrName) {
  char strValue[MAX_STR_LEN];
  FLAGCXCHECK(flagcxTopoGetStrFromSys(path, fileName, strValue));
  if (strValue[0] != '\0') {
    FLAGCXCHECK(xmlSetAttr(pciNode, attrName, strValue));
  }
  TRACE(FLAGCX_GRAPH, "Read from sys %s/%s -> %s=%s", path, fileName, attrName,
        strValue);
  return flagcxSuccess;
}

flagcxResult_t flagcxTopoGetPciNode(struct flagcxXml *xml, const char *busId,
                                    struct flagcxXmlNode **pciNode) {
  FLAGCXCHECK(xmlFindTagKv(xml, "pci", pciNode, "busid", busId));
  if (*pciNode == NULL) {
    FLAGCXCHECK(xmlAddNode(xml, NULL, "pci", pciNode));
    FLAGCXCHECK(xmlSetAttr(*pciNode, "busid", busId));
  }

  return flagcxSuccess;
}

// Check whether a string is in BDF format or not.
// BDF (Bus-Device-Function) is "BBBB:BB:DD.F" where B, D and F are hex digits.
// There can be trailing chars.
int isHex(char c) {
  return ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') ||
          (c >= 'A' && c <= 'F'));
}
int checkBDFFormat(char *bdf) {
  if (bdf[4] != ':' || bdf[7] != ':' || bdf[10] != '.')
    return 0;
  if (isHex(bdf[0]) == 0 || isHex(bdf[1] == 0) || isHex(bdf[2] == 0) ||
      isHex(bdf[3] == 0) || isHex(bdf[5] == 0) || isHex(bdf[6] == 0) ||
      isHex(bdf[8] == 0) || isHex(bdf[9] == 0) || isHex(bdf[11] == 0))
    return 0;
  return 1;
}

// TODO: it would be better if we have a device handle and can call APIs to get
// Apu information using that device handle
flagcxResult_t flagcxTopoGetXmlFromApu(struct flagcxXmlNode *pciNode,
                                       struct flagcxXml *xml,
                                       struct flagcxXmlNode **apuNodeRet) {
  struct flagcxXmlNode *apuNode = NULL;
  FLAGCXCHECK(xmlGetSub(pciNode, "apu", &apuNode));
  if (apuNode == NULL) {
    FLAGCXCHECK(xmlAddNode(xml, pciNode, "apu", &apuNode));
  }
  // TODO: maybe add vendor information to the xml node in the future
  *apuNodeRet = apuNode;
  return flagcxSuccess;
}

flagcxResult_t flagcxTopoGetXmlFromCpu(struct flagcxXmlNode *cpuNode,
                                       struct flagcxXml *xml) {
  int index;
  FLAGCXCHECK(xmlGetAttrIndex(cpuNode, "affinity", &index));
  if (index == -1) {
    const char *numaId;
    FLAGCXCHECK(xmlGetAttr(cpuNode, "numaid", &numaId));
    if (numaId == NULL) {
      WARN("GetXmlFromCpu : could not find CPU numa ID.");
      return flagcxInternalError;
    }
    // Set affinity
    char cpumaskPath[] = "/sys/devices/system/node/node0000";
    sprintf(cpumaskPath, "/sys/devices/system/node/node%s", numaId);
    FLAGCXCHECK(
        flagcxTopoSetAttrFromSys(cpuNode, cpumaskPath, "cpumap", "affinity"));
  }

  FLAGCXCHECK(xmlGetAttrIndex(cpuNode, "arch", &index));
  if (index == -1) {
    // Fill CPU type / vendor / model
#if defined(__PPC__)
    FLAGCXCHECK(xmlSetAttr(cpuNode, "arch", "ppc64"));
#elif defined(__aarch64__)
    FLAGCXCHECK(xmlSetAttr(cpuNode, "arch", "arm64"));
#elif defined(__x86_64__)
    FLAGCXCHECK(xmlSetAttr(cpuNode, "arch", "x86_64"));
#endif
  }

#if defined(__x86_64__)
  FLAGCXCHECK(xmlGetAttrIndex(cpuNode, "vendor", &index));
  if (index == -1) {
    union {
      struct {
        // CPUID 0 String register order
        uint32_t ebx;
        uint32_t edx;
        uint32_t ecx;
      };
      char vendor[12];
    } cpuid0;

    unsigned unused;
    __cpuid(0, unused, cpuid0.ebx, cpuid0.ecx, cpuid0.edx);
    char vendor[13];
    strncpy(vendor, cpuid0.vendor, 12);
    vendor[12] = '\0';
    FLAGCXCHECK(xmlSetAttr(cpuNode, "vendor", vendor));
  }

  FLAGCXCHECK(xmlGetAttrIndex(cpuNode, "familyid", &index));
  if (index == -1) {
    union {
      struct {
        unsigned steppingId : 4;
        unsigned modelId : 4;
        unsigned familyId : 4;
        unsigned processorType : 2;
        unsigned resv0 : 2;
        unsigned extModelId : 4;
        unsigned extFamilyId : 8;
        unsigned resv1 : 4;
      };
      uint32_t val;
    } cpuid1;
    unsigned unused;
    __cpuid(1, cpuid1.val, unused, unused, unused);
    int familyId = cpuid1.familyId + (cpuid1.extFamilyId << 4);
    int modelId = cpuid1.modelId + (cpuid1.extModelId << 4);
    FLAGCXCHECK(xmlSetAttrInt(cpuNode, "familyid", familyId));
    FLAGCXCHECK(xmlSetAttrInt(cpuNode, "modelid", modelId));
  }
#endif
  return flagcxSuccess;
}

flagcxResult_t flagcxTopoGetXmlFromSys(struct flagcxXmlNode *pciNode,
                                       struct flagcxXml *xml) {
  const char *busId;
  FLAGCXCHECK(xmlGetAttr(pciNode, "busid", &busId));
  char *path = NULL;
  getPciPath(busId, &path);

  if (path) {
    FLAGCXCHECK(flagcxTopoSetAttrFromSys(pciNode, path, "class", "class"));
  }
  int index;
  FLAGCXCHECK(xmlGetAttrIndex(pciNode, "vendor", &index));
  if (index == -1) {
    if (path)
      flagcxTopoSetAttrFromSys(pciNode, path, "vendor", "vendor");
  }
  FLAGCXCHECK(xmlGetAttrIndex(pciNode, "device", &index));
  if (index == -1) {
    if (path)
      flagcxTopoSetAttrFromSys(pciNode, path, "device", "device");
  }
  FLAGCXCHECK(xmlGetAttrIndex(pciNode, "subsystem_vendor", &index));
  if (index == -1) {
    if (path)
      flagcxTopoSetAttrFromSys(pciNode, path, "subsystem_vendor",
                               "subsystem_vendor");
  }
  FLAGCXCHECK(xmlGetAttrIndex(pciNode, "subsystem_device", &index));
  if (index == -1) {
    if (path)
      flagcxTopoSetAttrFromSys(pciNode, path, "subsystem_device",
                               "subsystem_device");
  }
  // ignore link speed for now
  // ignore link width for now

  struct flagcxXmlNode *parent = pciNode->parent;
  if (parent == NULL) {
    // try to find the parent along the pci path
    if (path) {
      // Save that for later in case next step is a CPU
      char numaIdStr[MAX_STR_LEN];
      FLAGCXCHECK(flagcxTopoGetStrFromSys(path, "numa_node", numaIdStr));

      // Go up one level in the PCI tree. Rewind two "/" and follow the upper
      // PCI switch, or stop if we reach a CPU root complex.
      int slashCount = 0;
      int parentOffset;
      for (parentOffset = strlen(path) - 1; parentOffset > 0; parentOffset--) {
        if (path[parentOffset] == '/') {
          slashCount++;
          path[parentOffset] = '\0';
          int start = parentOffset - 1;
          while (start > 0 && path[start] != '/')
            start--;
          // Check whether the parent path looks like "BBBB:BB:DD.F" or not.
          if (checkBDFFormat(path + start + 1) == 0) {
            // This a CPU root complex. Create a CPU tag and stop there.
            struct flagcxXmlNode *topNode;
            FLAGCXCHECK(xmlFindTag(xml, "system", &topNode));
            FLAGCXCHECK(
                xmlGetSubKv(topNode, "cpu", &parent, "numaid", numaIdStr));
            if (parent == NULL) {
              FLAGCXCHECK(xmlAddNode(xml, topNode, "cpu", &parent));
              FLAGCXCHECK(xmlSetAttrLong(parent, "host_hash", getHostHash()));
              FLAGCXCHECK(xmlSetAttr(parent, "numaid", numaIdStr));
            }
          } else if (slashCount == 2) {
            // Continue on the upper PCI switch
            for (int i = strlen(path) - 1; i > 0; i--) {
              if (path[i] == '/') {
                FLAGCXCHECK(
                    xmlFindTagKv(xml, "pci", &parent, "busid", path + i + 1));
                if (parent == NULL) {
                  FLAGCXCHECK(xmlAddNode(xml, NULL, "pci", &parent));
                  FLAGCXCHECK(xmlSetAttr(parent, "busid", path + i + 1));
                }
                break;
              }
            }
          }
        }
        if (parent)
          break;
      }
    } else {
      // No information on /sys, attach GPU to unknown CPU
      FLAGCXCHECK(xmlFindTagKv(xml, "cpu", &parent, "numaid", "-1"));
      if (parent == NULL) {
        struct flagcxXmlNode *topNode;
        FLAGCXCHECK(xmlFindTag(xml, "system", &topNode));
        FLAGCXCHECK(xmlAddNode(xml, topNode, "cpu", &parent));
        FLAGCXCHECK(xmlSetAttrLong(parent, "host_hash", getHostHash()));
        FLAGCXCHECK(xmlSetAttr(parent, "numaid", "-1"));
        FLAGCXCHECK(flagcxTopoGetXmlFromCpu(parent, xml));
      }
    }
    pciNode->parent = parent;
    // Keep PCI sub devices ordered by PCI Bus ID (Issue #820)
    int subIndex = parent->nSubs;
    const char *newBusId;
    FLAGCXCHECK(xmlGetAttrStr(pciNode, "busid", &newBusId));
    for (int s = 0; s < parent->nSubs; s++) {
      const char *busId;
      FLAGCXCHECK(xmlGetAttr(parent->subs[s], "busid", &busId));
      if (busId != NULL && strcmp(newBusId, busId) < 0) {
        subIndex = s;
        break;
      }
    }
    if (parent->nSubs == MAX_SUBS) {
      WARN("Error : XML parser is limited to %d subnodes", MAX_SUBS);
      return flagcxInternalError;
    }
    for (int s = parent->nSubs; s > subIndex; s--)
      parent->subs[s] = parent->subs[s - 1];
    parent->subs[subIndex] = pciNode;
    parent->nSubs++;
  }
  if (strcmp(parent->name, "pci") == 0) {
    FLAGCXCHECK(flagcxTopoGetXmlFromSys(parent, xml));
  } else if (strcmp(parent->name, "cpu") == 0) {
    FLAGCXCHECK(flagcxTopoGetXmlFromCpu(parent, xml));
  }
  free(path);
  return flagcxSuccess;
}

flagcxResult_t flagcxTopoFillApu(struct flagcxXml *xml, const char *busId,
                                 struct flagcxXmlNode **gpuNode) {
  struct flagcxXmlNode *pciNode;
  FLAGCXCHECK(flagcxTopoGetPciNode(xml, busId, &pciNode));
  FLAGCXCHECK(flagcxTopoGetXmlFromSys(pciNode, xml));

  FLAGCXCHECK(flagcxTopoGetXmlFromApu(pciNode, xml, gpuNode));
  return flagcxSuccess;
}

// Returns the subsystem name of a path, i.e. the end of the path
// where sysPath/subsystem points to.
flagcxResult_t flagcxTopoGetSubsystem(const char *sysPath, char *subSys) {
  char subSysPath[PATH_MAX];
  sprintf(subSysPath, "%s/subsystem", sysPath);
  char *path = realpath(subSysPath, NULL);
  if (path == NULL) {
    subSys[0] = '\0';
  } else {
    int offset;
    for (offset = strlen(path); offset > 0 && path[offset] != '/'; offset--)
      ;
    strcpy(subSys, path + offset + 1);
    free(path);
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxTopoFillNet(struct flagcxXml *xml, const char *pciPath,
                                 const char *netName,
                                 struct flagcxXmlNode **netNode) {
  FLAGCXCHECK(xmlFindTagKv(xml, "net", netNode, "name", netName));
  if (*netNode != NULL)
    return flagcxSuccess;

  const char *pciSysPath = pciPath;
  if (pciSysPath) {
    char subSystem[PATH_MAX];
    FLAGCXCHECK(flagcxTopoGetSubsystem(pciSysPath, subSystem));
    if (strcmp(subSystem, "pci") != 0) {
      INFO(FLAGCX_GRAPH,
           "Topology detection: network path %s is not a PCI device (%s). "
           "Attaching to first CPU",
           pciSysPath, subSystem);
      pciSysPath = NULL;
    }
  }

  struct flagcxXmlNode *parent = NULL;
  if (pciSysPath) {
    int offset;
    for (offset = strlen(pciSysPath) - 1; pciSysPath[offset] != '/'; offset--)
      ;
    char busId[FLAGCX_DEVICE_PCI_BUSID_BUFFER_SIZE];
    strcpy(busId, pciSysPath + offset + 1);
    FLAGCXCHECK(flagcxTopoGetPciNode(xml, busId, &parent));
    FLAGCXCHECK(flagcxTopoGetXmlFromSys(parent, xml));
  } else {
    // Virtual NIC, no PCI device, attach to first CPU
    FLAGCXCHECK(xmlFindTag(xml, "cpu", &parent));
  }

  struct flagcxXmlNode *nicNode = NULL;
  FLAGCXCHECK(xmlGetSub(parent, "nic", &nicNode));
  if (nicNode == NULL) {
    FLAGCXCHECK(xmlAddNode(xml, parent, "nic", &nicNode));
  }

  // We know that this net does not exist yet (we searched for it at the
  // beginning of this function), so we can add it.
  FLAGCXCHECK(xmlAddNode(xml, nicNode, "net", netNode));
  FLAGCXCHECK(xmlSetAttr(*netNode, "name", netName));
  return flagcxSuccess;
}