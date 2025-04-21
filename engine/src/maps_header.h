
#ifndef NETWORKSTATE_STD_MAP

#define USE_UNORDERED_MAP
#include <unordered_map>
#define STATE_MAP std::unordered_map
#define HASH_STRUCT hash

#else
#define STATE_MAP std::map
#endif

#define MAP std::map
