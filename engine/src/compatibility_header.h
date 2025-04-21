#ifdef _MSC_VER 
#define WINDOWS
//not #if defined(_WIN32) || defined(_WIN64) because we have strncasecmp in mingw
#define strncasecmp _strnicmp
#define strcasecmp _stricmp
#define strdup _strdup
#define unlink _unlink
#endif