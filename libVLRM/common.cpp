#include <VLRM/common.h>

#if defined(VLRM_Host)

namespace vlrm {
    std::filesystem::path getExecutableDirectory() {
#if defined(VLRM_Platform_Windows_MSVC)
        static std::filesystem::path ret;
        static bool done = false;
        if (!done) {
            TCHAR filepath[1024];
            auto length = GetModuleFileName(NULL, filepath, 1024);
            VLRMAssert(length > 0, "Failed to get executable path.");

            ret = filepath;
            ret = ret.remove_filename();

            done = true;
        }
        return ret;
#else
        VLRMAssert_NotImplemented();
        return "";
#endif
    }
}

#if defined(VLRM_Platform_Windows_MSVC)
void vlrmDevPrintf(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
}
#endif

void vlrmprintf(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
}

#endif
