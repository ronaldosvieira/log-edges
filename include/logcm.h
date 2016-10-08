#ifndef _INCLUDE_LOGCM_
#define _INCLUDE_LOGCM_

typedef const char Filter5[5][5];

Filter5 average = {
    {1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1}
};

Filter5 lapOfGau = {
    {0, 0, 1, 0, 0},
    {0, 1, 2, 1, 0},
    {1, 2, -16, 2, 1},
    {0, 1, 2, 1, 0},
    {0, 0, 1, 0, 0}
};

#endif /* _INCLUDE_LOGCM_ */