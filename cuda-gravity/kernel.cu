
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "body_array.cuh";
#include "gravity_box.cuh";

int main()
{
    float x[] = {0, 1, 2};
    float y[] = { 0, 1, 2 };
    float m[] = { 0, 1, 2 };
    BodyArray bodies(3, m, x, y);
    return 0;
}

