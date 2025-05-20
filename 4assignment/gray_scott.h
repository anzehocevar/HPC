#ifndef GRAY_SCOTT_H
#define GRAY_SCOTT_H

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct gs_config_t
    {
        int n;     // grid size
        int steps; // number of time steps
        float dt;  // time step size
        float du;  // diffusion rate for u
        float dv;  // diffusion rate for v
        float f;   // feed rate
        float k;   // kill rate

    } gs_config;

    double gray_scott2D(gs_config config);

#ifdef __cplusplus
}
#endif

#endif
