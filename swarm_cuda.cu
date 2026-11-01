#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <cuda_runtime.h>

#ifndef N
#define N      1000      // number of agents (change with -DN=... when compiling)
#endif
#ifndef STEPS
#define STEPS  1000      // time steps to simulate
#endif
#define DENSITY 0.1f     // agents per unit area (world grows with N)
#define DT     0.1f      // time step size
#define VMAX   2.0f      // max speed
#define PI     3.14159265f
#define R      5.0f      // neighbour radius
#define RSEP       1.0f   // separation radius
#ifndef SAVE_EVERY
#define SAVE_EVERY 5      // save positions every 5 steps (0 = don't save)
#endif
#define PREY      0
#define PREDATOR  1
#ifndef PRED_PCT
#define PRED_PCT  2       // percent of agents that are predators
#endif
#define NPRED     (N * PRED_PCT / 100)
#define NPRED_MAX (NPRED > 0 ? NPRED : 1)
#define VPRED     2.6f    // predators are a bit faster than prey
#define RFLEE     8.0f    // prey notice predators within this distance
#define RHUNT     15.0f   // predators notice prey within this distance
#define RCATCH    1.0f    // a predator this close to its prey catches it
#define FLEE      1.0f    // how hard prey turn away from a predator
#define CHASE     0.5f    // how hard predators turn toward their prey
#define MAXCAND    128    // max prey a predator considers when planning
#define NLEAD      16     // aiming angles a predator tries for each prey
#define PLAN_STEPS 200    // how far ahead (in steps) a predator simulates a chase
#define GRID       64     // density grid is GRID x GRID cells
#define BLUR_R     2      // blur mask radius: mask has 2*BLUR_R+1 weights
#define CONFUSION  20.0f  // penalty (in steps) per unit of crowding around a prey

float W, H;              // world size, set in main from N and DENSITY
float align_w = 0.1f;    // how strongly agents turn toward neighbours
float cohesion_w = 0.02f;   // strength of the pull toward the group's centre
float separation_w = 0.5f;  // strength of the push away from close neighbours
float x[N], y[N];        // positions
float vx[N], vy[N];      // velocities
float nvx[N], nvy[N];    // new velocities (computed each step)
float avg_neighbours;    // average neighbours within R
float avg_too_close;     // average neighbours within RSEP
int   type[N];           // PREY or PREDATOR
int   target[N];         // predator: index of the prey it is chasing (-1 = none)
int   num_fleeing;       // prey that currently see a predator
int   total_catches = 0;
double thread_busy[64];  // seconds each thread spent working in flock_agents

float nb_vx[N], nb_vy[N];
float nb_dx[N], nb_dy[N];
float nb_sepx[N], nb_sepy[N];
int   nb_count[N];
int   nb_close[N];
float nb_best_d2[N];
float nb_best_dx[N], nb_best_dy[N];
int   nb_best_j[N];
int   nb_ncand[NPRED_MAX];
int   nb_cand[NPRED_MAX * MAXCAND];
float nb_cdx[NPRED_MAX * MAXCAND], nb_cdy[NPRED_MAX * MAXCAND];
float dens_raw[GRID * GRID];     
float dens_tmp[GRID * GRID];     
float dens[GRID * GRID];       
float mask[2 * BLUR_R + 1];     

float rand_range(float lo, float hi)
{
    return lo + (hi - lo) * ((float)rand() / RAND_MAX);
}

float wrap(float v, float size)
{
    if (v >= size) v -= size;
    if (v < 0)     v += size;
    return v;
}

float max_speed(int i)
{
    return (type[i] == PREDATOR) ? VPRED : VMAX;
}

void init_agents(void)
{
    for (int i = 0; i < N; i++) {
        type[i] = (i < NPRED) ? PREDATOR : PREY;
        x[i] = rand_range(0, W);
        y[i] = rand_range(0, H);
        float angle = rand_range(0, 2 * PI);
        vx[i] = max_speed(i) * cosf(angle);
        vy[i] = max_speed(i) * sinf(angle);
    }
}

void move_agents(void)
{
    for (int i = 0; i < N; i++) {
        x[i] = wrap(x[i] + vx[i] * DT, W);
        y[i] = wrap(y[i] + vy[i] * DT, H);
    }
}

float wrap_delta(float d, float size)
{
    if (d >  size / 2) d -= size;
    if (d < -size / 2) d += size;
    return d;
}

void init_mask(void)
{
    float sum = 0;
    for (int k = -BLUR_R; k <= BLUR_R; k++) {
        mask[k + BLUR_R] = expf(-(k * k) / 2.0f);
        sum += mask[k + BLUR_R];
    }
    for (int k = 0; k < 2 * BLUR_R + 1; k++)
        mask[k] /= sum;
}

int cell_of(float pos, float size)
{
    int c = (int)(pos / size * GRID);
    if (c < 0) c = 0;
    if (c >= GRID) c = GRID - 1;
    return c;
}

float density_at(float px, float py)
{
    px = wrap(px, W);
    py = wrap(py, H);
    return dens[cell_of(py, H) * GRID + cell_of(px, W)];
}

int plan_hunt(int ncand, const int cand[], const float cdx[], const float cdy[],
              float *hx, float *hy)
{
    int best_prey = -1;
    float best_score = PLAN_STEPS + 1;

    for (int c = 0; c < ncand; c++) {
        int j = cand[c];
        float crowd = CONFUSION * density_at(x[j], y[j]);
        float base = atan2f(cdy[c], cdx[c]);           
        for (int k = 0; k < NLEAD; k++) {
            float angle = base + (k - NLEAD / 2) * 0.15f; 
            float ux = cosf(angle), uy = sinf(angle);
            float px = 0, py = 0;                        
            float qx = cdx[c], qy = cdy[c];              
            for (int t = 1; t <= PLAN_STEPS; t++) {      
                px += VPRED * ux * DT;
                py += VPRED * uy * DT;
                qx += vx[j] * DT;
                qy += vy[j] * DT;
                float ex = qx - px, ey = qy - py;
                if (ex * ex + ey * ey < RCATCH * RCATCH) { 
                    if (t + crowd < best_score) {
                        best_score = t + crowd;
                        best_prey = j;
                        *hx = ux;
                        *hy = uy;
                    }
                    break;
                }
            }
        }
    }
    return best_prey;
}

__device__ float wrap_delta_gpu(float d, float size)
{
    if (d >  size / 2) d -= size;
    if (d < -size / 2) d += size;
    return d;
}

__global__ void neighbour_kernel(const float *x, const float *y, const float *vx, const float *vy,
                                 const int *type, float W, float H,
                                 float *nb_vx, float *nb_vy, float *nb_dx, float *nb_dy,
                                 float *nb_sepx, float *nb_sepy, int *nb_count, int *nb_close,
                                 float *nb_best_d2, float *nb_best_dx, float *nb_best_dy, int *nb_best_j,
                                 int *nb_ncand, int *nb_cand, float *nb_cdx, float *nb_cdy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float sum_vx = 0, sum_vy = 0;
    float sum_dx = 0, sum_dy = 0;
    float sep_x = 0, sep_y = 0;
    int count = 0, too_close = 0;

    float best_d2 = (type[i] == PREY) ? RFLEE * RFLEE : RHUNT * RHUNT;
    float best_dx = 0, best_dy = 0;
    int   best_j  = -1;
    int   ncand   = 0;

    for (int j = 0; j < N; j++) {
        if (j == i) continue;
        float dx = wrap_delta_gpu(x[j] - x[i], W);
        float dy = wrap_delta_gpu(y[j] - y[i], H);
        float d2 = dx * dx + dy * dy;

        if (type[i] != type[j]) {
            if (d2 < best_d2) {
                best_d2 = d2; best_dx = dx; best_dy = dy; best_j = j;
            }
            if (type[i] == PREDATOR && d2 < RHUNT * RHUNT && ncand < MAXCAND) {
                nb_cand[i * MAXCAND + ncand] = j;
                nb_cdx[i * MAXCAND + ncand]  = dx;
                nb_cdy[i * MAXCAND + ncand]  = dy;
                ncand++;
            }
        } else if (type[i] == PREY && d2 < R * R) {
            sum_vx += vx[j];
            sum_vy += vy[j];
            sum_dx += dx;
            sum_dy += dy;
            count++;
            if (d2 < RSEP * RSEP && d2 > 0) {
                sep_x -= dx / d2;
                sep_y -= dy / d2;
                too_close++;
            }
        }
    }

    nb_vx[i] = sum_vx;    nb_vy[i] = sum_vy;
    nb_dx[i] = sum_dx;    nb_dy[i] = sum_dy;
    nb_sepx[i] = sep_x;   nb_sepy[i] = sep_y;
    nb_count[i] = count;  nb_close[i] = too_close;
    nb_best_d2[i] = best_d2;
    nb_best_dx[i] = best_dx;
    nb_best_dy[i] = best_dy;
    nb_best_j[i]  = best_j;
    if (type[i] == PREDATOR) nb_ncand[i] = ncand;
}

float *d_x, *d_y, *d_vx, *d_vy;
int   *d_type;
float *d_nb_vx, *d_nb_vy, *d_nb_dx, *d_nb_dy, *d_nb_sepx, *d_nb_sepy;
int   *d_nb_count, *d_nb_close;
float *d_nb_best_d2, *d_nb_best_dx, *d_nb_best_dy;
int   *d_nb_best_j;
int   *d_nb_ncand, *d_nb_cand;
float *d_nb_cdx, *d_nb_cdy;

cudaEvent_t ev_start, ev_stop;
float kernel_ms = 0;
float *d_dens_raw, *d_dens_tmp, *d_dens, *d_mask;
float density_ms = 0;

void check(cudaError_t err, const char *what)
{
    if (err != cudaSuccess) {
        printf("CUDA error in %s: %s\n", what, cudaGetErrorString(err));
        exit(1);
    }
}

void gpu_init(void)
{
    cudaDeviceProp prop;
    check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    printf("GPU: %s, %d multiprocessors\n", prop.name, prop.multiProcessorCount);

    int fN = N * sizeof(float), iN = N * sizeof(int);
    int fC = NPRED_MAX * MAXCAND * sizeof(float), iC = NPRED_MAX * MAXCAND * sizeof(int);

    check(cudaMalloc((void **)&d_x, fN), "cudaMalloc");
    check(cudaMalloc((void **)&d_y, fN), "cudaMalloc");
    check(cudaMalloc((void **)&d_vx, fN), "cudaMalloc");
    check(cudaMalloc((void **)&d_vy, fN), "cudaMalloc");
    check(cudaMalloc((void **)&d_type, iN), "cudaMalloc");
    check(cudaMalloc((void **)&d_nb_vx, fN), "cudaMalloc");
    check(cudaMalloc((void **)&d_nb_vy, fN), "cudaMalloc");
    check(cudaMalloc((void **)&d_nb_dx, fN), "cudaMalloc");
    check(cudaMalloc((void **)&d_nb_dy, fN), "cudaMalloc");
    check(cudaMalloc((void **)&d_nb_sepx, fN), "cudaMalloc");
    check(cudaMalloc((void **)&d_nb_sepy, fN), "cudaMalloc");
    check(cudaMalloc((void **)&d_nb_count, iN), "cudaMalloc");
    check(cudaMalloc((void **)&d_nb_close, iN), "cudaMalloc");
    check(cudaMalloc((void **)&d_nb_best_d2, fN), "cudaMalloc");
    check(cudaMalloc((void **)&d_nb_best_dx, fN), "cudaMalloc");
    check(cudaMalloc((void **)&d_nb_best_dy, fN), "cudaMalloc");
    check(cudaMalloc((void **)&d_nb_best_j, iN), "cudaMalloc");
    check(cudaMalloc((void **)&d_nb_ncand, NPRED_MAX * sizeof(int)), "cudaMalloc");
    check(cudaMalloc((void **)&d_nb_cand, iC), "cudaMalloc");
    check(cudaMalloc((void **)&d_nb_cdx, fC), "cudaMalloc");
    check(cudaMalloc((void **)&d_nb_cdy, fC), "cudaMalloc");

    check(cudaMemcpy(d_type, type, iN, cudaMemcpyHostToDevice), "copy type");
    int fG = GRID * GRID * sizeof(float);
    check(cudaMalloc((void **)&d_dens_raw, fG), "cudaMalloc");
    check(cudaMalloc((void **)&d_dens_tmp, fG), "cudaMalloc");
    check(cudaMalloc((void **)&d_dens, fG), "cudaMalloc");
    check(cudaMalloc((void **)&d_mask, (2 * BLUR_R + 1) * sizeof(float)), "cudaMalloc");
    check(cudaMemcpy(d_mask, mask, (2 * BLUR_R + 1) * sizeof(float), cudaMemcpyHostToDevice), "copy mask");

    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);
}

void gpu_free(void)
{
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_type);
    cudaFree(d_nb_vx); cudaFree(d_nb_vy); cudaFree(d_nb_dx); cudaFree(d_nb_dy);
    cudaFree(d_nb_sepx); cudaFree(d_nb_sepy); cudaFree(d_nb_count); cudaFree(d_nb_close);
    cudaFree(d_nb_best_d2); cudaFree(d_nb_best_dx); cudaFree(d_nb_best_dy); cudaFree(d_nb_best_j);
    cudaFree(d_nb_ncand); cudaFree(d_nb_cand); cudaFree(d_nb_cdx); cudaFree(d_nb_cdy);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    cudaFree(d_dens_raw); cudaFree(d_dens_tmp); cudaFree(d_dens); cudaFree(d_mask);
}

void neighbour_search(void)
{
    int fN = N * sizeof(float), iN = N * sizeof(int);
    int fC = NPRED_MAX * MAXCAND * sizeof(float), iC = NPRED_MAX * MAXCAND * sizeof(int);

    check(cudaMemcpy(d_x, x, fN, cudaMemcpyHostToDevice), "copy x");
    check(cudaMemcpy(d_y, y, fN, cudaMemcpyHostToDevice), "copy y");
    check(cudaMemcpy(d_vx, vx, fN, cudaMemcpyHostToDevice), "copy vx");
    check(cudaMemcpy(d_vy, vy, fN, cudaMemcpyHostToDevice), "copy vy");

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    cudaEventRecord(ev_start);
    neighbour_kernel<<<blocks, threads>>>(d_x, d_y, d_vx, d_vy, d_type, W, H,
                                          d_nb_vx, d_nb_vy, d_nb_dx, d_nb_dy,
                                          d_nb_sepx, d_nb_sepy, d_nb_count, d_nb_close,
                                          d_nb_best_d2, d_nb_best_dx, d_nb_best_dy, d_nb_best_j,
                                          d_nb_ncand, d_nb_cand, d_nb_cdx, d_nb_cdy);
    cudaEventRecord(ev_stop);
    check(cudaGetLastError(), "neighbour_kernel launch");
    cudaEventSynchronize(ev_stop);
    float ms;
    cudaEventElapsedTime(&ms, ev_start, ev_stop);
    kernel_ms += ms;

    check(cudaMemcpy(nb_vx, d_nb_vx, fN, cudaMemcpyDeviceToHost), "copy back");
    check(cudaMemcpy(nb_vy, d_nb_vy, fN, cudaMemcpyDeviceToHost), "copy back");
    check(cudaMemcpy(nb_dx, d_nb_dx, fN, cudaMemcpyDeviceToHost), "copy back");
    check(cudaMemcpy(nb_dy, d_nb_dy, fN, cudaMemcpyDeviceToHost), "copy back");
    check(cudaMemcpy(nb_sepx, d_nb_sepx, fN, cudaMemcpyDeviceToHost), "copy back");
    check(cudaMemcpy(nb_sepy, d_nb_sepy, fN, cudaMemcpyDeviceToHost), "copy back");
    check(cudaMemcpy(nb_count, d_nb_count, iN, cudaMemcpyDeviceToHost), "copy back");
    check(cudaMemcpy(nb_close, d_nb_close, iN, cudaMemcpyDeviceToHost), "copy back");
    check(cudaMemcpy(nb_best_d2, d_nb_best_d2, fN, cudaMemcpyDeviceToHost), "copy back");
    check(cudaMemcpy(nb_best_dx, d_nb_best_dx, fN, cudaMemcpyDeviceToHost), "copy back");
    check(cudaMemcpy(nb_best_dy, d_nb_best_dy, fN, cudaMemcpyDeviceToHost), "copy back");
    check(cudaMemcpy(nb_best_j, d_nb_best_j, iN, cudaMemcpyDeviceToHost), "copy back");
    check(cudaMemcpy(nb_ncand, d_nb_ncand, NPRED_MAX * sizeof(int), cudaMemcpyDeviceToHost), "copy back");
    check(cudaMemcpy(nb_cand, d_nb_cand, iC, cudaMemcpyDeviceToHost), "copy back");
    check(cudaMemcpy(nb_cdx, d_nb_cdx, fC, cudaMemcpyDeviceToHost), "copy back");
    check(cudaMemcpy(nb_cdy, d_nb_cdy, fC, cudaMemcpyDeviceToHost), "copy back");
}

__device__ int cell_of_gpu(float pos, float size)
{
    int c = (int)(pos / size * GRID);
    if (c < 0) c = 0;
    if (c >= GRID) c = GRID - 1;
    return c;
}

__global__ void deposit_kernel(const float *x, const float *y, const int *type,
                               float W, float H, float *dens_raw)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || type[i] != PREY) return;
    int c = cell_of_gpu(y[i], H) * GRID + cell_of_gpu(x[i], W);
    atomicAdd(&dens_raw[c], 1.0f);
}

__global__ void blur_rows_kernel(const float *in, float *out, const float *mask)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= GRID * GRID) return;
    int row = c / GRID, col = c % GRID;
    float s = 0;
    for (int k = -BLUR_R; k <= BLUR_R; k++)
        s += mask[k + BLUR_R] * in[row * GRID + (col + k + GRID) % GRID];
    out[c] = s;
}

__global__ void blur_cols_kernel(const float *in, float *out, const float *mask)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= GRID * GRID) return;
    int row = c / GRID, col = c % GRID;
    float s = 0;
    for (int k = -BLUR_R; k <= BLUR_R; k++)
        s += mask[k + BLUR_R] * in[((row + k + GRID) % GRID) * GRID + col];
    out[c] = s;
}

void compute_density(void)
{
    int threads = 256;
    int cells = GRID * GRID;

    cudaEventRecord(ev_start);
    check(cudaMemset(d_dens_raw, 0, cells * sizeof(float)), "cudaMemset");
    deposit_kernel<<<(N + threads - 1) / threads, threads>>>(d_x, d_y, d_type, W, H, d_dens_raw);
    blur_rows_kernel<<<(cells + threads - 1) / threads, threads>>>(d_dens_raw, d_dens_tmp, d_mask);
    blur_cols_kernel<<<(cells + threads - 1) / threads, threads>>>(d_dens_tmp, d_dens, d_mask);
    cudaEventRecord(ev_stop);
    check(cudaGetLastError(), "density kernels");
    cudaEventSynchronize(ev_stop);
    float ms;
    cudaEventElapsedTime(&ms, ev_start, ev_stop);
    density_ms += ms;

    check(cudaMemcpy(dens, d_dens, cells * sizeof(float), cudaMemcpyDeviceToHost), "copy density");
}

void decide_agents(void)
{
    long total_neighbours = 0, total_too_close = 0;
    int fleeing = 0;

    #pragma omp parallel reduction(+:total_neighbours, total_too_close, fleeing)
    {
    int tid = omp_get_thread_num();
    double t_start = omp_get_wtime();

    #pragma omp for schedule(runtime) nowait
    for (int i = 0; i < N; i++) {
        int   count   = nb_count[i];
        float best_d2 = nb_best_d2[i];
        int   best_j  = nb_best_j[i];

        nvx[i] = vx[i];
        nvy[i] = vy[i];

        if (type[i] == PREY) {
            if (count > 0) {
                float avg_vx = nb_vx[i] / count, avg_vy = nb_vy[i] / count;
                nvx[i] += align_w * (avg_vx - vx[i]);
                nvy[i] += align_w * (avg_vy - vy[i]);
                nvx[i] += cohesion_w * (nb_dx[i] / count);
                nvy[i] += cohesion_w * (nb_dy[i] / count);
                nvx[i] += separation_w * nb_sepx[i];
                nvy[i] += separation_w * nb_sepy[i];
            }
            if (best_j >= 0) {
                float d = sqrtf(best_d2);
                nvx[i] -= FLEE * VMAX * nb_best_dx[i] / d;
                nvy[i] -= FLEE * VMAX * nb_best_dy[i] / d;
                fleeing++;
            }
            total_neighbours += count;
            total_too_close  += nb_close[i];
        } else {
            float hx, hy;
            int planned = plan_hunt(nb_ncand[i], &nb_cand[i * MAXCAND],
                                    &nb_cdx[i * MAXCAND], &nb_cdy[i * MAXCAND], &hx, &hy);
            if (planned >= 0) {
                target[i] = planned;
                nvx[i] += CHASE * VPRED * hx;
                nvy[i] += CHASE * VPRED * hy;
            } else {
                target[i] = best_j;
                if (best_j >= 0) {
                    float d = sqrtf(best_d2);
                    nvx[i] += CHASE * VPRED * nb_best_dx[i] / d;
                    nvy[i] += CHASE * VPRED * nb_best_dy[i] / d;
                }
            }
        }

        float speed = sqrtf(nvx[i] * nvx[i] + nvy[i] * nvy[i]);
        if (speed > 0) {
            nvx[i] = nvx[i] / speed * max_speed(i);
            nvy[i] = nvy[i] / speed * max_speed(i);
        }
    }

    thread_busy[tid] += omp_get_wtime() - t_start;
    }

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        vx[i] = nvx[i];
        vy[i] = nvy[i];
    }

    num_fleeing    = fleeing;
    avg_neighbours = (float)total_neighbours / (N - NPRED);
    avg_too_close  = (float)total_too_close / (N - NPRED);
}

int catch_prey(void)
{
    int caught = 0;
    for (int i = 0; i < NPRED; i++) {
        int j = target[i];
        if (j < 0) continue;
        float dx = wrap_delta(x[j] - x[i], W);
        float dy = wrap_delta(y[j] - y[i], H);
        if (dx * dx + dy * dy < RCATCH * RCATCH) {
            x[j] = rand_range(0, W);
            y[j] = rand_range(0, H);
            caught++;
        }
    }
    return caught;
}

void save_density(FILE *f, int step)
{
    fprintf(f, "%d", step);
    for (int c = 0; c < GRID * GRID; c++)
        fprintf(f, ",%.3f", dens[c]);
    fprintf(f, "\n");
}

void save_frame(FILE *f, int step)
{
    for (int i = 0; i < N; i++)
        fprintf(f, "%d,%.3f,%.3f,%.3f,%.3f,%d\n", step, x[i], y[i], vx[i], vy[i], type[i]);
}

float polarization(void)
{
    float sx = 0, sy = 0;
    for (int i = NPRED; i < N; i++) {
        sx += vx[i];
        sy += vy[i];
    }
    return sqrtf(sx * sx + sy * sy) / ((N - NPRED) * VMAX);
}

void read_controls(void)
{
    FILE *f = fopen("controls.txt", "r");
    if (f == NULL) return;
    float a, c, sep;
    if (fscanf(f, "%f %f %f", &a, &c, &sep) == 3) {
        align_w = a; cohesion_w = c; separation_w = sep;
    }
    fclose(f);
}

int main(void)
{
    W = H = sqrtf(N / DENSITY); 
    printf("N = %d agents, world %.1f x %.1f, %d steps, %d threads\n",
           N, W, H, STEPS, omp_get_max_threads());

    srand(42);
    init_agents();
    init_mask();
    gpu_init();

    FILE *out = NULL;
    FILE *dout = NULL;
    FILE *mout = NULL;
    if (SAVE_EVERY > 0) {
        out = fopen("frames.csv", "w");
        if (out == NULL) {
            printf("could not open frames.csv\n");
            return 1;
        }
        fprintf(out, "step,x,y,vx,vy,type\n");
          dout = fopen("density.csv", "w");
          mout = fopen("metrics.csv", "w");
          fprintf(mout, "step,polarization,neighbours,fleeing,catches,ms_per_step\n");
    }

    double neigh_time = 0, density_time = 0, decide_time = 0;
    double start = omp_get_wtime();

    double last_save = omp_get_wtime();
    for (int step = 0; step < STEPS; step++) {
        if (step % 10 == 0)
            read_controls();
        double t0 = omp_get_wtime();
        neighbour_search();
        double t1 = omp_get_wtime();
        compute_density();
        double t2 = omp_get_wtime();
        decide_agents();
        double t3 = omp_get_wtime();
        neigh_time   += t1 - t0;
        density_time += t2 - t1;
        decide_time  += t3 - t2;

        total_catches += catch_prey();
        move_agents();
        if (SAVE_EVERY > 0 && step % SAVE_EVERY == 0) {
            save_frame(out, step);
            save_density(dout, step);
            double now = omp_get_wtime();
            fprintf(mout, "%d,%.4f,%.2f,%d,%d,%.3f\n", step, polarization(), avg_neighbours,
                    num_fleeing, total_catches, 1000 * (now - last_save) / SAVE_EVERY);
            last_save = now;
            fflush(out); fflush(dout); fflush(mout);
        }
        if (step % 100 == 0)
            printf("step %4d: polarization %.3f | avg neighbours %5.1f | fleeing %3d | catches %d\n",
                   step, polarization(), avg_neighbours, num_fleeing, total_catches);
    }
    double total_time = omp_get_wtime() - start;

    printf("final polarization: %.3f\n", polarization());
    printf("total time      : %.3f s (%.2f ms per step)\n", total_time, 1000 * total_time / STEPS);
    printf("neighbour search: %.3f s (%.1f%% of total)\n", neigh_time, 100 * neigh_time / total_time);
    printf("  GPU kernel    : %.3f s, copies + launch: %.3f s\n",
             kernel_ms / 1000, neigh_time - kernel_ms / 1000);
    printf("density field   : %.3f s (%.1f%% of total)\n", density_time, 100 * density_time / total_time);
    printf("  GPU kernels   : %.3f s, copy back + launch: %.3f s\n",
           density_ms / 1000, density_time - density_ms / 1000);
    printf("decide agents   : %.3f s (%.1f%% of total)\n", decide_time, 100 * decide_time / total_time);
    
    int nthreads = omp_get_max_threads();
    double max_busy = 0, sum_busy = 0;
    for (int t = 0; t < nthreads; t++) {
        if (thread_busy[t] > max_busy) max_busy = thread_busy[t];
        sum_busy += thread_busy[t];
    }
    printf("thread busy time:\n");
    for (int t = 0; t < nthreads; t++)
        printf("  thread %2d: %6.3f s  (%5.1f%% of the busiest)\n",
               t, thread_busy[t], 100 * thread_busy[t] / max_busy);
    printf("load imbalance  : %.2f (busiest / average, 1.00 = perfect)\n",
           max_busy / (sum_busy / nthreads));
    if (out != NULL) {
        fclose(out);
        fclose(dout);
        fclose(mout);
        printf("saved frames.csv, density.csv and metrics.csv\n");
    }
    gpu_free();
    return 0;
}
