#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>

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
#define ALIGN  0.1f      // how strongly agents turn toward neighbours
#define RSEP       1.0f   // separation radius
#define SEPARATION 0.5f   // strength of the push away from close neighbours
#define COHESION   0.02f  // strength of the pull toward the group's centre
#ifndef SAVE_EVERY
#define SAVE_EVERY 5      // save positions every 5 steps (0 = don't save)
#endif
#define PREY      0
#define PREDATOR  1
#define NPRED    (N / 50) // the first 2% of agents are predators
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
#define CAP        (2 * N) // room for local agents + ghost copies
#define RGHOST     RHUNT  // agents this close to a strip edge are copied to the neighbour
#define FPA        5      // floats per packed agent: x, y, vx, vy, type

float W, H;              // world size, set in main from N and DENSITY
float x[CAP], y[CAP];    // positions (local agents first, then ghosts)
float vx[CAP], vy[CAP];  // velocities
float nvx[N], nvy[N];    // new velocities (computed each step)
float avg_neighbours;    // average neighbours within R
float avg_too_close;     // average neighbours within RSEP
int   type[CAP];         // PREY or PREDATOR
int   target[N];         // predator: index of the prey it is chasing (-1 = none)
int   num_fleeing;       // prey that currently see a predator
int   total_catches = 0;
double thread_busy[64];  // seconds each thread spent working in flock_agents

int   rank, nprocs;      // this process and how many there are
int   left, right;       // ranks owning the strips to the left and right
float xmin, xmax;        // this rank's strip: xmin <= x < xmax
int   nlocal = 0;        // agents this rank owns
int   nghost = 0;        // copies of neighbours' border agents (stored after the local ones)
int   step_catches;      // catches this step on this rank
float sendbuf_l[N * FPA], sendbuf_r[N * FPA], recvbuf[N * FPA];
float dens_local[GRID * GRID];
double comm_time = 0;

float nb_vx[N], nb_vy[N];
float nb_dx[N], nb_dy[N];
float nb_sepx[N], nb_sepy[N];
int   nb_count[N];
int   nb_close[N];
float nb_best_d2[N];
float nb_best_dx[N], nb_best_dy[N];
int   nb_best_j[N];
int   nb_ncand[N];
int   nb_cand[N * MAXCAND];
float nb_cdx[N * MAXCAND], nb_cdy[N * MAXCAND];
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
    for (int i = 0; i < nlocal; i++) {
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

void compute_density(void)
{
    #pragma omp parallel for
    for (int c = 0; c < GRID * GRID; c++)
        dens_local[c] = 0;

    #pragma omp parallel for
    for (int i = 0; i < nlocal; i++) {
        if (type[i] != PREY) continue;
        int c = cell_of(y[i], H) * GRID + cell_of(x[i], W);
        #pragma omp atomic
        dens_local[c] += 1.0f;
    }

    double t0 = MPI_Wtime();
    MPI_Reduce(dens_local, dens_raw, GRID * GRID, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Bcast(dens_raw, GRID * GRID, MPI_FLOAT, 0, MPI_COMM_WORLD);
    comm_time += MPI_Wtime() - t0;

    #pragma omp parallel for
    for (int row = 0; row < GRID; row++)
        for (int col = 0; col < GRID; col++) {
            float s = 0;
            for (int k = -BLUR_R; k <= BLUR_R; k++)
                s += mask[k + BLUR_R] * dens_raw[row * GRID + (col + k + GRID) % GRID];
            dens_tmp[row * GRID + col] = s;
        }

    #pragma omp parallel for
    for (int row = 0; row < GRID; row++)
        for (int col = 0; col < GRID; col++) {
            float s = 0;
            for (int k = -BLUR_R; k <= BLUR_R; k++)
                s += mask[k + BLUR_R] * dens_tmp[((row + k + GRID) % GRID) * GRID + col];
            dens[row * GRID + col] = s;
        }
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

void neighbour_search(void)
{
    #pragma omp parallel for schedule(runtime)
    for (int i = 0; i < nlocal; i++) {
        float sum_vx = 0, sum_vy = 0;
        float sum_dx = 0, sum_dy = 0;
        float sep_x = 0, sep_y = 0;
        int count = 0, too_close = 0;

        float best_d2 = (type[i] == PREY) ? RFLEE * RFLEE : RHUNT * RHUNT;
        float best_dx = 0, best_dy = 0;
        int   best_j  = -1;
        int   ncand   = 0;

        for (int j = 0; j < nlocal + nghost; j++) {
            if (j == i) continue;
            float dx = wrap_delta(x[j] - x[i], W);
            float dy = wrap_delta(y[j] - y[i], H);
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
    for (int i = 0; i < nlocal; i++) {
        int   count   = nb_count[i];
        float best_d2 = nb_best_d2[i];
        int   best_j  = nb_best_j[i];

        nvx[i] = vx[i];
        nvy[i] = vy[i];

        if (type[i] == PREY) {
            if (count > 0) {
                float avg_vx = nb_vx[i] / count, avg_vy = nb_vy[i] / count;
                nvx[i] += ALIGN * (avg_vx - vx[i]);
                nvy[i] += ALIGN * (avg_vy - vy[i]);
                nvx[i] += COHESION * (nb_dx[i] / count);
                nvy[i] += COHESION * (nb_dy[i] / count);
                nvx[i] += SEPARATION * nb_sepx[i];
                nvy[i] += SEPARATION * nb_sepy[i];
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
    for (int i = 0; i < nlocal; i++) {
        vx[i] = nvx[i];
        vy[i] = nvy[i];
    }

    num_fleeing    = fleeing;
    avg_neighbours = total_neighbours;
    avg_too_close  = total_too_close;
}

int catch_prey(void)
{
    int caught = 0;
    for (int i = 0; i < nlocal; i++) {
        if (type[i] != PREDATOR) continue;
        int j = target[i];
        if (j < 0 || j >= nlocal) continue;
        float dx = wrap_delta(x[j] - x[i], W);
        float dy = wrap_delta(y[j] - y[i], H);
        if (dx * dx + dy * dy < RCATCH * RCATCH) {
            x[j] = rand_range(xmin, xmax);
            y[j] = rand_range(0, H);
            caught++;
        }
    }
    return caught;
}

int owner_of(float px)
{
    int r = (int)(px / (W / nprocs));
    if (r < 0) r = 0;
    if (r >= nprocs) r = nprocs - 1;
    return r;
}

void pack(float *buf, int k, int i)
{
    buf[k * FPA + 0] = x[i];
    buf[k * FPA + 1] = y[i];
    buf[k * FPA + 2] = vx[i];
    buf[k * FPA + 3] = vy[i];
    buf[k * FPA + 4] = type[i];
}

void unpack(const float *buf, int k, int i)
{
    x[i]    = buf[k * FPA + 0];
    y[i]    = buf[k * FPA + 1];
    vx[i]   = buf[k * FPA + 2];
    vy[i]   = buf[k * FPA + 3];
    type[i] = (int)buf[k * FPA + 4];
}

int shift(int to, int from, const float *sbuf, int nsend, float *rbuf)
{
    int nrecv;
    MPI_Status st;
    if (rank % 2 == 0) {
        MPI_Send(&nsend, 1, MPI_INT, to, 0, MPI_COMM_WORLD);
        MPI_Send(sbuf, nsend * FPA, MPI_FLOAT, to, 1, MPI_COMM_WORLD);
        MPI_Recv(&nrecv, 1, MPI_INT, from, 0, MPI_COMM_WORLD, &st);
        MPI_Recv(rbuf, nrecv * FPA, MPI_FLOAT, from, 1, MPI_COMM_WORLD, &st);
    } else {
        MPI_Recv(&nrecv, 1, MPI_INT, from, 0, MPI_COMM_WORLD, &st);
        MPI_Recv(rbuf, nrecv * FPA, MPI_FLOAT, from, 1, MPI_COMM_WORLD, &st);
        MPI_Send(&nsend, 1, MPI_INT, to, 0, MPI_COMM_WORLD);
        MPI_Send(sbuf, nsend * FPA, MPI_FLOAT, to, 1, MPI_COMM_WORLD);
    }
    return nrecv;
}

void migrate(void)
{
    if (nprocs == 1) return;
    double t0 = MPI_Wtime();

    int nl = 0, nr = 0, keep = 0;
    for (int i = 0; i < nlocal; i++) {
        int o = owner_of(x[i]);
        if (o == rank) {
            x[keep] = x[i]; y[keep] = y[i];
            vx[keep] = vx[i]; vy[keep] = vy[i];
            type[keep] = type[i];
            keep++;
        } else if (o == right) {
            pack(sendbuf_r, nr++, i);
        } else {
            pack(sendbuf_l, nl++, i);
        }
    }
    nlocal = keep;

    int n = shift(right, left, sendbuf_r, nr, recvbuf);
    for (int k = 0; k < n; k++) unpack(recvbuf, k, nlocal++);
    n = shift(left, right, sendbuf_l, nl, recvbuf);
    for (int k = 0; k < n; k++) unpack(recvbuf, k, nlocal++);

    comm_time += MPI_Wtime() - t0;
}

void exchange_ghosts(void)
{
    nghost = 0;
    if (nprocs == 1) return;
    double t0 = MPI_Wtime();

    int nl = 0, nr = 0;
    for (int i = 0; i < nlocal; i++) {
        if (x[i] - xmin < RGHOST) pack(sendbuf_l, nl++, i);
        if (xmax - x[i] < RGHOST) pack(sendbuf_r, nr++, i);
    }

    int n = shift(right, left, sendbuf_r, nr, recvbuf);
    for (int k = 0; k < n; k++) unpack(recvbuf, k, nlocal + nghost++);
    n = shift(left, right, sendbuf_l, nl, recvbuf);
    for (int k = 0; k < n; k++) unpack(recvbuf, k, nlocal + nghost++);


    comm_time += MPI_Wtime() - t0;
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
    if (rank != 0) {
        for (int i = 0; i < nlocal; i++) pack(sendbuf_r, i, i);
        MPI_Send(&nlocal, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(sendbuf_r, nlocal * FPA, MPI_FLOAT, 0, 3, MPI_COMM_WORLD);
        return;
    }
    for (int i = 0; i < nlocal; i++)
        fprintf(f, "%d,%.3f,%.3f,%.3f,%.3f,%d\n", step, x[i], y[i], vx[i], vy[i], type[i]);
    for (int r = 1; r < nprocs; r++) {
        int n;
        MPI_Status st;
        MPI_Recv(&n, 1, MPI_INT, r, 2, MPI_COMM_WORLD, &st);
        MPI_Recv(recvbuf, n * FPA, MPI_FLOAT, r, 3, MPI_COMM_WORLD, &st);
        for (int k = 0; k < n; k++)
            fprintf(f, "%d,%.3f,%.3f,%.3f,%.3f,%d\n", step, recvbuf[k * FPA], recvbuf[k * FPA + 1],
                    recvbuf[k * FPA + 2], recvbuf[k * FPA + 3], (int)recvbuf[k * FPA + 4]);
    }
}

float global_polarization;
float global_neighbours;
int   global_fleeing, global_agents;

void gather_metrics(void)
{
    double t0 = MPI_Wtime();
    float sx = 0, sy = 0;
    for (int i = 0; i < nlocal; i++) {
        if (type[i] != PREY) continue;
        sx += vx[i];
        sy += vy[i];
    }
    double local[6] = { sx, sy, avg_neighbours, num_fleeing, step_catches, nlocal };
    double global[6];
    MPI_Reduce(local, global, 6, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        global_polarization = sqrt(global[0] * global[0] + global[1] * global[1]) / ((N - NPRED) * VMAX);
        global_neighbours   = global[2] / (N - NPRED);
        global_fleeing      = (int)global[3];
        total_catches      += (int)global[4];
        global_agents       = (int)global[5];
    }
    comm_time += MPI_Wtime() - t0;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    W = H = sqrtf(N / DENSITY);
    left  = (rank - 1 + nprocs) % nprocs;
    right = (rank + 1) % nprocs;
    xmin  = rank * (W / nprocs);
    xmax  = (rank + 1) * (W / nprocs);

    if (nprocs > 1 && (nprocs % 2 != 0 || W / nprocs < 2 * RGHOST)) {
        if (rank == 0)
            printf("need an even number of processes and strips at least %.0f wide\n", 2 * RGHOST);
        MPI_Finalize();
        return 1;
    }
    if (rank == 0)
        printf("N = %d agents, world %.1f x %.1f, %d steps, %d processes x %d threads\n",
               N, W, H, STEPS, nprocs, omp_get_max_threads());

    if (rank == 0) {
        srand(42);
        nlocal = N;
        init_agents();
    } else {
        srand(42 + rank);
    }
    MPI_Bcast(x, N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(y, N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(vx, N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(vy, N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(type, N, MPI_INT, 0, MPI_COMM_WORLD);

    nlocal = 0;
    for (int i = 0; i < N; i++) {
        if (owner_of(x[i]) != rank) continue;
        x[nlocal] = x[i]; y[nlocal] = y[i];
        vx[nlocal] = vx[i]; vy[nlocal] = vy[i];
        type[nlocal] = type[i];
        nlocal++;
    }

    init_mask();
    FILE *out = NULL;
    FILE *dout = NULL;
    if (SAVE_EVERY > 0 && rank == 0) {
        out = fopen("frames.csv", "w");
        if (out == NULL) {
            printf("could not open frames.csv\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fprintf(out, "step,x,y,vx,vy,type\n");
        dout = fopen("density.csv", "w");
    }

    double neigh_time = 0, density_time = 0, decide_time = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    for (int step = 0; step < STEPS; step++) {
        exchange_ghosts();
        double t0 = MPI_Wtime();
        neighbour_search();
        double t1 = MPI_Wtime();
        compute_density();
        double t2 = MPI_Wtime();
        decide_agents();
        double t3 = MPI_Wtime();
        neigh_time   += t1 - t0;
        density_time += t2 - t1;
        decide_time  += t3 - t2;

        step_catches = catch_prey();
        move_agents();
        migrate();
        gather_metrics();

        if (SAVE_EVERY > 0 && step % SAVE_EVERY == 0) {
            save_frame(out, step);
            if (rank == 0) save_density(dout, step);
        }
        if (rank == 0 && step % 100 == 0)
            printf("step %4d: polarization %.3f | avg neighbours %5.1f | fleeing %3d | catches %d | agents %d\n",
                   step, global_polarization, global_neighbours, global_fleeing, total_catches, global_agents);
    }
    double total_time = MPI_Wtime() - start;

    double mine[5] = { total_time, neigh_time, density_time, decide_time, comm_time };
    double worst[5];
    MPI_Reduce(mine, worst, 5, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("final polarization: %.3f\n", global_polarization);
        printf("total time      : %.3f s (%.2f ms per step)\n", worst[0], 1000 * worst[0] / STEPS);
        printf("neighbour search: %.3f s (%.1f%% of total)\n", worst[1], 100 * worst[1] / worst[0]);
        printf("density field   : %.3f s (%.1f%% of total)\n", worst[2], 100 * worst[2] / worst[0]);
        printf("decide agents   : %.3f s (%.1f%% of total)\n", worst[3], 100 * worst[3] / worst[0]);
        printf("communication   : %.3f s (%.1f%% of total)\n", worst[4], 100 * worst[4] / worst[0]);
        printf("(times are the slowest rank's)\n");
    }
    if (out != NULL) {
        fclose(out);
        fclose(dout);
        printf("saved frames.csv and density.csv\n");
    }
    MPI_Finalize();
    return 0;
}
