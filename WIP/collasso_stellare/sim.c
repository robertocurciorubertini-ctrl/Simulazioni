/*
 * sim.c — SPH Stellar Collapse Simulation Core
 *
 * Versione rivista:
 * - API JavaScript invariata
 * - layout dei buffer invariato
 * - fisica SPH resa più coerente
 *   * densità/pressione inizializzate già in sim_init()
 *   * gravità softening con legge ~ 1/(r^2 + eps^2)^(3/2)
 *   * forza di pressione SPH simmetrica
 *   * viscosità artificiale di Monaghan
 *   * equazione energetica SPH tramite termine di compressione/urto
 *   * integrazione kick-drift-kick
 *
 * NOTA IMPORTANTE:
 * Questo file mantiene intenzionalmente intatta l'interfaccia con l'HTML:
 *   sim_set_params, sim_init, sim_step, sim_get_state,
 *   sim_get_diagnostics, sim_get_phase, sim_get_N
 * e mantiene invariato anche il layout dei buffer di uscita.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <emscripten/emscripten.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#define N_MAX      1000
#define CANVAS_W   760.0f
#define CANVAS_H   760.0f

#define VISC_ALPHA  1.0f
#define VISC_BETA   2.0f
#define VISC_EPS    0.01f

/* Parametri runtime */
static float g_G       = 12.0f;
static float g_kpress  = 0.004f;
static float g_gamma   = 1.66f;
static float g_hsmooth = 20.0f;
static float g_soft    = 200.0f;   /* softening gravitazionale in px^2 */
static float g_cool0   = 0.9980f;
static float g_cool1   = 0.9995f;
static float g_mass    = 1.0f;

/* Soglie di fase */
static float g_thr01    = 0.15f;
static float g_thr12    = 0.60f;
static float g_thr23t   = 800.0f;
static float g_thr23rho = 3.0f;
static float g_thr34t   = 1100.0f;

/* Stato particelle */
int   N         = 0;
float px[N_MAX], py[N_MAX];
float vx[N_MAX], vy[N_MAX];
float ax[N_MAX], ay[N_MAX];
float rho[N_MAX], p[N_MAX], u[N_MAX];
int   alive[N_MAX];

float sim_time  = 0.0f;
int   phase     = 0;
int   ns_active = 0;
float ns_x      = 0.0f;
float ns_y      = 0.0f;
float ns_pulse  = 0.0f;

float diag_buf[8];
float state_buf[8 + N_MAX * 4];
int   diag_n_alive = 0;

/* work arrays interni */
static float du_dt[N_MAX];

static inline float clampf(float x, float a, float b) {
    return (x < a) ? a : ((x > b) ? b : x);
}

/* Kernel B-spline cubico 2D */
static float kernel(float dist) {
    float h = g_hsmooth;
    float q = dist / h;
    if (q >= 2.0f) return 0.0f;
    float norm = 10.0f / (7.0f * M_PI * h * h);
    if (q <= 1.0f)
        return norm * (1.0f - 1.5f*q*q + 0.75f*q*q*q);
    return norm * 0.25f * powf(2.0f - q, 3.0f);
}

/* gradiente di W rispetto a r_i - r_j */
static void kernel_grad(float dx, float dy, float dist, float *gx, float *gy) {
    float h = g_hsmooth;
    float q = dist / h;
    if (q >= 2.0f || dist < 1e-6f) {
        *gx = 0.0f;
        *gy = 0.0f;
        return;
    }
    float norm = 10.0f / (7.0f * M_PI * h * h * h);
    float dWdq = (q <= 1.0f)
        ? norm * (-3.0f*q + 2.25f*q*q)
        : norm * (-0.75f * powf(2.0f - q, 2.0f));
    *gx = dWdq * dx / dist;
    *gy = dWdq * dy / dist;
}

static void center_of_mass(float *cx, float *cy) {
    float sx = 0.0f, sy = 0.0f;
    int cnt = 0;
    for (int i = 0; i < N; i++) {
        if (alive[i]) {
            sx += px[i];
            sy += py[i];
            cnt++;
        }
    }
    if (cnt > 0) {
        *cx = sx / cnt;
        *cy = sy / cnt;
    } else {
        *cx = CANVAS_W / 2.0f;
        *cy = CANVAS_H / 2.0f;
    }
}

static void update_density_pressure(void) {
    float h2   = 2.0f * g_hsmooth;
    float h2sq = h2 * h2;

    for (int i = 0; i < N; i++) {
        if (!alive[i]) continue;

        rho[i] = 0.01f; /* floor numerico */

        for (int j = 0; j < N; j++) {
            if (!alive[j]) continue;
            float dx = px[i] - px[j];
            float dy = py[i] - py[j];
            float r2 = dx*dx + dy*dy;
            if (r2 < h2sq) {
                rho[i] += g_mass * kernel(sqrtf(r2));
            }
        }

        p[i] = g_kpress * powf(rho[i], g_gamma);

        /* contributo termico esplicito nelle fasi più calde */
        if (phase >= 2) {
            p[i] += (g_gamma - 1.0f) * rho[i] * u[i];
        }

        /* floor di sicurezza */
        if (rho[i] < 1e-4f) rho[i] = 1e-4f;
        if (p[i]   < 0.0f)  p[i]   = 0.0f;
    }
}

static void update_forces_and_energy_rhs(void) {
    float h2 = 2.0f * g_hsmooth;

    for (int i = 0; i < N; i++) {
        ax[i] = 0.0f;
        ay[i] = 0.0f;
        du_dt[i] = 0.0f;
    }

    for (int i = 0; i < N; i++) {
        if (!alive[i]) continue;

        for (int j = i + 1; j < N; j++) {
            if (!alive[j]) continue;

            float rx = px[j] - px[i];
            float ry = py[j] - py[i];
            float r2 = rx*rx + ry*ry;
            float dist = sqrtf(r2 + 1e-12f);

            /* Gravità softened: a ~ r / (r^2 + eps^2)^(3/2)
               Qui g_soft ha dimensione di px^2. */
            float invr3 = 1.0f / powf(r2 + g_soft, 1.5f);
            float ag = g_G * g_mass * invr3;

            ax[i] += ag * rx;
            ay[i] += ag * ry;
            ax[j] -= ag * rx;
            ay[j] -= ag * ry;

            if (dist < h2) {
                float gx, gy;
                /* gradiente rispetto a r_i - r_j = -(r_j - r_i) */
                kernel_grad(-rx, -ry, dist, &gx, &gy);

                float rhoi = fmaxf(rho[i], 1e-4f);
                float rhoj = fmaxf(rho[j], 1e-4f);

                float pi2 = p[i] / (rhoi * rhoi);
                float pj2 = p[j] / (rhoj * rhoj);

                /* viscosità artificiale di Monaghan */
                float dvx = vx[i] - vx[j];
                float dvy = vy[i] - vy[j];
                float vij_rij = dvx * (-rx) + dvy * (-ry);
                float Pi_ij = 0.0f;

                if (vij_rij < 0.0f) {
                    float csi = sqrtf(fmaxf(g_gamma * p[i] / rhoi, 1e-6f));
                    float csj = sqrtf(fmaxf(g_gamma * p[j] / rhoj, 1e-6f));
                    float csij = 0.5f * (csi + csj);
                    float rhoij = 0.5f * (rhoi + rhoj);
                    float muij = g_hsmooth * vij_rij / (r2 + VISC_EPS * g_hsmooth * g_hsmooth);
                    Pi_ij = (-VISC_ALPHA * csij * muij + VISC_BETA * muij * muij) / rhoij;
                    if (Pi_ij < 0.0f) Pi_ij = 0.0f;
                }

                float coeff = g_mass * (pi2 + pj2 + Pi_ij);

                /* pressione/viscosità: forma antisimmetrica */
                ax[i] -= coeff * gx;
                ay[i] -= coeff * gy;
                ax[j] += coeff * gx;
                ay[j] += coeff * gy;

                /* equazione dell'energia interna:
                   du_i/dt = 1/2 sum_j m_j (Pi/rho^2 + Pj/rho^2 + Pi_ij) v_ij · gradW_ij
                   con gradW_ij rispetto a r_i - r_j */
                float vij_gradW = dvx * gx + dvy * gy;
                float e_coeff = 0.5f * g_mass * (pi2 + pj2 + Pi_ij) * vij_gradW;
                du_dt[i] += e_coeff;
                du_dt[j] += e_coeff;
            }
        }
    }
}

static void recount_alive(void) {
    diag_n_alive = 0;
    for (int i = 0; i < N; i++) {
        if (alive[i]) diag_n_alive++;
    }
}

EMSCRIPTEN_KEEPALIVE
void sim_set_params(float g_grav, float kpress, float cool, float soft) {
    if (g_grav > 0.0f) g_G = g_grav;
    if (kpress > 0.0f) g_kpress = kpress;
    if (cool > 0.0f && cool < 1.0f) {
        g_cool0 = cool;
        g_cool1 = 1.0f - (1.0f - cool) * 0.25f;
    }
    if (soft > 0.0f) g_soft = soft;

    float gs   = g_G / 12.0f;
    g_thr01    = 0.15f * gs;
    g_thr12    = 0.60f * gs;
    g_thr23rho = 3.0f  * gs;
    g_thr23t   = 800.0f  / fmaxf(gs, 1e-3f);
    g_thr34t   = 1100.0f / fmaxf(gs, 1e-3f);
}

EMSCRIPTEN_KEEPALIVE
void sim_init(int n_req, float R, float omega0, float u0) {
    N         = (n_req > N_MAX) ? N_MAX : ((n_req < 1) ? 1 : n_req);
    sim_time  = 0.0f;
    phase     = 0;
    ns_active = 0;
    ns_x      = 0.0f;
    ns_y      = 0.0f;
    ns_pulse  = 0.0f;

    float cx = CANVAS_W / 2.0f;
    float cy = CANVAS_H / 2.0f;

    for (int i = 0; i < N; i++) {
        float r = R * sqrtf((float)rand() / (float)RAND_MAX);
        float theta = ((float)rand() / (float)RAND_MAX) * 2.0f * M_PI;

        px[i] = cx + r * cosf(theta);
        py[i] = cy + r * sinf(theta);

        /* rotazione rigida */
        vx[i] = -omega0 * (py[i] - cy);
        vy[i] =  omega0 * (px[i] - cx);

        ax[i] = 0.0f;
        ay[i] = 0.0f;
        u[i]  = fmaxf(u0, 1e-4f);
        rho[i] = 0.0f;
        p[i]   = 0.0f;
        alive[i] = 1;
    }

    for (int i = N; i < N_MAX; i++) {
        alive[i] = 0;
    }

    recount_alive();
    update_density_pressure();
    update_forces_and_energy_rhs();
}

EMSCRIPTEN_KEEPALIVE
void sim_step(int iterations) {
    float dt = 0.12f;

    for (int step = 0; step < iterations; step++) {

        /* thinning in fase finale */
        if (phase == 4 && diag_n_alive > 80 && ((int)(sim_time / dt) % 50 == 0)) {
            float far2 = (CANVAS_W * 1.2f) * (CANVAS_W * 1.2f);
            for (int i = 0; i < N; i++) {
                if (!alive[i]) continue;
                float ddx = px[i] - ns_x;
                float ddy = py[i] - ns_y;
                if (rho[i] < 0.003f && (ddx*ddx + ddy*ddy) > far2) {
                    alive[i] = 0;
                }
            }
            recount_alive();
        }

        /* 1) kick di mezzo passo */
        for (int i = 0; i < N; i++) {
            if (!alive[i]) continue;
            vx[i] += 0.5f * dt * ax[i];
            vy[i] += 0.5f * dt * ay[i];
        }

        /* 2) drift */
        for (int i = 0; i < N; i++) {
            if (!alive[i]) continue;
            px[i] += dt * vx[i];
            py[i] += dt * vy[i];
        }

        /* 3) nuovo stato idrodinamico */
        update_density_pressure();
        update_forces_and_energy_rhs();

        /* 4) secondo mezzo kick + energia interna */
        float max_rho = 0.0f;
        float sum_rho = 0.0f;
        float cmx, cmy;
        center_of_mass(&cmx, &cmy);
        recount_alive();

        for (int i = 0; i < N; i++) {
            if (!alive[i]) continue;

            vx[i] += 0.5f * dt * ax[i];
            vy[i] += 0.5f * dt * ay[i];

            /* heating/cooling */
            u[i] += dt * du_dt[i];
            u[i] *= (phase == 0) ? g_cool0 : g_cool1;

            /* floors / clamp */
            u[i] = clampf(u[i], 1e-5f, 50.0f);

            if (rho[i] > max_rho) max_rho = rho[i];
            sum_rho += rho[i];

            /* confinamento morbido per evitare escape numerici dal canvas */
            if (px[i] < -0.25f * CANVAS_W || px[i] > 1.25f * CANVAS_W ||
                py[i] < -0.25f * CANVAS_H || py[i] > 1.25f * CANVAS_H) {
                float cx = CANVAS_W * 0.5f;
                float cy = CANVAS_H * 0.5f;
                float dx = cx - px[i];
                float dy = cy - py[i];
                vx[i] += 0.002f * dx;
                vy[i] += 0.002f * dy;
            }
        }

        /* transizioni di fase */
        if (phase == 0 && max_rho > g_thr01) phase = 1;
        if (phase == 1 && max_rho > g_thr12) phase = 2;
        if (phase == 2 && (sim_time > g_thr23t || max_rho > g_thr23rho)) phase = 3;

        /* fase 3: bounce / shock */
        if (phase == 3) {
            for (int i = 0; i < N; i++) {
                if (!alive[i]) continue;
                float dx = px[i] - cmx;
                float dy = py[i] - cmy;
                float d  = sqrtf(dx*dx + dy*dy);
                if (d > 1e-3f && d < 40.0f) {
                    float boost = 6.5f * (1.0f - d / 40.0f);
                    vx[i] += (dx / d) * boost;
                    vy[i] += (dy / d) * boost;
                    u[i]  += 1.5f * boost;
                }
            }
            if (sim_time > g_thr34t) {
                phase = 4;
                ns_active = 1;
                ns_x = cmx;
                ns_y = cmy;
            }
        }

        /* fase 4: assorbimento nel core compatto */
        if (ns_active) {
            ns_pulse += 0.3f;
            for (int i = 0; i < N; i++) {
                if (!alive[i]) continue;
                float dx = px[i] - ns_x;
                float dy = py[i] - ns_y;
                if (dx*dx + dy*dy < 144.0f) {
                    alive[i] = 0;
                }
            }
            recount_alive();
        }

        sim_time += dt;
    }
}

EMSCRIPTEN_KEEPALIVE
int sim_get_phase(void) { return phase; }

EMSCRIPTEN_KEEPALIVE
int sim_get_N(void) { return N; }

EMSCRIPTEN_KEEPALIVE
float* sim_get_diagnostics(void) {
    float mr = 0.0f, mu = 0.0f, sr = 0.0f, scs2 = 0.0f;
    int cnt = 0;

    for (int i = 0; i < N; i++) {
        if (!alive[i]) continue;
        if (rho[i] > mr) mr = rho[i];
        if (u[i]   > mu) mu = u[i];
        sr += rho[i];
        scs2 += g_gamma * p[i] / fmaxf(rho[i], 1e-4f);
        cnt++;
    }

    float mean_rho = (cnt > 0) ? sr / cnt : 0.0f;
    float mean_cs2 = (cnt > 0) ? scs2 / cnt : 0.0f;
    float L_J = 0.0f, M_J = 0.0f;

    if (g_G > 0.0f && mean_rho > 1e-6f) {
        L_J = sqrtf(mean_cs2 / (g_G * mean_rho));
        M_J = mean_rho * L_J * L_J;
    }

    diag_buf[0] = mr;
    diag_buf[1] = mu;
    diag_buf[2] = mean_rho;
    diag_buf[3] = sim_time;
    diag_buf[4] = M_J;
    diag_buf[5] = L_J;
    diag_buf[6] = (float)diag_n_alive;
    diag_buf[7] = (float)phase;
    return diag_buf;
}

EMSCRIPTEN_KEEPALIVE
float* sim_get_state(void) {
    float cmx, cmy;
    center_of_mass(&cmx, &cmy);

    recount_alive();

    state_buf[0] = (float)diag_n_alive;
    state_buf[1] = (float)ns_active;
    state_buf[2] = ns_x;
    state_buf[3] = ns_y;
    state_buf[4] = ns_pulse;
    state_buf[5] = cmx;
    state_buf[6] = cmy;
    state_buf[7] = 0.0f;

    int out = 8;
    for (int i = 0; i < N; i++) {
        if (alive[i]) {
            state_buf[out++] = px[i];
            state_buf[out++] = py[i];
            state_buf[out++] = u[i];
            state_buf[out++] = rho[i];
        }
    }
    return state_buf;
}
