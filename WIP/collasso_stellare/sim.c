/*
 * sim.c — SPH Stellar Collapse Simulation Core
 *
 * Compilare con Emscripten:
 *
 *   emcc sim.c -o sim.js \
 *     -s EXPORTED_FUNCTIONS='["_sim_set_params","_sim_init","_sim_step","_sim_get_state","_sim_get_diagnostics","_sim_get_phase","_sim_get_N"]' \
 *     -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap"]' \
 *     -s MODULARIZE=1 -s EXPORT_NAME=SimModule \
 *     -s ALLOW_MEMORY_GROWTH=1 \
 *     -O2 -lm
 *
 * I file prodotti (sim.js + sim.wasm) devono stare nella stessa cartella
 * dell'HTML. Aprire con un server locale (es. python3 -m http.server).
 *
 * API JavaScript:
 *   sim_set_params(G, kpress, cool, soft)  — imposta parametri fisici
 *   sim_init(N, R, omega0, u0)             — inizializza la nube
 *   sim_step(iterations)                   — avanza la simulazione
 *   sim_get_state()   → ptr Float32        — stato particelle (vedi sotto)
 *   sim_get_diagnostics() → ptr Float32    — 8 diagnostici
 *   sim_get_phase()   → int                — fase corrente 0-4
 *   sim_get_N()       → int                — N particelle totali (vive+morte)
 *
 * Layout sim_get_state():
 *   [0] n_alive  [1] ns_active  [2] ns_x  [3] ns_y
 *   [4] ns_pulse [5] cmx        [6] cmy    [7] (riservato)
 *   Poi per ogni particella viva: px, py, u, rho  (4 float ciascuna)
 *
 * Layout sim_get_diagnostics():
 *   [0] max_rho  [1] max_u  [2] mean_rho  [3] sim_time
 *   [4] M_Jeans  [5] L_Jeans [6] n_alive  [7] phase
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

/* ── Parametri runtime ──────────────────────────────────────────────────── */
static float g_G       = 12.0f;
static float g_kpress  = 0.004f;
static float g_gamma   = 1.66f;
static float g_hsmooth = 20.0f;
static float g_soft    = 200.0f;
static float g_cool0   = 0.9980f;
static float g_cool1   = 0.9995f;
static float g_mass    = 1.0f;

static float g_thr01    = 0.15f;
static float g_thr12    = 0.60f;
static float g_thr23t   = 800.0f;
static float g_thr23rho = 3.0f;
static float g_thr34t   = 1100.0f;

/* ── Stato particelle ───────────────────────────────────────────────────── */
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

/* ── Kernel B-spline cubico 2D ──────────────────────────────────────────── */
static float kernel(float dist) {
    float h = g_hsmooth;
    float q = dist / h;
    if (q >= 2.0f) return 0.0f;
    float norm = 10.0f / (7.0f * M_PI * h * h);
    if (q <= 1.0f)
        return norm * (1.0f - 1.5f*q*q + 0.75f*q*q*q);
    return norm * 0.25f * powf(2.0f - q, 3.0f);
}

/* gradiente del kernel rispetto a (r_i - r_j), con dx = px[i]-px[j] */
static void kernel_grad(float dx, float dy, float dist, float *gx, float *gy) {
    float h = g_hsmooth;
    float q = dist / h;
    if (q >= 2.0f || dist < 1e-6f) { *gx = 0.0f; *gy = 0.0f; return; }
    float norm = 10.0f / (7.0f * M_PI * h * h * h);
    float dWdq = (q <= 1.0f)
        ? norm * (-3.0f*q + 2.25f*q*q)
        : norm * (-0.75f * powf(2.0f - q, 2.0f));
    *gx = dWdq * dx / dist;
    *gy = dWdq * dy / dist;
}

static void center_of_mass(float *cx, float *cy) {
    float sx = 0.0f, sy = 0.0f; int cnt = 0;
    for (int i = 0; i < N; i++)
        if (alive[i]) { sx += px[i]; sy += py[i]; cnt++; }
    if (cnt > 0) { *cx = sx/cnt; *cy = sy/cnt; }
    else         { *cx = CANVAS_W/2.0f; *cy = CANVAS_H/2.0f; }
}

/* ══════════════════════════════════════════════════════════════════════════
   API esportata verso JavaScript
   ══════════════════════════════════════════════════════════════════════════ */

/*
 * sim_set_params — imposta i parametri fisici runtime.
 *   g_grav : costante gravitazionale adimensionale (default 12)
 *   kpress : K nell'equazione di stato P = K*rho^gamma (default 0.004)
 *   cool   : fattore di raffreddamento radiativo per passo, fase 0 (default 0.998)
 *   soft   : softening gravitazionale in px² (default 200)
 *
 * Le soglie di transizione di fase vengono ri-scalate linearmente con G
 * rispetto al valore di default G=12, in modo che il collasso resti
 * visivamente calibrato qualunque sia G scelto dall'utente.
 */
EMSCRIPTEN_KEEPALIVE
void sim_set_params(float g_grav, float kpress, float cool, float soft) {
    if (g_grav > 0.0f) g_G      = g_grav;
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
    g_thr23t   = 800.0f  / gs;
    g_thr34t   = 1100.0f / gs;
}

/*
 * sim_init — inizializza la nube di particelle.
 *   n_req  : numero particelle richieste (clampato a N_MAX=1000)
 *   R      : raggio iniziale della nube [px]
 *   omega0 : velocità angolare iniziale [rad/passo] — rotazione rigida
 *   u0     : energia interna iniziale per particella (temperatura iniziale)
 *
 * Le particelle vengono distribuite uniformemente in area (r ∝ sqrt(rand))
 * con velocità tangenziale proporzionale al raggio (rotazione di corpo rigido).
 */
EMSCRIPTEN_KEEPALIVE
void sim_init(int n_req, float R, float omega0, float u0) {
    N         = (n_req > N_MAX) ? N_MAX : n_req;
    sim_time  = 0.0f;
    phase     = 0;
    ns_active = 0;
    ns_pulse  = 0.0f;
    diag_n_alive = N;

    float cx = CANVAS_W / 2.0f;
    float cy = CANVAS_H / 2.0f;

    for (int i = 0; i < N; i++) {
        float r     = R * sqrtf((float)rand() / (float)RAND_MAX);
        float theta = ((float)rand() / (float)RAND_MAX) * 2.0f * M_PI;
        px[i] = cx + r * cosf(theta);
        py[i] = cy + r * sinf(theta);
        vx[i] = -omega0 * (py[i] - cy);
        vy[i] =  omega0 * (px[i] - cx);
        u[i]     = u0;
        rho[i]   = 0.0f;
        p[i]     = 0.0f;
        ax[i]    = 0.0f;
        ay[i]    = 0.0f;
        alive[i] = 1;
    }
    for (int i = N; i < N_MAX; i++) alive[i] = 0;
}

/*
 * sim_step — avanza la simulazione di `iterations` passi di integrazione.
 *
 * Schema numerico: leapfrog (Störmer-Verlet) con dt=0.2 adimensionale.
 * Fisica implementata:
 *   - Densità SPH (Smoothed Particle Hydrodynamics) con kernel B-spline cubico W2D
 *   - Equazione di stato politropica:  P = K * rho^gamma
 *     + contributo termico (fase >= 2):  P += (gamma-1) * rho * u
 *   - Gravità con softening:  F_grav = G*m / (r² + epsilon²)
 *   - Forza di pressione SPH:  a_i -= m * (P_i/rho_i² + P_j/rho_j²) * grad_W
 *   - Viscosità artificiale di Monaghan (1992) per cattura degli urti:
 *       Pi_ij = (-alpha*c_ij*mu_ij + beta*mu_ij²) / rho_ij
 *       attiva solo se v_ij · r_ij < 0 (particelle che si avvicinano)
 *   - Raffreddamento radiativo: u *= cool_factor per passo
 *
 * Particle thinning (fase 4 — stella di neutroni):
 *   Ogni 50 passi, le particelle con rho < 0.003 e distanza dalla stella
 *   di neutroni > 1.2*CANVAS_W vengono disattivate, riducendo il costo
 *   O(N²) quando la stella è già formata e il gas diffuso è irrilevante.
 */
EMSCRIPTEN_KEEPALIVE
void sim_step(int iterations) {
    float dt   = 0.2f;
    float h2   = g_hsmooth * 2.0f;
    float h2sq = h2 * h2;

    for (int step = 0; step < iterations; step++) {

        /* ── Thinning in fase stabile ────────────────────────────────────── */
        if (phase == 4 && diag_n_alive > 80 && ((int)(sim_time/dt) % 50 == 0)) {
            float far2 = (CANVAS_W * 1.2f) * (CANVAS_W * 1.2f);
            for (int i = 0; i < N; i++) {
                if (!alive[i]) continue;
                float ddx = px[i] - ns_x, ddy = py[i] - ns_y;
                if (rho[i] < 0.003f && (ddx*ddx + ddy*ddy) > far2)
                    alive[i] = 0;
            }
        }

        /* ── 1. Densità e pressione ──────────────────────────────────────── */
        for (int i = 0; i < N; i++) {
            if (!alive[i]) continue;
            rho[i] = 0.01f;
            for (int j = 0; j < N; j++) {
                if (!alive[j]) continue;
                float dx = px[i]-px[j], dy = py[i]-py[j];
                float r2 = dx*dx + dy*dy;
                if (r2 < h2sq) rho[i] += g_mass * kernel(sqrtf(r2));
            }
            p[i] = g_kpress * powf(rho[i], g_gamma);
            if (phase >= 2)
                p[i] += (g_gamma - 1.0f) * rho[i] * u[i];
        }

        /* ── 2. Accelerazioni ────────────────────────────────────────────── */
        for (int i = 0; i < N; i++) {
            if (!alive[i]) continue;
            ax[i] = 0.0f; ay[i] = 0.0f;
            for (int j = 0; j < N; j++) {
                if (i == j || !alive[j]) continue;
                float dx   = px[j]-px[i], dy = py[j]-py[i];
                float r2   = dx*dx + dy*dy;
                float dist = sqrtf(r2);

                /* Gravità con softening */
                float fg = (g_G * g_mass) / (r2 + g_soft);
                ax[i] += fg * dx / dist;
                ay[i] += fg * dy / dist;

                /* Pressione + viscosità artificiale (solo nel raggio del kernel) */
                if (dist < h2) {
                    float gx, gy;
                    kernel_grad(-dx, -dy, dist, &gx, &gy);

                    float pi2 = p[i]/(rho[i]*rho[i]);
                    float pj2 = p[j]/(rho[j]*rho[j]);
                    float pc  = g_mass * (pi2 + pj2);
                    ax[i] -= pc * gx;
                    ay[i] -= pc * gy;

                    /* Viscosità artificiale di Monaghan */
                    float dvx = vx[i]-vx[j], dvy = vy[i]-vy[j];
                    float vdr = dvx*(-dx) + dvy*(-dy);
                    if (vdr < 0.0f) {
                        float rm  = 0.5f*(rho[i]+rho[j]);
                        float ci  = sqrtf(g_gamma*p[i]/rho[i]);
                        float cj  = sqrtf(g_gamma*p[j]/rho[j]);
                        float cm  = 0.5f*(ci+cj);
                        float mu  = g_hsmooth * vdr / (r2 + VISC_EPS*g_hsmooth*g_hsmooth);
                        float Piv = (-VISC_ALPHA*cm*mu + VISC_BETA*mu*mu) / rm;
                        ax[i] -= g_mass * Piv * gx;
                        ay[i] -= g_mass * Piv * gy;
                    }
                }
            }
        }

        /* ── 3. Integrazione (leapfrog) + energia termica ────────────────── */
        float max_rho = 0.0f;
        float sum_rho = 0.0f;
        float cmx, cmy;
        center_of_mass(&cmx, &cmy);
        diag_n_alive = 0;

        for (int i = 0; i < N; i++) {
            if (!alive[i]) continue;
            diag_n_alive++;
            vx[i] += ax[i]*dt; vy[i] += ay[i]*dt;
            px[i] += vx[i]*dt; py[i] += vy[i]*dt;
            /* riscaldamento adiabatico per compressione */
            u[i]  += (p[i]/rho[i]) * 0.001f;
            /* raffreddamento radiativo */
            u[i]  *= (phase == 0) ? g_cool0 : g_cool1;
            if (rho[i] > max_rho) max_rho = rho[i];
            sum_rho += rho[i];
        }

        /* ── 4. Transizioni di fase ──────────────────────────────────────── */
        if (phase == 0 && max_rho > g_thr01) phase = 1;
        if (phase == 1 && max_rho > g_thr12) phase = 2;
        if (phase == 2 && (sim_time > g_thr23t || max_rho > g_thr23rho)) phase = 3;

        /* Fase 3: rimbalzo — onda d'urto della supernova */
        if (phase == 3) {
            for (int i = 0; i < N; i++) {
                if (!alive[i]) continue;
                float dx = px[i]-cmx, dy = py[i]-cmy;
                float d  = sqrtf(dx*dx+dy*dy);
                if (d > 1e-3f && d < 40.0f) {
                    vx[i] += (dx/d)*10.0f;
                    vy[i] += (dy/d)*10.0f;
                    u[i]  += 3.0f;
                }
            }
            if (sim_time > g_thr34t) {
                phase = 4; ns_active = 1; ns_x = cmx; ns_y = cmy;
            }
        }

        /* Fase 4: stella di neutroni — assorbe le particelle vicine */
        if (ns_active) {
            ns_pulse += 0.3f;
            for (int i = 0; i < N; i++) {
                if (!alive[i]) continue;
                float dx = px[i]-ns_x, dy = py[i]-ns_y;
                if (dx*dx+dy*dy < 144.0f) alive[i] = 0;
            }
        }

        sim_time += dt;
    }
}

/* ── Funzioni di lettura stato ──────────────────────────────────────────── */

EMSCRIPTEN_KEEPALIVE
int sim_get_phase(void) { return phase; }

EMSCRIPTEN_KEEPALIVE
int sim_get_N(void) { return N; }

/*
 * sim_get_diagnostics — restituisce puntatore a buffer interno di 8 float:
 *   [0] max_rho    densità massima
 *   [1] max_u      energia interna massima
 *   [2] mean_rho   densità media
 *   [3] sim_time   tempo simulato
 *   [4] M_Jeans    massa di Jeans 2D (unità adimensionali)
 *   [5] L_Jeans    lunghezza di Jeans 2D [px]
 *   [6] n_alive    numero particelle attive
 *   [7] phase      fase corrente
 *
 * Criterio di Jeans 2D:
 *   c_s = sqrt(gamma * P_medio / rho_medio)   velocità del suono adiabatica
 *   L_J = c_s / sqrt(G * Sigma)               lunghezza di Jeans
 *   M_J = Sigma * L_J²                        massa di Jeans
 */
EMSCRIPTEN_KEEPALIVE
float* sim_get_diagnostics(void) {
    float mr=0, mu=0, sr=0, scs2=0;
    int cnt = 0;
    for (int i = 0; i < N; i++) {
        if (!alive[i]) continue;
        if (rho[i] > mr) mr = rho[i];
        if (u[i]   > mu) mu = u[i];
        sr   += rho[i];
        scs2 += g_gamma * p[i] / rho[i];
        cnt++;
    }
    float mean_rho = (cnt > 0) ? sr/cnt   : 0.0f;
    float mean_cs2 = (cnt > 0) ? scs2/cnt : 0.0f;
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

/*
 * sim_get_state — restituisce puntatore a buffer interno.
 * Layout:
 *   [0] n_alive   [1] ns_active  [2] ns_x   [3] ns_y
 *   [4] ns_pulse  [5] cmx        [6] cmy     [7] (riservato, 0)
 *   Poi per ogni particella viva (diag_n_alive voci da 4 float):
 *     px, py, u, rho
 */
EMSCRIPTEN_KEEPALIVE
float* sim_get_state(void) {
    float cmx, cmy;
    center_of_mass(&cmx, &cmy);
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
