/*
 * sim.c — SPH Stellar Collapse Simulation Core
 *
 * Rispetto alla versione precedente:
 *   - Tutti i parametri fisici sono ora variabili runtime (non #define),
 *     impostabili da JavaScript tramite sim_set_params() prima di sim_init().
 *   - sim_init() accetta raggio iniziale, velocità angolare, energia interna.
 *   - Particle thinning in fase stabile (phase==4): le particelle lontane
 *     dal centro e con bassa densità vengono disattivate per ridurre il
 *     costo O(N²) quando la stella è già formata.
 *   - sim_get_diagnostics() restituisce 8 float includendo massa e
 *     lunghezza di Jeans 2D.
 *   - Parametri di default calibrati per collasso visibile in ~60s reali.
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

/* gradiente rispetto a r_i - r_j (dx = px[i]-px[j]) */
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
   API Emscripten
   ══════════════════════════════════════════════════════════════════════════ */

/*
 * sim_set_params
 *   g_grav  : costante gravitazionale adimensionale
 *   kpress  : K nell'equazione di stato P = K*rho^gamma
 *   cool    : fattore di raffreddamento per passo in fase 0 (es. 0.998)
 *   soft    : softening gravitazionale [px^2]
 *
 * Le soglie di transizione vengono ri-scalate proporzionalmente a G
 * rispetto al valore di default G=12.
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
 * sim_init
 *   n_req   : numero di particelle richieste (clampato a N_MAX)
 *   R       : raggio iniziale della nube [px]
 *   omega0  : velocità angolare iniziale [rad/passo]
 *   u0      : energia interna iniziale per particella
 */
EMSCRIPTEN_KEEPALIVE
void sim_init(int n_req, float R, float omega0, float u0) {
    N         = (n_req > N_MAX) ? N_MAX : n_req;
    sim_time  = 0.0f;
    phase     = 0;
    ns_active = 0;
    ns_pulse  = 0.0f;

    float cx = CANVAS_W / 2.0f;
    float cy = CANVAS_H / 2.0f;

    for (int i = 0; i < N; i++) {
        float r     = R * sqrtf((float)rand() / (float)RAND_MAX);
        float theta = ((float)rand() / (float)RAND_MAX) * 2.0f * M_PI;
        px[i] = cx + r * cosf(theta);
        py[i] = cy + r * sinf(theta);

        /* rotazione rigida: v_tang = omega0 * r, perpendicolare al raggio */
        vx[i] = -omega0 * (py[i] - cy);
        vy[i] =  omega0 * (px[i] - cx);

        u[i]     = u0;
        rho[i]   = 0.0f;
        alive[i] = 1;
    }
}

/*
 * sim_step — avanza di `iterations` passi.
 *
 * Particle thinning (phase==4):
 *   Le particelle che soddisfano entrambe le condizioni
 *     rho < 0.003  AND  dist_from_ns > 1.2 * CANVAS_W
 *   vengono disattivate ogni 50 passi, riducendo N_eff e quindi il
 *   costo quadratico del loop densità+forze. Il nucleo compatto
 *   (stella di neutroni) viene preservato dal criterio di distanza.
 */
EMSCRIPTEN_KEEPALIVE
void sim_step(int iterations) {
    float dt   = 0.2f;
    float h2   = g_hsmooth * 2.0f;
    float h2sq = h2 * h2;

    for (int step = 0; step < iterations; step++) {

        /* ── Thinning in fase stabile ────────────────────────────────── */
        if (phase == 4 && diag_n_alive > 80 && ((int)(sim_time/dt) % 50 == 0)) {
            float far2 = (CANVAS_W * 1.2f) * (CANVAS_W * 1.2f);
            for (int i = 0; i < N; i++) {
                if (!alive[i]) continue;
                float ddx = px[i] - ns_x, ddy = py[i] - ns_y;
                if (rho[i] < 0.003f && (ddx*ddx + ddy*ddy) > far2)
                    alive[i] = 0;
            }
        }

        /* ── 1. Densità e pressione ───────────────────────────────────── */
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

        /* ── 2. Forze ─────────────────────────────────────────────────── */
        for (int i = 0; i < N; i++) {
            if (!alive[i]) continue;
            ax[i] = 0.0f; ay[i] = 0.0f;
            for (int j = 0; j < N; j++) {
                if (i == j || !alive[j]) continue;
                float dx   = px[j]-px[i], dy = py[j]-py[i];
                float r2   = dx*dx + dy*dy;
                float dist = sqrtf(r2);

                float fg = (g_G * g_mass) / (r2 + g_soft);
                ax[i] += fg * dx / dist;
                ay[i] += fg * dy / dist;

                if (dist < h2) {
                    float gx, gy;
                    kernel_grad(-dx, -dy, dist, &gx, &gy);

                    float pi2 = p[i]/(rho[i]*rho[i]);
                    float pj2 = p[j]/(rho[j]*rho[j]);
                    float pc  = g_mass * (pi2 + pj2);
                    ax[i] -= pc * gx;
                    ay[i] -= pc * gy;

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

        /* ── 3. Integrazione ed energia termica ───────────────────────── */
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
            u[i]  += (p[i]/rho[i]) * 0.001f;
            u[i]  *= (phase == 0) ? g_cool0 : g_cool1;
            if (rho[i] > max_rho) max_rho = rho[i];
            sum_rho += rho[i];
        }

        /* ── 4. Transizioni di fase ───────────────────────────────────── */
        if (phase == 0 && max_rho > g_thr01) phase = 1;
        if (phase == 1 && max_rho > g_thr12) phase = 2;
        if (phase == 2 && (sim_time > g_thr23t || max_rho > g_thr23rho)) phase = 3;

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

/* ── Output ─────────────────────────────────────────────────────────────── */

EMSCRIPTEN_KEEPALIVE
int sim_get_phase(void) { return phase; }

/*
 * sim_get_diagnostics — 8 float:
 *  [0] max_rho    [1] max_u    [2] mean_rho    [3] sim_time
 *  [4] M_Jeans    [5] L_Jeans  [6] n_alive     [7] phase
 *
 * Criterio di Jeans 2D (instabilità gravitazionale in un disco):
 *   c_s  = sqrt(gamma * P_medio / rho_medio)   velocità del suono
 *   Sigma = rho_medio                           densità superficiale effettiva
 *   L_J  = c_s / sqrt(G * Sigma)
 *   M_J  = Sigma * L_J^2
 * La nube collassa quando la sua dimensione supera L_J
 * (o la sua massa supera M_J).
 */
EMSCRIPTEN_KEEPALIVE
float* sim_get_diagnostics(void) {
    float mr=0,mu=0,sr=0,scs2=0;
    int cnt=0;
    for (int i=0;i<N;i++) {
        if (!alive[i]) continue;
        if (rho[i]>mr) mr=rho[i];
        if (u[i]>mu)   mu=u[i];
        sr   += rho[i];
        scs2 += g_gamma * p[i] / rho[i];
        cnt++;
    }
    float mean_rho = (cnt>0) ? sr/cnt  : 0.0f;
    float mean_cs2 = (cnt>0) ? scs2/cnt: 0.0f;
    float Sigma    = mean_rho;
    float L_J=0.0f, M_J=0.0f;
    if (g_G>0.0f && Sigma>1e-6f) {
        L_J = sqrtf(mean_cs2 / (g_G * Sigma));
        M_J = Sigma * L_J * L_J;
    }
    diag_buf[0]=(float)mr;
    diag_buf[1]=(float)mu;
    diag_buf[2]=(float)mean_rho;
    diag_buf[3]=sim_time;
    diag_buf[4]=M_J;
    diag_buf[5]=L_J;
    diag_buf[6]=(float)diag_n_alive;
    diag_buf[7]=(float)phase;
    return diag_buf;
}

EMSCRIPTEN_KEEPALIVE
float* sim_get_state(void) {
    float cmx,cmy;
    center_of_mass(&cmx,&cmy);
    state_buf[0]=(float)diag_n_alive;
    state_buf[1]=(float)ns_active;
    state_buf[2]=ns_x; state_buf[3]=ns_y;
    state_buf[4]=ns_pulse;
    state_buf[5]=cmx;  state_buf[6]=cmy;
    state_buf[7]=0.0f;
    int out=8;
    for (int i=0;i<N;i++) {
        if (alive[i]) {
            state_buf[out++]=px[i]; state_buf[out++]=py[i];
            state_buf[out++]=u[i];  state_buf[out++]=rho[i];
        }
    }
    return state_buf;
}
