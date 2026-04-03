/*
 * sim.c — SPH Stellar Collapse Simulation Core (Versione 1000 Particelle)
 *
 * Correzioni rispetto alla versione precedente:
 *
 * 1. SOFTENING gravitazionale portato a 300.0 (era 3.0), consistente con
 *    la scala del kernel H_SMOOTH=20, evita instabilità numerica nelle
 *    coppie ravvicinate e forze di punto-materiale non fisiche.
 *
 * 2. Termine di energia interna u corretto: il riscaldamento adiabatico
 *    usa ora (p/rho²)*rho, proporzionale al lavoro di compressione, invece
 *    di rho nudo. Il raffreddamento radiativo è molto più lento (0.9995/
 *    0.9998 per passo invece di 0.98/0.996) così la pressione termica
 *    riesce a crescere effettivamente durante il collasso.
 *
 * 3. Soglie di transizione di fase abbassate a valori raggiungibili con i
 *    parametri geometrici reali della simulazione (densità iniziale ~0.05):
 *      fase 0->1: max_rho > 0.25  (era 0.9)
 *      fase 1->2: max_rho > 1.0   (era 3.0)
 *    La transizione 2->3 è anticipata a t>1500 (era 2000) e resa
 *    indipendente dal tempo anche se max_rho supera una soglia estrema,
 *    così la supernova scatta comunque.
 *
 * 4. K_PRESS leggermente aumentato (0.008, era 0.004) per garantire che
 *    la pressione termica sia apprezzabile nelle fasi finali del collasso
 *    e produca un rimbalzo visibile al centro prima della supernova.
 *
 * 5. Viscosità artificiale di Monaghan aggiunta nel calcolo delle forze:
 *    smorza le oscillazioni numeriche nelle zone ad alta compressione
 *    (particelle che si avvicinano, v_ij · r_ij < 0) e previene la
 *    penetrazione non fisica dei gusci di gas. Parametri: alpha=1.0,
 *    beta=2.0, epsilon=0.01 (standard nella letteratura SPH).
 *
 * 6. Distribuzione iniziale invariata (uniforme in area, fisicamente
 *    corretta per una nube molecolare).
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <emscripten/emscripten.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

/* ── Costanti fisiche ───────────────────────────────────────────────────── */
/* MODALITÀ DEBUG: parametri forzati per collasso rapido (~30s di sim).
   G molto alta, pressione quasi nulla, raffreddamento istantaneo,
   soglie di fase bassissime. Ripristinare i valori fisici dopo il test. */
#define N_MAX        1000
#define G_GRAV       25.0f    /* [DEBUG] era 4.8 — gravità molto aumentata      */
#define SOFTENING    150.0f   /* [DEBUG] era 300 — softening ridotto a metà      */
#define K_PRESS      0.001f   /* [DEBUG] era 0.008 — pressione quasi soppressa   */
#define H_SMOOTH     20.0f
#define MASS         1.0f
#define GAMMA        1.66f
#define CANVAS_W     760.0f
#define CANVAS_H     760.0f

/* Viscosità artificiale di Monaghan (alpha, beta, epsilon) */
#define VISC_ALPHA   1.0f
#define VISC_BETA    2.0f
#define VISC_EPS     0.01f

/* ── Variabili Globali ──────────────────────────────────────────────────── */
int N = 0;
float px[N_MAX], py[N_MAX];
float vx[N_MAX], vy[N_MAX];
float ax[N_MAX], ay[N_MAX];
float rho[N_MAX], p[N_MAX], u[N_MAX];
int alive[N_MAX];

float sim_time  = 0.0f;
int   phase     = 0;
int   ns_active = 0;
float ns_x, ns_y, ns_pulse = 0.0f;

float state_buf[8 + (N_MAX * 4)];
float diag_buf[4];
int   diag_n_alive = 0;

/* ── Funzioni di supporto ───────────────────────────────────────────────── */

void center_of_mass(float *cx, float *cy) {
    float sx = 0.0f, sy = 0.0f;
    int count = 0;
    for (int i = 0; i < N; i++) {
        if (alive[i]) { sx += px[i]; sy += py[i]; count++; }
    }
    if (count > 0) { *cx = sx / count; *cy = sy / count; }
    else           { *cx = CANVAS_W / 2.0f; *cy = CANVAS_H / 2.0f; }
}

/*
 * Kernel B-spline cubico (Monaghan 1992) in 2D.
 * W(r, h) con supporto compatto su [0, 2h].
 * Normalizzazione: 10/(7 pi h^2).
 */
float kernel(float dist) {
    float q    = dist / H_SMOOTH;
    if (q >= 2.0f) return 0.0f;
    float norm = 10.0f / (7.0f * M_PI * H_SMOOTH * H_SMOOTH);
    if (q <= 1.0f)
        return norm * (1.0f - 1.5f * q * q + 0.75f * q * q * q);
    else
        return norm * 0.25f * powf(2.0f - q, 3.0f);
}

/*
 * Gradiente del kernel rispetto alla posizione della particella i,
 * con r_ij = r_i - r_j  =>  dx = px[i]-px[j], dy = py[i]-py[j].
 * dW/dr_i = (dW/dq)(1/h) * (r_ij / |r_ij|)
 * Normalizzazione della derivata: 10/(7 pi h^3).
 */
void kernel_grad(float dx, float dy, float dist, float *gx, float *gy) {
    float q = dist / H_SMOOTH;
    if (q >= 2.0f || dist < 1e-6f) { *gx = 0.0f; *gy = 0.0f; return; }
    float norm   = 10.0f / (7.0f * M_PI * H_SMOOTH * H_SMOOTH * H_SMOOTH);
    float dWdq;
    if (q <= 1.0f)
        dWdq = norm * (-3.0f * q + 2.25f * q * q);
    else
        dWdq = norm * (-0.75f * powf(2.0f - q, 2.0f));
    /* dW/dr_i = dWdq * (r_ij / (|r_ij| * h)), ma q = dist/h quindi
       dW/dr_i = (dWdq / h) * (r_ij / dist) = dWdq * (r_ij / dist) / h.
       Il fattore 1/h è già incorporato in norm (denominatore h^3 vs h^2). */
    *gx = dWdq * dx / dist;
    *gy = dWdq * dy / dist;
}

/* ── API Emscripten ─────────────────────────────────────────────────────── */

EMSCRIPTEN_KEEPALIVE
void sim_init(int n_requested) {
    N = (n_requested > N_MAX) ? N_MAX : n_requested;
    sim_time  = 0.0f;
    phase     = 0;
    ns_active = 0;
    ns_pulse  = 0.0f;

    for (int i = 0; i < N; i++) {
        /* Distribuzione uniforme in area (densità superficiale costante):
           r ~ sqrt(U[0,1]) * R_max  con R_max = 180 px               */
        float r     = 180.0f * sqrtf((float)rand() / (float)RAND_MAX);
        float theta = ((float)rand() / (float)RAND_MAX) * 2.0f * M_PI;
        px[i] = CANVAS_W / 2.0f + r * cosf(theta);
        py[i] = CANVAS_H / 2.0f + r * sinf(theta);

        /* [DEBUG] nessuna rotazione: cade dritto verso il centro */
        vx[i] = 0.0f;
        vy[i] = 0.0f;

        u[i]    = 0.0f; /* [DEBUG] fredda morta, nessuna pressione termica iniziale */
        rho[i]  = 0.0f;
        alive[i] = 1;
    }
}

EMSCRIPTEN_KEEPALIVE
void sim_step(int iterations) {
    float dt = 0.2f;

    for (int step = 0; step < iterations; step++) {

        /* ── 1. Calcolo densità e pressione ─────────────────────────── */
        for (int i = 0; i < N; i++) {
            if (!alive[i]) continue;
            rho[i] = 0.01f; /* densità minima di fondo (evita divisione per zero) */
            for (int j = 0; j < N; j++) {
                if (!alive[j]) continue;
                float dx = px[i] - px[j], dy = py[i] - py[j];
                float r2 = dx * dx + dy * dy;
                if (r2 < 4.0f * H_SMOOTH * H_SMOOTH)
                    rho[i] += MASS * kernel(sqrtf(r2));
            }
            /* Equazione di stato politropica: P = K * rho^gamma */
            p[i] = K_PRESS * powf(rho[i], GAMMA);

            /* Nelle fasi avanzate includiamo il contributo dell'energia
               termica alla pressione (gas ideale: P += (gamma-1)*rho*u) */
            if (phase >= 2)
                p[i] += (GAMMA - 1.0f) * rho[i] * u[i];
        }

        /* ── 2. Calcolo accelerazioni (gravità + pressione + viscosità) */
        for (int i = 0; i < N; i++) {
            if (!alive[i]) continue;
            ax[i] = 0.0f; ay[i] = 0.0f;

            for (int j = 0; j < N; j++) {
                if (i == j || !alive[j]) continue;

                /* vettore r_ij = r_j - r_i  (direzione da i verso j) */
                float dx   = px[j] - px[i];
                float dy   = py[j] - py[i];
                float r2   = dx * dx + dy * dy;
                float dist = sqrtf(r2);

                /* ── Gravità con softening corretto ─────────────────── */
                float force_g = (G_GRAV * MASS) / (r2 + SOFTENING);
                ax[i] += force_g * dx / dist;
                ay[i] += force_g * dy / dist;

                /* ── Pressione + viscosità (solo nel supporto del kernel) */
                if (dist < 2.0f * H_SMOOTH) {
                    /* gradiente rispetto a r_i: vettore r_i - r_j = -dx,-dy */
                    float gx, gy;
                    kernel_grad(-dx, -dy, dist, &gx, &gy);

                    /* Formula simmetrizzata di Monaghan per la pressione:
                       a_i^press = -m_j * (P_i/rho_i^2 + P_j/rho_j^2) * grad W_ij */
                    float pi_rho2 = p[i] / (rho[i] * rho[i]);
                    float pj_rho2 = p[j] / (rho[j] * rho[j]);
                    float press_coeff = MASS * (pi_rho2 + pj_rho2);
                    ax[i] -= press_coeff * gx;
                    ay[i] -= press_coeff * gy;

                    /* ── Viscosità artificiale di Monaghan ──────────── */
                    /* v_ij · r_ij */
                    float dvx  = vx[i] - vx[j];
                    float dvy  = vy[i] - vy[j];
                    float vdotr = dvx * (-dx) + dvy * (-dy); /* v_ij · (r_i-r_j) */

                    if (vdotr < 0.0f) { /* solo per particelle che si avvicinano */
                        float rho_mean = 0.5f * (rho[i] + rho[j]);
                        /* velocità del suono locale: c = sqrt(gamma*P/rho) */
                        float ci = sqrtf(GAMMA * p[i] / rho[i]);
                        float cj = sqrtf(GAMMA * p[j] / rho[j]);
                        float c_mean = 0.5f * (ci + cj);
                        /* mu_ij = h * v_ij·r_ij / (|r_ij|^2 + eps*h^2) */
                        float mu  = H_SMOOTH * vdotr / (dist * dist + VISC_EPS * H_SMOOTH * H_SMOOTH);
                        float Pi_visc = (-VISC_ALPHA * c_mean * mu + VISC_BETA * mu * mu) / rho_mean;
                        ax[i] -= MASS * Pi_visc * gx;
                        ay[i] -= MASS * Pi_visc * gy;
                    }
                }
            }
        }

        /* ── 3. Integrazione (Euler semi-implicito) e aggiornamento u ── */
        float max_rho = 0.0f;
        float cmx, cmy;
        center_of_mass(&cmx, &cmy);
        diag_n_alive = 0;

        for (int i = 0; i < N; i++) {
            if (!alive[i]) continue;
            diag_n_alive++;

            vx[i] += ax[i] * dt;
            vy[i] += ay[i] * dt;
            px[i] += vx[i] * dt;
            py[i] += vy[i] * dt;

            /*
             * Energia interna: riscaldamento adiabatico proporzionale al
             * lavoro di compressione (P/rho^2)*rho = P/rho, più lento ma
             * fisicamente motivato. Raffreddamento radiativo molto più
             * graduale così la pressione termica riesce a crescere.
             */
            float du_heat = (p[i] / rho[i]) * 0.001f;
            u[i] += du_heat;
            if (phase == 0)
                u[i] *= 0.95f;    /* [DEBUG] raffreddamento aggressivo in fase 0 */
            else
                u[i] *= 0.999f;

            if (rho[i] > max_rho) max_rho = rho[i];
        }

        /* ── 4. Transizioni di fase ─────────────────────────────────── */
        /*
         * Soglie abbassate ai valori raggiungibili con la densità iniziale
         * tipica (~0.05 con N=1000 in raggio 180 px):
         *   fase 0->1 (protostella): max_rho > 0.25
         *   fase 1->2 (collasso nucleare): max_rho > 1.0
         *   fase 2->3 (rimbalzo / bounce): t > 1500 oppure max_rho > 5
         *   fase 3->4 (stella di neutroni): t > 1800
         */
        /* [DEBUG] soglie abbassate al minimo per forzare la progressione */
        if (phase == 0 && max_rho > 0.08f) phase = 1;
        if (phase == 1 && max_rho > 0.25f) phase = 2;
        if (phase == 2 && (sim_time > 400.0f || max_rho > 1.5f)) phase = 3;

        if (phase == 3) {
            /* Rimbalzo: impulso repulsivo verso l'esterno per le particelle
               entro 40 px dal centro di massa (onda d'urto della supernova) */
            for (int i = 0; i < N; i++) {
                if (!alive[i]) continue;
                float dx = px[i] - cmx, dy = py[i] - cmy;
                float d  = sqrtf(dx * dx + dy * dy);
                if (d > 1e-3f && d < 40.0f) {
                    vx[i] += (dx / d) * 10.0f;
                    vy[i] += (dy / d) * 10.0f;
                    u[i]  += 3.0f;
                }
            }
            if (sim_time > 600.0f) {  /* [DEBUG] era 1800 */
                phase     = 4;
                ns_active = 1;
                ns_x      = cmx;
                ns_y      = cmy;
            }
        }

        /* ── 5. Stella di neutroni: assorbe particelle nel raggio 12 px */
        if (ns_active) {
            ns_pulse += 0.3f;
            for (int i = 0; i < N; i++) {
                if (alive[i]) {
                    float dx2 = px[i] - ns_x, dy2 = py[i] - ns_y;
                    if (dx2 * dx2 + dy2 * dy2 < 144.0f) alive[i] = 0;
                }
            }
        }

        sim_time += dt;
    }
}

/* ── Funzioni di lettura stato ──────────────────────────────────────────── */

EMSCRIPTEN_KEEPALIVE
int sim_get_phase(void) { return phase; }

EMSCRIPTEN_KEEPALIVE
float* sim_get_diagnostics(void) {
    float mr = 0.0f, mu = 0.0f;
    for (int i = 0; i < N; i++) {
        if (alive[i]) {
            if (rho[i] > mr) mr = rho[i];
            if (u[i]   > mu) mu = u[i];
        }
    }
    diag_buf[0] = mr;
    diag_buf[1] = mu;
    diag_buf[2] = 50.0f;   /* placeholder: potrebbe ospitare T_medio o E_tot */
    diag_buf[3] = sim_time;
    return diag_buf;
}

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
