/*
 * sim.c — SPH Stellar Collapse Simulation Core (Versione 1000 Particelle)
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <emscripten/emscripten.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

/* ── Costanti fisiche ───────────────────────────────────────────────────── */
#define N_MAX        1000   // Limite aumentato
#define G_GRAV       4.8f   // Gravità aumentata per vincere la pressione iniziale
#define SOFTENING    3.0f   
#define K_PRESS      0.004f // Ridotta per facilitare la contrazione
#define H_SMOOTH     20.0f
#define MASS         1.0f
#define GAMMA        1.66f  
#define CANVAS_W     760.0f
#define CANVAS_H     760.0f

/* ── Variabili Globali ──────────────────────────────────────────────────── */
int N = 0;
float px[N_MAX], py[N_MAX];
float vx[N_MAX], vy[N_MAX];
float ax[N_MAX], ay[N_MAX];
float rho[N_MAX], p[N_MAX], u[N_MAX];
int alive[N_MAX];

float sim_time = 0.0f;
int phase = 0; 
int ns_active = 0;
float ns_x, ns_y, ns_pulse = 0.0f;

/* Buffer di output calcolato correttamente per 1000 particelle */
/* Header (8 float) + (1000 particelle * 4 dati l'una) */
float state_buf[8 + (N_MAX * 4)];
float diag_buf[4];
int diag_n_alive = 0;

/* ── Funzioni Logiche ───────────────────────────────────────────────────── */

void center_of_mass(float *cx, float *cy) {
    float sx = 0, sy = 0;
    int count = 0;
    for (int i = 0; i < N; i++) {
        if (alive[i]) {
            sx += px[i]; sy += py[i]; count++;
        }
    }
    if (count > 0) { *cx = sx / count; *cy = sy / count; }
    else { *cx = CANVAS_W / 2; *cy = CANVAS_H / 2; }
}

float kernel(float dist) {
    float q = dist / H_SMOOTH;
    if (q >= 2.0f) return 0.0f;
    float norm = 10.0f / (7.0f * M_PI * H_SMOOTH * H_SMOOTH);
    if (q <= 1.0f) return norm * (1.0f - 1.5f * q * q + 0.75f * q * q * q);
    else return norm * 0.25f * powf(2.0f - q, 3.0f);
}

void kernel_grad(float dx, float dy, float dist, float *gx, float *gy) {
    float q = dist / H_SMOOTH;
    if (q >= 2.0f || dist < 1e-6f) { *gx = 0; *gy = 0; return; }
    float norm = 10.0f / (7.0f * M_PI * powf(H_SMOOTH, 3.0f));
    float factor = (q <= 1.0f) ? (norm * (-3.0f * q + 2.25f * q * q)) : (norm * -0.75f * powf(2.0f - q, 2.0f));
    *gx = factor * dx / dist;
    *gy = factor * dy / dist;
}

/* ── API ────────────────────────────────────────────────────────────────── */

EMSCRIPTEN_KEEPALIVE
void sim_init(int n_requested) {
    N = (n_requested > N_MAX) ? N_MAX : n_requested;
    sim_time = 0.0f;
    phase = 0;
    ns_active = 0;
    ns_pulse = 0.0f;

    for (int i = 0; i < N; i++) {
        float r = 180.0f * (float)sqrt((float)rand() / RAND_MAX); 
        float theta = ((float)rand() / RAND_MAX) * 2.0f * M_PI;
        px[i] = CANVAS_W / 2.0f + r * cosf(theta);
        py[i] = CANVAS_H / 2.0f + r * sinf(theta);
        
        float vt = 0.15f; 
        vx[i] = -vt * (py[i] - CANVAS_H / 2.0f) / 10.0f;
        vy[i] =  vt * (px[i] - CANVAS_W / 2.0f) / 10.0f;

        u[i] = 0.01f; // Molto fredda
        rho[i] = 0.0f;
        alive[i] = 1;
    }
}

EMSCRIPTEN_KEEPALIVE
void sim_step(int iterations) {
    float dt = 0.2f;
    for (int step = 0; step < iterations; step++) {
        // Densità
        for (int i = 0; i < N; i++) {
            if (!alive[i]) continue;
            rho[i] = 0.01f; 
            for (int j = 0; j < N; j++) {
                if (!alive[j]) continue;
                float dx = px[i] - px[j], dy = py[i] - py[j];
                float r2 = dx * dx + dy * dy;
                if (r2 < 4.0f * H_SMOOTH * H_SMOOTH) rho[i] += MASS * kernel(sqrtf(r2));
            }
            p[i] = K_PRESS * powf(rho[i], GAMMA);
            if (phase >= 2) p[i] *= (1.0f + u[i] * 8.0f);
        }

        // Forze
        for (int i = 0; i < N; i++) {
            if (!alive[i]) continue;
            ax[i] = 0; ay[i] = 0;
            for (int j = 0; j < N; j++) {
                if (i == j || !alive[j]) continue;
                float dx = px[j] - px[i], dy = py[j] - py[i];
                float r2 = dx * dx + dy * dy;
                float dist = sqrtf(r2);
                float force_g = (G_GRAV * MASS) / (r2 + SOFTENING);
                ax[i] += force_g * dx / dist;
                ay[i] += force_g * dy / dist;
                if (dist < H_SMOOTH * 2.0f) {
                    float gx, gy;
                    kernel_grad(-dx, -dy, dist, &gx, &gy);
                    float press_term = (p[i] / (rho[i] * rho[i]) + p[j] / (rho[j] * rho[j]));
                    ax[i] -= MASS * press_term * gx;
                    ay[i] -= MASS * press_term * gy;
                }
            }
        }

        // Integrazione
        float max_rho = 0;
        float cmx, cmy;
        center_of_mass(&cmx, &cmy);
        diag_n_alive = 0;

        for (int i = 0; i < N; i++) {
            if (!alive[i]) continue;
            diag_n_alive++;
            vx[i] += ax[i] * dt; vy[i] += ay[i] * dt;
            px[i] += vx[i] * dt; py[i] += vy[i] * dt;
            u[i] += rho[i] * 0.0002f; 
            if (phase == 0) u[i] *= 0.98f; else u[i] *= 0.996f;
            if (rho[i] > max_rho) max_rho = rho[i];
        }

        // Switch fasi
        if (phase == 0 && max_rho > 0.9f) phase = 1;
        if (phase == 1 && max_rho > 3.0f) phase = 2;
        if (phase == 2 && sim_time > 2000.0f) phase = 3;
        
        if (phase == 3) {
            for (int i = 0; i < N; i++) {
                if (!alive[i]) continue;
                float dx = px[i] - cmx, dy = py[i] - cmy;
                float d = sqrtf(dx*dx + dy*dy);
                if (d < 40.0f) { vx[i] += (dx/d)*10.0f; vy[i] += (dy/d)*10.0f; u[i] += 3.0f; }
            }
            if (sim_time > 2300.0f) { phase = 4; ns_active = 1; ns_x = cmx; ns_y = cmy; }
        }
        if (ns_active) {
            ns_pulse += 0.3f;
            for (int i = 0; i < N; i++) {
                if (alive[i] && powf(px[i]-ns_x,2)+powf(py[i]-ns_y,2) < 144.0f) alive[i] = 0;
            }
        }
        sim_time += dt;
    }
}

EMSCRIPTEN_KEEPALIVE
int sim_get_phase(void) { return phase; }

EMSCRIPTEN_KEEPALIVE
float* sim_get_diagnostics(void) {
    float mr = 0, mu = 0;
    for(int i=0; i<N; i++) if(alive[i]) { if(rho[i]>mr) mr=rho[i]; if(u[i]>mu) mu=u[i]; }
    float cx, cy; center_of_mass(&cx, &cy);
    diag_buf[0] = mr; diag_buf[1] = mu; diag_buf[2] = 50.0f; diag_buf[3] = sim_time;
    return diag_buf;
}

EMSCRIPTEN_KEEPALIVE
float* sim_get_state(void) {
    float cmx, cmy; center_of_mass(&cmx, &cmy);
    state_buf[0] = (float)diag_n_alive;
    state_buf[1] = (float)ns_active;
    state_buf[2] = ns_x; state_buf[3] = ns_y;
    state_buf[4] = ns_pulse;
    state_buf[5] = cmx; state_buf[6] = cmy;
    state_buf[7] = 0;
    int out = 8;
    for (int i = 0; i < N; i++) {
        if (alive[i]) {
            state_buf[out++] = px[i]; state_buf[out++] = py[i];
            state_buf[out++] = u[i];  state_buf[out++] = rho[i];
        }
    }
    return state_buf;
}