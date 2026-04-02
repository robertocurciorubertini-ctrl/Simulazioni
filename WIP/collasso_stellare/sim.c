/*
 * sim.c — SPH Stellar Collapse Simulation Core
 *
 * Compilare con Emscripten:
 *   emcc sim.c -o sim.js \
 *     -O2 \
 *     -s WASM=1 \
 *     -s EXPORTED_FUNCTIONS="['_sim_init','_sim_step','_sim_get_state','_sim_get_phase','_sim_get_diagnostics']" \
 *     -s EXPORTED_RUNTIME_METHODS="['ccall','cwrap']" \
 *     -s ALLOW_MEMORY_GROWTH=1 \
 *     -s MODULARIZE=1 \
 *     -s EXPORT_NAME="SimModule"
 *
 * Fisica:
 *   - SPH (Smoothed Particle Hydrodynamics) con kernel cubico di Monaghan
 *   - Gravità Barnes-Hut O(N log N) con stack iterativo
 *   - Equazione di stato politropica P = K * rho^gamma
 *   - Viscosità artificiale di Monaghan (1992)
 *   - Integratore leapfrog simplettico con timestep adattivo
 *   - Raffreddamento radiativo nella fase di nube molecolare
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <emscripten/emscripten.h>

/* ── Costanti fisiche ───────────────────────────────────────────────────── */
#define N_MAX        600
#define G_GRAV       2.2f
#define SOFTENING    4.0f
#define K_PRESS      0.010f
#define H_SMOOTH     20.0f
#define ALPHA_VISC   1.0f
#define BETA_VISC    2.0f
#define CLOUD_R      130.0f
#define OMEGA_INIT   0.009f
#define DT_BASE      0.08f
#define BH_THETA     0.75f
#define CX           380.0f
#define CY           380.0f
#define CANVAS_W     760.0f
#define CANVAS_H     760.0f

/* Cooling rates */
#define COOL_CLOUD   0.9910f   /* raffreddamento aggressivo fase 0: favorisce collasso */
#define COOL_STABLE  0.9996f   /* raffreddamento lento fase 2: stella stabile */

/* Soglie di transizione di fase */
#define RHO_PROTO    0.55f     /* densità nucleo → proto-stella */
#define RMS_STABLE   55.0f     /* raggio RMS → stella stabile */
#define T_SUPERNOVA  280       /* frame in fase stabile → supernova */
#define T_NS         65        /* frame dopo supernova → stella di neutroni */

/* ── Strutture dati (SoA — Structure of Arrays, cache-friendly) ─────────── */
static int   N = 0;
static float px[N_MAX], py[N_MAX];       /* posizioni */
static float vx[N_MAX], vy[N_MAX];       /* velocità */
static float ax[N_MAX], ay[N_MAX];       /* accelerazioni */
static float rho[N_MAX];                 /* densità SPH */
static float pres[N_MAX];                /* pressione */
static float u[N_MAX];                   /* energia interna (proxy temperatura) */
static float mass[N_MAX];               /* massa */
static int   alive[N_MAX];              /* 1 = viva, 0 = rimossa */

/* Stato globale */
static int   phase       = 0;
static int   phase_timer = 0;
static float sim_time    = 0.0f;
static int   sn_triggered = 0;
static int   ns_formed    = 0;

/* Stella di neutroni */
static float ns_x = 0, ns_y = 0;
static float ns_vx = 0, ns_vy = 0;
static float ns_pulse = 0.0f;
static int   ns_active = 0;

/* Buffer di output condiviso con JavaScript (posizioni + colori) */
/* Layout: [x0,y0,u0,rho0, x1,y1,u1,rho1, ...] float32 */
#define STATE_STRIDE 4
static float state_buf[N_MAX * STATE_STRIDE + 8]; /* +8 per neutron star */

/* Diagnostica */
static float diag_max_rho  = 0;
static float diag_max_u    = 0;
static float diag_rms      = 0;
static float diag_energy   = 0;
static int   diag_n_alive  = 0;

/* ── Barnes-Hut quadtree (allocazione su pool statico) ───────────────────── */
#define BH_POOL_SIZE 32768

typedef struct {
    float cx, cy, size;
    float mass;
    float cmx, cmy;
    int   children[4];   /* indici nel pool, -1 = assente */
    int   particle;      /* indice particella foglia, -1 = nessuna */
} BHNode;

static BHNode bh_pool[BH_POOL_SIZE];
static int    bh_pool_top = 0;

static int bh_alloc(float cx, float cy, float size) {
    if (bh_pool_top >= BH_POOL_SIZE) return -1;
    int idx = bh_pool_top++;
    BHNode *n = &bh_pool[idx];
    n->cx = cx; n->cy = cy; n->size = size;
    n->mass = 0; n->cmx = 0; n->cmy = 0;
    n->children[0] = n->children[1] = n->children[2] = n->children[3] = -1;
    n->particle = -1;
    return idx;
}

static void bh_insert(int node_idx, int p_idx) {
    if (node_idx < 0) return;
    BHNode *node = &bh_pool[node_idx];

    /* nodo vuoto → diventa foglia */
    if (node->mass == 0.0f && node->children[0] < 0) {
        node->particle = p_idx;
        node->mass = mass[p_idx];
        node->cmx  = px[p_idx];
        node->cmy  = py[p_idx];
        return;
    }

    /* foglia occupata → suddividi */
    if (node->particle >= 0 && node->children[0] < 0) {
        float h = node->size * 0.5f;
        for (int q = 0; q < 4; q++) {
            float ocx = node->cx + ((q & 1) ? +h*0.5f : -h*0.5f);
            float ocy = node->cy + ((q & 2) ? +h*0.5f : -h*0.5f);
            node->children[q] = bh_alloc(ocx, ocy, h);
        }
        int op = node->particle;
        node->particle = -1;
        int qi = (px[op] > node->cx ? 1 : 0) | (py[op] > node->cy ? 2 : 0);
        bh_insert(node->children[qi], op);
    }

    /* aggiorna centro di massa */
    float tm = node->mass + mass[p_idx];
    node->cmx = (node->cmx * node->mass + px[p_idx] * mass[p_idx]) / tm;
    node->cmy = (node->cmy * node->mass + py[p_idx] * mass[p_idx]) / tm;
    node->mass = tm;

    /* inserisci nel figlio corretto */
    int qi = (px[p_idx] > node->cx ? 1 : 0) | (py[p_idx] > node->cy ? 2 : 0);
    if (node->children[qi] >= 0)
        bh_insert(node->children[qi], p_idx);
}

/* Stack iterativo per il calcolo della forza — evita stack overflow */
#define BH_STACK_SIZE 4096
static int bh_stack[BH_STACK_SIZE];

static void bh_force(int root, int p_idx, float *out_ax, float *out_ay) {
    *out_ax = 0; *out_ay = 0;
    if (root < 0) return;

    int top = 0;
    bh_stack[top++] = root;

    while (top > 0) {
        int ni = bh_stack[--top];
        if (ni < 0) continue;
        BHNode *node = &bh_pool[ni];
        if (node->mass == 0.0f) continue;
        if (node->particle == p_idx) continue;

        float dx = node->cmx - px[p_idx];
        float dy = node->cmy - py[p_idx];
        float r2 = dx*dx + dy*dy + SOFTENING*SOFTENING;
        float r  = sqrtf(r2);

        if (node->size / r < BH_THETA || node->children[0] < 0) {
            /* approssimazione multipolo o foglia */
            if (node->particle == p_idx) continue;
            float f = G_GRAV * node->mass / r2;
            *out_ax += f * dx / r;
            *out_ay += f * dy / r;
        } else {
            /* espandi i figli */
            for (int q = 0; q < 4; q++) {
                if (node->children[q] >= 0 && top < BH_STACK_SIZE - 1)
                    bh_stack[top++] = node->children[q];
            }
        }
    }
}

/* ── Kernel SPH — spline cubica (Monaghan 1992) ──────────────────────────── */
static const float KNORM = 10.0f / (7.0f * 3.14159265f * H_SMOOTH * H_SMOOTH);

static float W_sph(float r) {
    float q = r / H_SMOOTH;
    if (q > 2.0f) return 0.0f;
    if (q > 1.0f) {
        float d = 2.0f - q;
        return KNORM * 0.25f * d*d*d;
    }
    return KNORM * (1.0f - 1.5f*q*q + 0.75f*q*q*q);
}

static float dW_sph(float r) {
    if (r < 1e-6f) return 0.0f;
    float q = r / H_SMOOTH;
    if (q > 2.0f) return 0.0f;
    if (q > 1.0f) return KNORM * (-0.75f * (2.0f-q)*(2.0f-q)) / H_SMOOTH;
    return KNORM * (-3.0f*q + 2.25f*q*q) / H_SMOOTH;
}

/* ── Equazione di stato ──────────────────────────────────────────────────── */
static float gamma_eos(void) {
    if (phase >= 3) return 1.28f;   /* collasso finale: EOS morbida */
    if (phase >= 2) return 1.667f;  /* stella stabile: adiabatica */
    if (phase >= 1) return 1.10f;   /* proto-stella */
    return 1.0f;                     /* nube: isoterma P ∝ ρ */
}

/* ── Calcolo densità e pressione SPH ────────────────────────────────────── */
static void compute_density(void) {
    float gam = gamma_eos();
    for (int i = 0; i < N; i++) {
        if (!alive[i]) continue;
        float r_i = 0.0f;
        for (int j = 0; j < N; j++) {
            if (!alive[j]) continue;
            float dx = px[i]-px[j], dy = py[i]-py[j];
            float r  = sqrtf(dx*dx + dy*dy);
            r_i += mass[j] * W_sph(r);
        }
        rho[i]  = (r_i < 1e-6f) ? 1e-6f : r_i;
        pres[i] = K_PRESS * powf(rho[i], gam);
    }
}

/* ── Calcolo accelerazioni SPH + viscosità artificiale + gravità BH ──────── */
static void compute_accel(int bh_root) {
    float gam = gamma_eos();
    float h2  = (H_SMOOTH * 2.0f) * (H_SMOOTH * 2.0f);

    for (int i = 0; i < N; i++) {
        if (!alive[i]) { ax[i] = ay[i] = 0; continue; }

        float axi = 0, ayi = 0;

        /* gravità Barnes-Hut */
        bh_force(bh_root, i, &axi, &ayi);

        /* gravità stella di neutroni */
        if (ns_active && phase == 4) {
            float dx = ns_x - px[i], dy = ns_y - py[i];
            float r2 = dx*dx + dy*dy + (SOFTENING*3)*(SOFTENING*3);
            float r  = sqrtf(r2);
            float f  = G_GRAV * 80.0f / r2;
            axi += f * dx/r;
            ayi += f * dy/r;
        }

        /* pressione SPH + viscosità artificiale */
        for (int j = 0; j < N; j++) {
            if (!alive[j] || j == i) continue;
            float dx = px[i]-px[j], dy = py[i]-py[j];
            float r2 = dx*dx + dy*dy;
            if (r2 > h2 || r2 < 1e-8f) continue;
            float r   = sqrtf(r2);
            float dw  = dW_sph(r);
            float gWx = dw * dx / r;
            float gWy = dw * dy / r;

            float pterm = pres[i]/(rho[i]*rho[i]) + pres[j]/(rho[j]*rho[j]);

            /* viscosità artificiale Monaghan */
            float vdr = (vx[i]-vx[j])*dx + (vy[i]-vy[j])*dy;
            float visc = 0;
            if (vdr < 0) {
                float mu  = H_SMOOTH * vdr / (r2 + 0.01f*H_SMOOTH*H_SMOOTH);
                float cs  = sqrtf(fabsf(gam * pres[i] / rho[i]) + 1e-6f);
                float rho_avg = (rho[i] + rho[j]) * 0.5f;
                visc = (-ALPHA_VISC * cs * mu + BETA_VISC * mu*mu) / rho_avg;
            }

            axi -= mass[j] * (pterm + visc) * gWx;
            ayi -= mass[j] * (pterm + visc) * gWy;
        }

        ax[i] = axi;
        ay[i] = ayi;
    }
}

/* ── Integratore leapfrog ────────────────────────────────────────────────── */
static void integrate(float dt) {
    for (int i = 0; i < N; i++) {
        if (!alive[i]) continue;
        vx[i] += ax[i] * dt;
        vy[i] += ay[i] * dt;
        px[i] += vx[i] * dt;
        py[i] += vy[i] * dt;

        /* riscaldamento adiabatico da compressione */
        float compression = -(ax[i]*vx[i] + ay[i]*vy[i]);
        if (compression > 0) u[i] += 0.003f * compression * dt;
        if (u[i] < 0.001f) u[i] = 0.001f;

        /* raffreddamento radiativo per fase */
        if (phase == 0) u[i] *= COOL_CLOUD;
        if (phase == 2) u[i] *= COOL_STABLE;
        if (phase == 4) u[i] *= 0.995f;
    }

    /* integra stella di neutroni */
    if (ns_active) {
        ns_x += ns_vx * dt;
        ns_y += ns_vy * dt;
        ns_pulse += dt * 0.20f;
    }
}

/* ── Centro di massa (per camera seguente) ──────────────────────────────── */
static void center_of_mass(float *cmx, float *cmy) {
    float sx = 0, sy = 0, sm = 0;
    for (int i = 0; i < N; i++) {
        if (!alive[i]) continue;
        sx += px[i] * mass[i];
        sy += py[i] * mass[i];
        sm += mass[i];
    }
    *cmx = (sm > 0) ? sx/sm : CX;
    *cmy = (sm > 0) ? sy/sm : CY;
}

/* ── Diagnostica ────────────────────────────────────────────────────────── */
static void update_diagnostics(void) {
    float max_rho = 0, max_u = 0;
    float cmx, cmy;
    center_of_mass(&cmx, &cmy);
    float r2sum = 0;
    float KE = 0, TE = 0;
    int n_alive = 0;

    for (int i = 0; i < N; i++) {
        if (!alive[i]) continue;
        n_alive++;
        if (rho[i] > max_rho) { max_rho = rho[i]; max_u = u[i]; }
        r2sum += (px[i]-cmx)*(px[i]-cmx) + (py[i]-cmy)*(py[i]-cmy);
        KE += 0.5f * mass[i] * (vx[i]*vx[i] + vy[i]*vy[i]);
        TE += mass[i] * u[i];
    }

    diag_max_rho = max_rho;
    diag_max_u   = max_u;
    diag_rms     = (n_alive > 0) ? sqrtf(r2sum / n_alive) : 0;
    diag_energy  = KE + TE;
    diag_n_alive = n_alive;
}

/* ── Gestione fasi ──────────────────────────────────────────────────────── */
static void trigger_supernova(float cmx, float cmy) {
    sn_triggered = 1;
    for (int i = 0; i < N; i++) {
        if (!alive[i]) continue;
        float dx = px[i]-cmx, dy = py[i]-cmy;
        float r  = sqrtf(dx*dx + dy*dy) + 1.0f;
        float kick = 16.0f / (1.0f + r * 0.014f);
        vx[i] += kick * dx/r;
        vy[i] += kick * dy/r;
        u[i]  *= 4.5f;
    }
}

static void form_neutron_star(float cmx, float cmy) {
    ns_formed = 1;

    /* trova le 50 particelle più vicine al centro → rimosso per residuo */
    /* semplice selection: killa le N_KILL più interne */
    int killed = 0;
    for (int pass = 0; pass < 50 && killed < 50; pass++) {
        float min_r2 = 1e12f;
        int   min_i  = -1;
        for (int i = 0; i < N; i++) {
            if (!alive[i]) continue;
            float dx = px[i]-cmx, dy = py[i]-cmy;
            float r2 = dx*dx + dy*dy;
            if (r2 < min_r2) { min_r2 = r2; min_i = i; }
        }
        if (min_i >= 0) { alive[min_i] = 0; killed++; }
    }

    ns_x = cmx; ns_y = cmy;
    ns_vx = 0.2f * ((float)rand()/RAND_MAX - 0.5f);
    ns_vy = 0.2f * ((float)rand()/RAND_MAX - 0.5f);
    ns_pulse  = 0;
    ns_active = 1;
}

static void update_phase(void) {
    phase_timer++;
    float cmx, cmy;
    center_of_mass(&cmx, &cmy);

    if (phase == 0 && diag_max_rho > RHO_PROTO  && phase_timer > 30) phase = 1, phase_timer = 0;
    if (phase == 1 && diag_rms     < RMS_STABLE  && phase_timer > 50) phase = 2, phase_timer = 0;
    if (phase == 2 && phase_timer > T_SUPERNOVA  && !sn_triggered)   { trigger_supernova(cmx,cmy); phase = 3; phase_timer = 0; }
    if (phase == 3 && phase_timer > T_NS         && !ns_formed)       { form_neutron_star(cmx,cmy); phase = 4; phase_timer = 0; }
}

/* ── API esportata a JavaScript ─────────────────────────────────────────── */

EMSCRIPTEN_KEEPALIVE
void sim_init(int n) {
    if (n > N_MAX) n = N_MAX;
    N = n;
    phase = 0; phase_timer = 0; sim_time = 0;
    sn_triggered = 0; ns_formed = 0; ns_active = 0;

    float total_vx = 0, total_vy = 0;

    for (int i = 0; i < N; i++) {
        /* distribuzione radiale ~ r^0.4 (profilo Bonnor-Ebert approssimato) */
        float r, angle;
        r     = powf((float)rand()/RAND_MAX, 0.40f) * CLOUD_R;
        angle = ((float)rand()/RAND_MAX) * 6.28318f;

        px[i] = CX + r * cosf(angle);
        py[i] = CY + r * sinf(angle);

        float dx = px[i]-CX, dy = py[i]-CY;
        vx[i] = -OMEGA_INIT * dy;
        vy[i] =  OMEGA_INIT * dx;

        total_vx += vx[i];
        total_vy += vy[i];

        ax[i] = ay[i] = 0;
        rho[i] = 0; pres[i] = 0;
        u[i]   = 0.002f;   /* quasi-zero: nube freddissima */
        mass[i] = 1.0f;
        alive[i] = 1;
    }

    /* azzera il momento lineare totale → impedisce il drift del sistema */
    float mean_vx = total_vx / N;
    float mean_vy = total_vy / N;
    for (int i = 0; i < N; i++) {
        vx[i] -= mean_vx;
        vy[i] -= mean_vy;
    }
}

EMSCRIPTEN_KEEPALIVE
void sim_step(int n_substeps) {
    for (int s = 0; s < n_substeps; s++) {
        float dt = DT_BASE;

        /* costruisci albero BH */
        bh_pool_top = 0;
        int root = bh_alloc(CX, CY, CANVAS_W * 1.6f);
        for (int i = 0; i < N; i++) {
            if (alive[i]) bh_insert(root, i);
        }

        compute_density();
        compute_accel(root);
        integrate(dt);
        update_diagnostics();
        update_phase();

        /* rimuovi particelle fuggite post-supernova */
        float cmx, cmy;
        center_of_mass(&cmx, &cmy);
        for (int i = 0; i < N; i++) {
            if (!alive[i]) continue;
            float dx = px[i]-cmx, dy = py[i]-cmy;
            if (phase >= 3 && dx*dx+dy*dy > (CANVAS_W*0.78f)*(CANVAS_W*0.78f))
                alive[i] = 0;
        }

        sim_time += dt;
    }
}

/*
 * sim_get_state — scrive nel buffer di output:
 * [n_alive, ns_active, ns_x, ns_y, ns_pulse, cam_x, cam_y, 0,   ← 8 float header
 *  x0,y0,u0,rho0, x1,y1,u1,rho1, ...]
 * Restituisce il puntatore al buffer (usato da JS tramite HEAPF32)
 */
EMSCRIPTEN_KEEPALIVE
float* sim_get_state(void) {
    float cmx, cmy;
    center_of_mass(&cmx, &cmy);

    /* header */
    state_buf[0] = (float)diag_n_alive;
    state_buf[1] = (float)ns_active;
    state_buf[2] = ns_x;
    state_buf[3] = ns_y;
    state_buf[4] = ns_pulse;
    state_buf[5] = cmx;   /* camera center x */
    state_buf[6] = cmy;   /* camera center y */
    state_buf[7] = 0;

    /* dati particelle (coordinate già in spazio canvas) */
    int out = 8;
    for (int i = 0; i < N; i++) {
        if (!alive[i]) continue;
        state_buf[out++] = px[i];
        state_buf[out++] = py[i];
        state_buf[out++] = u[i];
        state_buf[out++] = rho[i];
    }

    return state_buf;
}

EMSCRIPTEN_KEEPALIVE
int sim_get_phase(void) {
    return phase;
}

/*
 * sim_get_diagnostics — scrive 5 float:
 * [max_rho, max_u, rms_radius, total_energy, n_alive]
 */
EMSCRIPTEN_KEEPALIVE
float* sim_get_diagnostics(void) {
    static float dbuf[5];
    dbuf[0] = diag_max_rho;
    dbuf[1] = diag_max_u;
    dbuf[2] = diag_rms;
    dbuf[3] = diag_energy;
    dbuf[4] = (float)diag_n_alive;
    return dbuf;
}
