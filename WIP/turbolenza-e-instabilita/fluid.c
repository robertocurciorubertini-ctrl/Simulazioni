/*
 * fluid.c — Solver Navier-Stokes 2D incomprimibile
 * Metodo: Stable Fluids (Stam 1999)
 * Fedele alla versione JS che funzionava.
 *
 * Compilare con Emscripten:
 *   emcc fluid.c -O3 -o fluid.js \
 *     -s EXPORTED_FUNCTIONS='["_step","_compute_vorticity","_add_force","_add_turbulence","_set_params","_reset_fields","_build_obstacle","_get_ux","_get_uy","_get_vort","_get_pres","_get_dye","_get_wall","_NX","_NY"]' \
 *     -s EXPORTED_RUNTIME_METHODS='["cwrap","HEAPF32","HEAPU8"]' \
 *     -s ALLOW_MEMORY_GROWTH=1 \
 *     -s MODULARIZE=1 \
 *     -s EXPORT_NAME='FluidModule'
 */

#include <math.h>
#include <string.h>
#include <stdlib.h>

/* ── Dimensioni griglia ─────────────────────────────────────── */
#define NX 300
#define NY 160
#define N  (NX * NY)

int _NX = NX;
int _NY = NY;

/* ── Campi fisici ───────────────────────────────────────────── */
float _ux[N],   _uy[N];       /* velocità corrente              */
float _ux0[N],  _uy0[N];      /* buffer temporaneo              */
float _pres[N], _div[N];      /* pressione e divergenza         */
float _dye[N],  _dye0[N];     /* tracciante passivo             */
float _vort[N];               /* vorticità ω = duy/dx - dux/dy */
unsigned char _wall[N];       /* maschera solido (1=ostacolo)   */

/* ── Parametri (modificabili a runtime) ─────────────────────── */
static float inlet_vel = 0.15f;
static float viscosity  = 0.0008f;
static float dt         = 1.0f;

/* ── Indice con clamp ───────────────────────────────────────── */
static inline int IX(int x, int y) {
    if (x < 0)    x = 0;
    if (x >= NX)  x = NX - 1;
    if (y < 0)    y = 0;
    if (y >= NY)  y = NY - 1;
    return x + y * NX;
}

/* ═══════════════════════════════════════════════════════════════
   CONDIZIONI AL CONTORNO
   Identiche alla versione JS che funzionava:
   - Inlet sinistro: velocità uniforme con profilo Poiseuille leggero
   - Outlet destro:  Neumann (copia dal vicino)
   - Pareti sup/inf: no-slip
   - Ostacoli:       no-slip
═══════════════════════════════════════════════════════════════ */
static void apply_bc(float *ux, float *uy) {
    int i, j;

    /* Inlet */
    for (j = 1; j < NY - 1; j++) {
        ux[IX(0, j)] = inlet_vel;
        uy[IX(0, j)] = 0.0f;
    }
    /* Outlet: Neumann */
    for (j = 0; j < NY; j++) {
        ux[IX(NX-1, j)] = ux[IX(NX-2, j)];
        uy[IX(NX-1, j)] = uy[IX(NX-2, j)];
    }
    /* Pareti orizzontali: no-slip */
    for (i = 0; i < NX; i++) {
        ux[IX(i, 0)]    = 0.0f;  uy[IX(i, 0)]    = 0.0f;
        ux[IX(i, NY-1)] = 0.0f;  uy[IX(i, NY-1)] = 0.0f;
    }
    /* Ostacoli: no-slip */
    for (i = 0; i < N; i++) {
        if (_wall[i]) { ux[i] = 0.0f; uy[i] = 0.0f; }
    }
}

/* ═══════════════════════════════════════════════════════════════
   DIFFUSIONE IMPLICITA — Gauss-Seidel
   (I - ν·Δt·∇²) dst = src
═══════════════════════════════════════════════════════════════ */
static void diffuse(float *dst, const float *src, float nu, float dt_) {
    float a     = dt_ * nu;
    float c_inv = 1.0f / (1.0f + 4.0f * a);
    int i, j, iter;

    memcpy(dst, src, N * sizeof(float));

    for (iter = 0; iter < 4; iter++) {
        for (j = 1; j < NY - 1; j++) {
            for (i = 1; i < NX - 1; i++) {
                if (_wall[IX(i,j)]) continue;
                dst[IX(i,j)] = (src[IX(i,j)] + a * (
                    dst[IX(i-1,j)] + dst[IX(i+1,j)] +
                    dst[IX(i,j-1)] + dst[IX(i,j+1)]
                )) * c_inv;
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
   ADVECTION SEMI-LAGRANGIANA
   dst(x) = src(x - u·Δt)  con interpolazione bilineare
═══════════════════════════════════════════════════════════════ */
static void advect(float *dst, const float *src,
                   const float *ufx, const float *ufy, float dt_) {
    int i, j;
    for (j = 1; j < NY - 1; j++) {
        for (i = 1; i < NX - 1; i++) {
            if (_wall[IX(i,j)]) { dst[IX(i,j)] = 0.0f; continue; }

            float xi = (float)i - dt_ * ufx[IX(i,j)];
            float yi = (float)j - dt_ * ufy[IX(i,j)];

            /* clamp entro il dominio */
            if (xi < 0.5f)        xi = 0.5f;
            if (xi > NX - 1.5f)   xi = NX - 1.5f;
            if (yi < 0.5f)        yi = 0.5f;
            if (yi > NY - 1.5f)   yi = NY - 1.5f;

            int i0 = (int)xi, i1 = i0 + 1;
            int j0 = (int)yi, j1 = j0 + 1;
            float sx = xi - i0, sy = yi - j0;

            dst[IX(i,j)] =
                (1.0f - sx) * ((1.0f - sy) * src[IX(i0,j0)] + sy * src[IX(i0,j1)]) +
                         sx  * ((1.0f - sy) * src[IX(i1,j0)] + sy * src[IX(i1,j1)]);
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
   PROIEZIONE DI HELMHOLTZ-HODGE
   Rimuove la componente a divergenza non-nulla:
   1) div ← -∇·u
   2) ∇²p = div   (Poisson, Gauss-Seidel)
   3) u  ← u - ∇p
═══════════════════════════════════════════════════════════════ */
static void project(void) {
    int i, j, iter;

    /* divergenza */
    for (j = 1; j < NY - 1; j++) {
        for (i = 1; i < NX - 1; i++) {
            if (_wall[IX(i,j)]) {
                _div[IX(i,j)]  = 0.0f;
                _pres[IX(i,j)] = 0.0f;
                continue;
            }
            _div[IX(i,j)]  = -0.5f * (
                _ux[IX(i+1,j)] - _ux[IX(i-1,j)] +
                _uy[IX(i,j+1)] - _uy[IX(i,j-1)]
            );
            _pres[IX(i,j)] = 0.0f;
        }
    }

    /* Poisson per la pressione — 20 iterazioni come nella versione JS */
    for (iter = 0; iter < 20; iter++) {
        for (j = 1; j < NY - 1; j++) {
            for (i = 1; i < NX - 1; i++) {
                if (_wall[IX(i,j)]) continue;
                _pres[IX(i,j)] = (
                    _div[IX(i,j)] +
                    _pres[IX(i-1,j)] + _pres[IX(i+1,j)] +
                    _pres[IX(i,j-1)] + _pres[IX(i,j+1)]
                ) * 0.25f;
            }
        }
    }

    /* sottrai gradiente pressione */
    for (j = 1; j < NY - 1; j++) {
        for (i = 1; i < NX - 1; i++) {
            if (_wall[IX(i,j)]) continue;
            _ux[IX(i,j)] -= 0.5f * (_pres[IX(i+1,j)] - _pres[IX(i-1,j)]);
            _uy[IX(i,j)] -= 0.5f * (_pres[IX(i,j+1)] - _pres[IX(i,j-1)]);
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
   STEP PRINCIPALE — identico alla versione JS
═══════════════════════════════════════════════════════════════ */
void step(void) {
    int k;

    /* 1. Diffusione implicita velocità */
    diffuse(_ux0, _ux, viscosity, dt);
    diffuse(_uy0, _uy, viscosity, dt);
    apply_bc(_ux0, _uy0);

    /* 2. Self-advection */
    advect(_ux, _ux0, _ux0, _uy0, dt);
    advect(_uy, _uy0, _ux0, _uy0, dt);
    apply_bc(_ux, _uy);

    /* 3. Proiezione */
    project();
    apply_bc(_ux, _uy);

    /* 4. Advection tracciante passivo */
    advect(_dye0, _dye, _ux, _uy, dt);

    /* decay + ricarica inlet */
    for (k = 0; k < N; k++) {
        _dye0[k] *= 0.998f;
        if (_wall[k]) _dye0[k] = 0.0f;
    }

    /* swap dye */
    memcpy(_dye, _dye0, N * sizeof(float));
}

/* ═══════════════════════════════════════════════════════════════
   VORTICITÀ   ω = ∂uy/∂x − ∂ux/∂y
═══════════════════════════════════════════════════════════════ */
void compute_vorticity(void) {
    int i, j;
    for (j = 1; j < NY - 1; j++) {
        for (i = 1; i < NX - 1; i++) {
            _vort[IX(i,j)] =
                (_uy[IX(i+1,j)] - _uy[IX(i-1,j)]) * 0.5f -
                (_ux[IX(i,j+1)] - _ux[IX(i,j-1)]) * 0.5f;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
   INTERAZIONE ESTERNA
═══════════════════════════════════════════════════════════════ */
void add_force(int cx, int cy, float fx, float fy, int radius) {
    int di, dj;
    for (dj = -radius; dj <= radius; dj++) {
        for (di = -radius; di <= radius; di++) {
            if (di*di + dj*dj > radius*radius) continue;
            int xi = cx + di, yi = cy + dj;
            if (xi < 0 || xi >= NX || yi < 0 || yi >= NY) continue;
            if (_wall[IX(xi,yi)]) continue;
            float w = 1.0f - (float)(di*di+dj*dj) / (float)(radius*radius);
            _ux[IX(xi,yi)] += fx * w;
            _uy[IX(xi,yi)] += fy * w;
        }
    }
}

/* LCG per numeri pseudo-casuali leggero */
static unsigned int rng = 12345u;
static float randf(void) {
    rng = rng * 1664525u + 1013904223u;
    return (float)(rng >> 16) / 65535.0f - 0.5f;
}

void add_turbulence(float strength) {
    int i, j;
    for (j = NY/8; j < NY*7/8; j++)
        for (i = NX/8; i < NX*7/8; i++) {
            if (_wall[IX(i,j)]) continue;
            _ux[IX(i,j)] += randf() * strength;
            _uy[IX(i,j)] += randf() * strength;
        }
}

/* ═══════════════════════════════════════════════════════════════
   SETTER PARAMETRI
═══════════════════════════════════════════════════════════════ */
void set_params(float vel, float visc) {
    inlet_vel = vel;
    viscosity  = visc;
}

/* ═══════════════════════════════════════════════════════════════
   RESET CAMPI — esattamente come la versione JS
═══════════════════════════════════════════════════════════════ */
void reset_fields(void) {
    int i, j;
    memset(_ux,   0, N * sizeof(float));
    memset(_uy,   0, N * sizeof(float));
    memset(_ux0,  0, N * sizeof(float));
    memset(_uy0,  0, N * sizeof(float));
    memset(_pres, 0, N * sizeof(float));
    memset(_dye,  0, N * sizeof(float));
    memset(_dye0, 0, N * sizeof(float));

    /* Inizializza flusso uniforme nelle celle libere */
    for (j = 1; j < NY-1; j++)
        for (i = 1; i < NX-1; i++)
            if (!_wall[IX(i,j)])
                _ux[IX(i,j)] = inlet_vel;

    /* Piccola perturbazione verticale per stimolare instabilità */
    for (j = 1; j < NY-1; j++)
        for (i = 1; i < NX-1; i++)
            if (!_wall[IX(i,j)])
                _uy[IX(i,j)] = randf() * inlet_vel * 0.04f;

    /* Dye a strisce all'inlet */
    for (j = 1; j < NY-1; j++)
        _dye[IX(0,j)] = ((j / 20) % 2 == 0) ? 1.0f : 0.0f;
}

/* ═══════════════════════════════════════════════════════════════
   COSTRUZIONE OSTACOLI
   Tutte le build azzerano il wall e aggiungono le pareti.
   Il chiamante (JS) chiama poi reset_fields().
═══════════════════════════════════════════════════════════════ */
static void add_walls(void) {
    int i;
    for (i = 0; i < NX; i++) {
        _wall[IX(i, 0)]    = 1;
        _wall[IX(i, NY-1)] = 1;
    }
}

/* Scenario 1: cilindro singolo — scia di Kármán */
void build_obstacle(int scenario, int param) {
    int i, j;
    memset(_wall, 0, N);

    int cx = NX * 28 / 100;
    int cy = NY / 2;
    int r  = param;   /* raggio o parametro geometrico */

    switch (scenario) {

    case 0: /* Cilindro singolo */
        for (j = 0; j < NY; j++)
            for (i = 0; i < NX; i++) {
                int dx = i-cx, dy = j-cy;
                if (dx*dx + dy*dy < r*r) _wall[IX(i,j)] = 1;
            }
        break;

    case 1: /* Due cilindri sovrapposti */
        for (j = 0; j < NY; j++)
            for (i = 0; i < NX; i++) {
                int dx = i-cx;
                int dy1 = j - (cy - r*2), dy2 = j - (cy + r*2);
                int r2 = r * 4 / 5;
                if (dx*dx+dy1*dy1 < r2*r2 || dx*dx+dy2*dy2 < r2*r2)
                    _wall[IX(i,j)] = 1;
            }
        break;

    case 2: /* Fessura / Venturi */
        {
            int gap  = param;
            int xpos = NX * 28 / 100;
            for (j = 0; j < NY; j++) {
                int inGap = (j > NY/2 - gap/2 && j < NY/2 + gap/2);
                if (!inGap)
                    for (i = xpos-2; i <= xpos+2; i++)
                        _wall[IX(i,j)] = 1;
            }
        }
        break;

    case 3: /* Griglia di ostacoli */
        {
            int r2   = r * 6 / 10;
            int xs[] = {NX*28/100, NX*50/100, NX*72/100};
            int ys[] = {NY/4, NY/2, NY*3/4};
            int pi, yi;
            for (j = 0; j < NY; j++)
                for (i = 0; i < NX; i++)
                    for (pi = 0; pi < 3; pi++)
                        for (yi = 0; yi < 3; yi++) {
                            int dx = i-xs[pi], dy = j-ys[yi];
                            if (dx*dx+dy*dy < r2*r2) _wall[IX(i,j)] = 1;
                        }
        }
        break;

    case 4: /* Gradino — backward-facing step */
        for (j = 0; j < NY/2; j++)
            for (i = 0; i < NX/4; i++)
                _wall[IX(i,j)] = 1;
        break;

    case 5: /* Kelvin-Helmholtz: no ostacolo, shear layer */
        /* Nessun ostacolo. L'init del campo velocità va fatto separatamente. */
        break;
    }

    add_walls();
}

/* Inizializza il campo per Kelvin-Helmholtz:
   strato superiore a +v, inferiore a -v, interfaccia perturbata */
void init_kelvin_helmholtz(void) {
    int i, j;
    memset(_ux,   0, N * sizeof(float));
    memset(_uy,   0, N * sizeof(float));
    memset(_pres, 0, N * sizeof(float));
    memset(_dye,  0, N * sizeof(float));

    float v0 = inlet_vel * 0.8f;
    float pi2 = 6.28318f;

    for (j = 0; j < NY; j++) {
        /* Profilo smussato dello shear layer */
        float yc  = (float)(j - NY/2) / (NY * 0.04f);
        float sgn = tanhf(yc);     /* da -1 a +1 attraverso l'interfaccia */

        /* Perturbazione sinusoidale per seedare il modo instabile */
        float pert = 0.03f * v0 * sinf(pi2 * 4.0f * i / NX);

        for (i = 0; i < NX; i++) {
            if (_wall[IX(i,j)]) continue;
            _ux[IX(i,j)] = v0 * sgn;
            _uy[IX(i,j)] = pert;
            /* Dye: sopra bianco, sotto nero */
            _dye[IX(i,j)] = (j < NY/2) ? 1.0f : 0.0f;
        }
    }
}

/* Ricarica il dye all'inlet per scenari con flusso da sinistra */
void refresh_inlet_dye(int stripe_width) {
    int j;
    for (j = 1; j < NY-1; j++)
        if (!_wall[IX(0,j)])
            _dye[IX(0,j)] = ((j / stripe_width) % 2 == 0) ? 1.0f : 0.0f;
}

/* ═══════════════════════════════════════════════════════════════
   ACCESSOR — restituiscono puntatori ai buffer per JS/WASM
═══════════════════════════════════════════════════════════════ */
float         *get_ux(void)   { return _ux;   }
float         *get_uy(void)   { return _uy;   }
float         *get_vort(void) { return _vort; }
float         *get_pres(void) { return _pres; }
float         *get_dye(void)  { return _dye;  }
unsigned char *get_wall(void) { return _wall; }
