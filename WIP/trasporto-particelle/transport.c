/*
 * transport.c  —  Simulazione Monte Carlo del trasporto di particelle accoppiate
 *
 * Fisica implementata:
 *   NEUTRONI  : scattering elastico (cinematica CM), cattura radiativa, fissione (U-235)
 *               con sezioni d'urto semplificate ma energeticamente dipendenti
 *   FOTONI    : effetto fotoelettrico, scattering Compton (Klein-Nishina esatto),
 *               produzione di coppie (sopra 1.022 MeV)
 *   ELETTRONI : perdita continua di energia (Bethe-Bloch), scattering angolare
 *               multiplo (Molière), produzione Bremsstrahlung
 *
 * Compilazione con Emscripten:
 *   emcc transport.c -O2 -o transport.js \
 *        -s EXPORTED_FUNCTIONS='["_sim_init","_sim_step","_sim_reset",
 *                                "_get_fluence","_get_dose","_get_tracks",
 *                                "_get_track_count","_set_source","_set_geometry"]' \
 *        -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap"]' \
 *        -s ALLOW_MEMORY_GROWTH=1 \
 *        -s INITIAL_MEMORY=67108864
 *
 * Unità interne:
 *   Lunghezze : cm
 *   Energie   : MeV
 *   Tempo     : ns (non usato direttamente in steady-state)
 *   Sezioni d'urto : cm^-1 (macroscopiche, Σ = n·σ)
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#define EXPORT EMSCRIPTEN_KEEPALIVE
#else
#define EXPORT
#endif

/* ─────────────────────────────────────────────
   COSTANTI FISICHE
   ───────────────────────────────────────────── */

#define PI              3.14159265358979323846
#define TWO_PI          6.28318530717958647692
#define M_E_MEV         0.510998950          /* massa dell'elettrone in MeV/c^2      */
#define M_N_MEV         939.565420           /* massa del neutrone in MeV/c^2        */
#define R_E_CM          2.8179403227e-13     /* raggio classico dell'elettrone in cm */
#define AVOGADRO        6.02214076e23        /* numero di Avogadro                   */
#define E_PAIR_THRESH   1.021999             /* soglia produzione di coppie in MeV   */
#define E_CUTOFF_PHOTON 0.001                /* cutoff fotone 1 keV                  */
#define E_CUTOFF_ELEC   0.001                /* cutoff elettrone 1 keV               */
#define E_CUTOFF_NEUT   1e-5                 /* cutoff neutrone termico ~0.025 eV    */

/* ─────────────────────────────────────────────
   GEOMETRIA E GRIGLIA
   ───────────────────────────────────────────── */

#define GRID_W          200
#define GRID_H          200
#define CELL_SIZE_CM    0.1                  /* ogni cella = 1 mm in unità reali     */
#define DOMAIN_W_CM     (GRID_W * CELL_SIZE_CM)
#define DOMAIN_H_CM     (GRID_H * CELL_SIZE_CM)

/* ─────────────────────────────────────────────
   POOL DI PARTICELLE
   ───────────────────────────────────────────── */

#define MAX_PARTICLES   50000
#define MAX_TRACKS      8000
#define MAX_TRACK_PTS   64
#define SECONDARY_QUEUE 20000

/* ─────────────────────────────────────────────
   TIPI DI PARTICELLE
   ───────────────────────────────────────────── */

typedef enum {
    PTYPE_NEUTRON = 0,
    PTYPE_PHOTON  = 1,
    PTYPE_ELECTRON = 2,
    PTYPE_POSITRON = 3
} ParticleType;

/* ─────────────────────────────────────────────
   MATERIALI
   Ogni cella della griglia ha un indice materiale.
   Le proprietà sono pre-calcolate e dipendono
   dall'energia in modo parametrico.
   ───────────────────────────────────────────── */

typedef enum {
    MAT_VACUUM  = 0,
    MAT_AIR     = 1,
    MAT_WATER   = 2,
    MAT_IRON    = 3,
    MAT_LEAD    = 4,
    MAT_U235    = 5,
    MAT_CONCRETE= 6,
    MAT_COUNT   = 7
} MaterialID;

/* Proprietà macroscopiche del materiale (densità + composizione) */
typedef struct {
    double density;        /* g/cm^3                                   */
    double Z_eff;          /* numero atomico efficace                  */
    double A_eff;          /* numero di massa efficace                 */
    double I_eV;           /* potenziale medio di ionizzazione (eV)    */
    /* Neutroni */
    double sig_s_barn;     /* sezione d'urto di scattering elastico (barn, 1 MeV ref) */
    double sig_c_barn;     /* cattura radiativa (barn, 0.025 eV termico)               */
    double sig_f_barn;     /* fissione (solo per U-235, barn, 0.025 eV)                */
    double nu_fiss;        /* neutroni per fissione                                    */
    double E_fiss_MeV;     /* energia rilasciata per fissione (MeV)                    */
    /* Fotoni — coefficienti di attenuazione a 1 MeV (cm^2/g) */
    double mu_pe_1MeV;     /* fotoelettrico                                            */
    double mu_compton_1MeV;/* Compton                                                  */
    double mu_pair_1MeV;   /* produzione di coppie                                     */
    /* Elettroni — stopping power parametrico */
    double rho_e;          /* densità elettronica (e/cm^3)                             */
} Material;

static const Material MATERIALS[MAT_COUNT] = {
    /* VACUUM */
    { 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0 },
    /* AIR (Z_eff=7.36, A_eff=14.5, rho=0.001293 g/cm^3) */
    { 1.293e-3, 7.36, 14.5, 85.7,
      10.0, 1.0, 0.0, 0.0, 0.0,
      0.0003, 0.1535, 0.0,
      3.88e20 },
    /* WATER (Z=7.22 eff, rho=1.0) */
    { 1.0, 7.22, 11.9, 75.0,
      20.0, 0.3, 0.0, 0.0, 0.0,
      0.0025, 0.0706, 0.0,
      3.34e23 },
    /* IRON (Z=26, A=55.85, rho=7.874) */
    { 7.874, 26.0, 55.85, 286.0,
      2.5, 2.6, 0.0, 0.0, 0.0,
      0.0294, 0.05951, 0.00567,
      2.20e24 },
    /* LEAD (Z=82, A=207.2, rho=11.35) */
    { 11.35, 82.0, 207.2, 823.0,
      11.0, 0.17, 0.0, 0.0, 0.0,
      5.550, 0.04551, 0.04572,
      3.30e24 },
    /* U-235 (Z=92, A=235, rho=18.95) */
    { 18.95, 92.0, 235.0, 890.0,
      4.0, 99.0, 584.0, 2.43, 200.0,
      8.0, 0.03696, 0.08265,
      4.84e24 },
    /* CONCRETE (Z_eff=11.0, rho=2.3) */
    { 2.3, 11.0, 22.0, 135.7,
      3.0, 0.1, 0.0, 0.0, 0.0,
      0.0023, 0.06388, 0.00032,
      6.93e23 }
};

/* ─────────────────────────────────────────────
   GENERATORE DI NUMERI CASUALI
   xoshiro256** — periodo 2^256-1, veloce e di buona qualità
   ───────────────────────────────────────────── */

static uint64_t rng_s[4];

static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static uint64_t rng_next(void) {
    const uint64_t result = rotl(rng_s[1] * 5, 7) * 9;
    const uint64_t t = rng_s[1] << 17;
    rng_s[2] ^= rng_s[0]; rng_s[3] ^= rng_s[1];
    rng_s[1] ^= rng_s[2]; rng_s[0] ^= rng_s[3];
    rng_s[2] ^= t;
    rng_s[3] = rotl(rng_s[3], 45);
    return result;
}

static inline double rng_double(void) {
    return (rng_next() >> 11) * (1.0 / (UINT64_C(1) << 53));
}

static void rng_seed(uint64_t seed) {
    /* splitmix64 per l'inizializzazione */
    for (int i = 0; i < 4; i++) {
        seed += 0x9e3779b97f4a7c15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        rng_s[i] = z ^ (z >> 31);
    }
}

/* ─────────────────────────────────────────────
   STRUTTURA PARTICELLA
   ───────────────────────────────────────────── */

typedef struct {
    double x, y;           /* posizione (cm)                           */
    double dx, dy;         /* direzione (versore, |d|=1)               */
    double energy;         /* energia cinetica (MeV)                   */
    double weight;         /* peso statistico (importance sampling)    */
    ParticleType type;
    int    alive;
    int    generation;     /* 0=primaria, 1=secondaria, ...            */
    uint8_t mat_idx;       /* materiale corrente                       */
} Particle;

/* ─────────────────────────────────────────────
   TRACCIA (per visualizzazione)
   ───────────────────────────────────────────── */

typedef struct {
    float px[MAX_TRACK_PTS];
    float py[MAX_TRACK_PTS];
    float pe[MAX_TRACK_PTS];   /* energia al punto (MeV)               */
    int   n_pts;
    ParticleType type;
    int   generation;
} Track;

/* ─────────────────────────────────────────────
   STATO GLOBALE DELLA SIMULAZIONE
   ───────────────────────────────────────────── */

static Particle  particles[MAX_PARTICLES];
static Particle  sec_queue[SECONDARY_QUEUE];
static int       n_sec = 0;

static Track     tracks[MAX_TRACKS];
static int       n_tracks = 0;
static int       track_write_idx = 0;

/* Griglia materiali */
static uint8_t   mat_grid[GRID_H][GRID_W];

/* Buffer di fluenza separati per tipo [H][W] */
static float     fluence_n[GRID_H][GRID_W];   /* neutroni        */
static float     fluence_p[GRID_H][GRID_W];   /* fotoni          */
static float     fluence_e[GRID_H][GRID_W];   /* elettroni       */

/* Mappa di dose (energia depositata per unità di massa, Gy equivalente) */
static float     dose_map[GRID_H][GRID_W];

/* Parametri sorgente */
static double    src_x, src_y;            /* posizione sorgente (cm)     */
static double    src_energy;              /* energia iniziale (MeV)       */
static ParticleType src_type;
static int       src_isotropic;           /* 1=isotropica, 0=direzionale  */
static double    src_dx, src_dy;          /* direzione se non isotropica  */

/* Statistiche */
static long      total_histories;
static long      total_fissions;
static long      total_captures;
static long      total_scatterings;

/* ─────────────────────────────────────────────
   FUNZIONI GEOMETRICHE
   ───────────────────────────────────────────── */

static inline int in_domain(double x, double y) {
    return (x >= 0.0 && x < DOMAIN_W_CM && y >= 0.0 && y < DOMAIN_H_CM);
}

static inline int get_cell_x(double x) { return (int)(x / CELL_SIZE_CM); }
static inline int get_cell_y(double y) { return (int)(y / CELL_SIZE_CM); }

static inline MaterialID get_material(double x, double y) {
    if (!in_domain(x, y)) return MAT_VACUUM;
    int cx = get_cell_x(x);
    int cy = get_cell_y(y);
    if (cx < 0 || cx >= GRID_W || cy < 0 || cy >= GRID_H) return MAT_VACUUM;
    return (MaterialID)mat_grid[cy][cx];
}

static inline void deposit_fluence(double x, double y, ParticleType t, double w) {
    if (!in_domain(x, y)) return;
    int cx = get_cell_x(x); int cy = get_cell_y(y);
    if (cx < 0 || cx >= GRID_W || cy < 0 || cy >= GRID_H) return;
    switch(t) {
        case PTYPE_NEUTRON:  fluence_n[cy][cx] += (float)w; break;
        case PTYPE_PHOTON:   fluence_p[cy][cx] += (float)w; break;
        case PTYPE_ELECTRON:
        case PTYPE_POSITRON: fluence_e[cy][cx] += (float)w; break;
    }
}

static inline void deposit_dose(double x, double y, double dE_MeV, double rho) {
    if (!in_domain(x, y) || rho <= 0.0) return;
    int cx = get_cell_x(x); int cy = get_cell_y(y);
    if (cx < 0 || cx >= GRID_W || cy < 0 || cy >= GRID_H) return;
    /* Dose = dE / (rho * V_cell) — normalizziamo per rho soltanto per confronto relativo */
    double cell_vol = CELL_SIZE_CM * CELL_SIZE_CM * CELL_SIZE_CM; /* cm^3 */
    double mass_g   = rho * cell_vol;
    /* converti MeV in J: 1 MeV = 1.602176634e-13 J; Gy = J/kg = J / (mass_g * 1e-3) */
    double dose_gy  = dE_MeV * 1.602176634e-13 / (mass_g * 1e-3);
    dose_map[cy][cx] += (float)dose_gy;
}

/* ─────────────────────────────────────────────
   SAMPLING DI DIREZIONI
   ───────────────────────────────────────────── */

static void sample_isotropic(double *dx, double *dy) {
    double phi   = TWO_PI * rng_double();
    double costh = 2.0 * rng_double() - 1.0;
    double sinth = sqrt(1.0 - costh * costh);
    *dx = sinth * cos(phi);
    *dy = sinth * sin(phi);
    /* In 2D usiamo solo la proiezione nel piano xy */
    double norm = sqrt((*dx)*(*dx) + (*dy)*(*dy));
    if (norm > 1e-12) { *dx /= norm; *dy /= norm; }
    else { *dx = 1.0; *dy = 0.0; }
}

/* Rotazione 2D di un versore (dx,dy) di un angolo theta */
static void rotate_direction(double *dx, double *dy, double costh, double sinth) {
    double phi = TWO_PI * rng_double();
    double old_dx = *dx, old_dy = *dy;
    /* In 2D: rotazione diretta + componente fuori-piano campionata separatamente */
    double sinphi = sin(phi), cosphi = cos(phi);
    *dx = costh * old_dx - sinth * sinphi * old_dy;
    *dy = costh * old_dy + sinth * cosphi * old_dx;
    double norm = sqrt((*dx)*(*dx) + (*dy)*(*dy));
    if (norm > 1e-12) { *dx /= norm; *dy /= norm; }
}

/* ─────────────────────────────────────────────
   SEZIONI D'URTO MACROSCOPICHE
   Formulazioni semplificate ma fisicamente corrette nell'andamento con E.
   ───────────────────────────────────────────── */

/* ── NEUTRONI ──────────────────────────────── */

/*
 * Sezione d'urto totale macroscopica del neutrone.
 * Σ = (N_A * ρ / A) * σ(E)
 * Per scattering elastico: σ ≈ σ_ref * (E_ref/E)^0.1  (debole dipendenza, hard-sphere)
 * Per cattura: σ ≈ σ_c * sqrt(E_thermal/E)             (legge 1/v)
 * Per fissione: σ ≈ σ_f * sqrt(E_thermal/E) per termici, con picco di risonanza ignorato
 */
static void neutron_xs(MaterialID mid, double E_MeV,
                        double *sig_s, double *sig_c, double *sig_f) {
    const Material *m = &MATERIALS[mid];
    if (m->density <= 0.0 || m->A_eff <= 0.0) {
        *sig_s = *sig_c = *sig_f = 0.0;
        return;
    }
    double N = AVOGADRO * m->density / m->A_eff; /* atomi/cm^3 */
    double E_th = 2.53e-8;    /* 0.025 eV in MeV */
    double E_ref = 1.0;       /* MeV */

    /* Scattering: dipendenza debole con E (hard sphere + risonanze ignorate) */
    double xs_s_b = m->sig_s_barn * pow(E_ref / fmax(E_MeV, 1e-10), 0.1);
    /* Cattura: legge 1/v */
    double xs_c_b = m->sig_c_barn * sqrt(E_th / fmax(E_MeV, E_th));
    /* Fissione: legge 1/v + soppressione per veloci */
    double xs_f_b = 0.0;
    if (m->sig_f_barn > 0.0) {
        xs_f_b = m->sig_f_barn * sqrt(E_th / fmax(E_MeV, E_th));
        /* i neutroni veloci (>1 MeV) inducono fissione con efficienza ridotta */
        if (E_MeV > 1.0) xs_f_b *= (0.3 + 0.7 * exp(-(E_MeV - 1.0) / 2.0));
    }
    double b_to_cm2 = 1e-24;
    *sig_s = N * xs_s_b * b_to_cm2;
    *sig_c = N * xs_c_b * b_to_cm2;
    *sig_f = N * xs_f_b * b_to_cm2;
}

/* Cinematica CM per scattering elastico neutrone-nucleo.
 * Energia ceduta: ΔE = E * (1-α)/2 * (1 - cosθ_CM)
 * con α = ((A-1)/(A+1))^2
 * L'angolo θ_CM è campionato uniformemente in [0, π] (isotropico nel CM).
 */
static double neutron_elastic_scatter(double E_in, double A,
                                       double *costh_lab) {
    double alpha = ((A - 1.0) / (A + 1.0));
    alpha *= alpha;
    double cos_cm = 2.0 * rng_double() - 1.0;    /* isotropico nel CM */
    double E_out = E_in * (alpha + (1.0 - alpha) * (1.0 + cos_cm) / 2.0);
    /* Angolo nel lab (approssimazione 2D) */
    *costh_lab = (1.0 + A * cos_cm) / sqrt(A*A + 2.0*A*cos_cm + 1.0);
    return fmax(E_out, 0.0);
}

/* ── FOTONI ──────────────────────────────────
 *
 * Coefficienti di attenuazione totali (cm^-1):
 *   μ(E) = ρ * [μ_pe(E) + μ_compton(E) + μ_pair(E)]
 *
 * Scalatura con E basata su formule analitiche approssimate:
 *   Fotoelettrico:      μ_pe ∝ Z^4.5 / E^3.5    (regime tra 0.1-0.5 MeV)
 *   Compton:            formula Klein-Nishina integrata
 *   Produzione coppie:  μ_pp ∝ Z^2 * ln(E)      (sopra soglia)
 */

/* Cross section di Klein-Nishina totale (in unità di r_e^2):
 * σ_KN = 2π r_e^2 { (1+k)/k^2 [2(1+k)/(1+2k) - ln(1+2k)/k] + ln(1+2k)/(2k) - (1+3k)/(1+2k)^2 }
 * dove k = E / (m_e c^2)
 */
static double klein_nishina_total(double E_MeV) {
    double k   = E_MeV / M_E_MEV;
    double k2  = k * k;
    if (k < 1e-6) return 8.0 * PI * R_E_CM * R_E_CM / 3.0; /* limite Thomson */
    double ln1p2k = log(1.0 + 2.0 * k);
    double term1  = (1.0 + k) / (k2) * (2.0 * (1.0 + k) / (1.0 + 2.0 * k) - ln1p2k / k);
    double term2  = ln1p2k / (2.0 * k);
    double term3  = -(1.0 + 3.0 * k) / ((1.0 + 2.0 * k) * (1.0 + 2.0 * k));
    return 2.0 * PI * R_E_CM * R_E_CM * (term1 + term2 + term3);
}

static void photon_xs(MaterialID mid, double E_MeV,
                      double *mu_pe, double *mu_compton, double *mu_pair,
                      double *mu_total) {
    const Material *m = &MATERIALS[mid];
    if (m->density <= 0.0) {
        *mu_pe = *mu_compton = *mu_pair = *mu_total = 0.0;
        return;
    }
    /* Coefficienti di massa (cm^2/g) scalati a partire dal valore a 1 MeV */
    /* Fotoelettrico: μ/ρ ∝ Z^4.5 / E^3.5 */
    double pe_scale = pow(1.0 / fmax(E_MeV, 0.001), 3.5);
    double mu_pe_rho = m->mu_pe_1MeV * pe_scale;

    /* Compton: proporzionale alla KN / (Z / A_eff) — sezione elettronica */
    double kn_E     = klein_nishina_total(E_MeV);
    double n_e_mol  = AVOGADRO * m->Z_eff / m->A_eff; /* e/g */
    double mu_compton_rho = n_e_mol * kn_E;            /* cm^2/g */

    /* Produzione di coppie: attiva sopra soglia, ∝ Z^2 ln(2E/m_e) */
    double mu_pair_rho = 0.0;
    if (E_MeV > E_PAIR_THRESH && m->mu_pair_1MeV > 0.0) {
        double ln_factor = log(2.0 * E_MeV / M_E_MEV) / log(2.0 * 1.0 / M_E_MEV);
        mu_pair_rho = m->mu_pair_1MeV * fmax(ln_factor, 0.0);
    }
    *mu_pe      = m->density * mu_pe_rho;
    *mu_compton = m->density * mu_compton_rho;
    *mu_pair    = m->density * mu_pair_rho;
    *mu_total   = *mu_pe + *mu_compton + *mu_pair;
}

/*
 * Campionamento dello scattering Compton via metodo di Klein-Nishina.
 * Usa il metodo di Kahn (rejection sampling) per campionare il coseno
 * dell'angolo di scattering θ e l'energia del fotone diffuso.
 *
 * k = E_in / m_e
 * ε = E_out / E_in  ∈ [1/(1+2k), 1]
 * dσ/dε ∝ [ε + 1/ε - (1-cosθ)^2] con cosθ = 1 + 1/k - 1/(kε)
 */
static double compton_scatter(double E_in, double *costh) {
    double k    = E_in / M_E_MEV;
    double eps_min = 1.0 / (1.0 + 2.0 * k);
    double eps;
    /* Kahn rejection sampling */
    for (;;) {
        double r1 = rng_double(), r2 = rng_double(), r3 = rng_double();
        if (r1 < (2.0 * k + 1.0) / (4.0 * k + 1.0 + 2.0 * k * k)) {
            /* ramo superiore */
            eps = 1.0 - rng_double() * (1.0 - eps_min);
            if (r2 <= 1.0 - (1.0 - 1.0/eps) / k * (2.0 - (1.0 - 1.0/eps) / k)) continue;
        } else {
            /* ramo inferiore */
            eps = eps_min + rng_double() * (1.0 - eps_min);
            double cos_th = 1.0 + 1.0 / k - 1.0 / (k * eps);
            double sin2th = 1.0 - cos_th * cos_th;
            if (r3 > eps * (2.0 - sin2th)) continue;
        }
        break;
    }
    *costh = 1.0 + 1.0 / k - 1.0 / (k * eps);
    if (*costh < -1.0) *costh = -1.0;
    if (*costh >  1.0) *costh =  1.0;
    return E_in * eps;
}

/* ── ELETTRONI ────────────────────────────────
 *
 * Stopping power di Bethe-Bloch (formula non-relativistica + correzione relativistica):
 *
 *   -dE/dx = (4π e^4 N_e) / (m_e v^2) * [ln(2 m_e v^2 / I) - ln(1-β^2) - β^2]
 *
 * In unità MeV/cm, con fattore di Shell correction trascurato:
 *
 *   -dE/dx = 0.3071 * (Z/A) * ρ / β^2 * [ln(T^2(T+2)/(2I^2)) - (2/√(1+τ)-1+β^2)*ln2 + ... ]
 *
 * Usiamo la forma semplificata ma corretta nell'ordine di grandezza:
 */
static double bethe_bloch_MeV_per_cm(MaterialID mid, double E_MeV) {
    const Material *m = &MATERIALS[mid];
    if (m->density <= 0.0 || m->Z_eff <= 0.0) return 0.0;
    double T    = E_MeV;                        /* energia cinetica (MeV)           */
    double tau  = T / M_E_MEV;                  /* in unità di m_e c^2              */
    double beta2= tau * (tau + 2.0) / ((tau + 1.0) * (tau + 1.0));
    if (beta2 <= 0.0) beta2 = 1e-10;
    double I_MeV= m->I_eV * 1e-6;              /* potenziale di ionizzazione (MeV) */
    /* Parametro di Bloch */
    double arg  = 2.0 * M_E_MEV * beta2 * tau * (tau + 2.0) / (I_MeV * I_MeV);
    if (arg <= 1.0) arg = 1.001;
    /* Fattore di densità di Sternheimer (approssimazione) */
    double delta = 0.0;
    if (beta2 > 0.01) delta = 2.0 * log(sqrt(beta2 / (1.0 - beta2))) - 1.0;
    if (delta < 0.0) delta = 0.0;
    /* Costante K = 4π N_A r_e^2 m_e c^2 = 0.307075 MeV cm^2 / g */
    double K = 0.307075;
    double stop_power = K * (m->Z_eff / m->A_eff) * m->density / beta2
                       * (0.5 * log(arg) - beta2 - delta / 2.0);
    return fmax(stop_power, 0.0);
}

/*
 * Scattering angolare multiplo di Molière (approssimazione Highland):
 * σ_θ = (13.6 MeV / (β p)) * √(x/X_0) * [1 + 0.038 ln(x/X_0)]
 * dove X_0 è la lunghezza di radiazione.
 *
 * Lunghezza di radiazione approssimata (Dahl):
 * X_0 ≈ 716.4 * A / (Z*(Z+1) * ln(287/√Z)) g/cm^2
 */
static double radiation_length_g_cm2(MaterialID mid) {
    const Material *m = &MATERIALS[mid];
    if (m->Z_eff < 1.0) return 1e30;
    double num  = 716.4 * m->A_eff;
    double den  = m->Z_eff * (m->Z_eff + 1.0) * log(287.0 / sqrt(m->Z_eff));
    return (den > 0.0) ? num / den : 1e30;
}

static double moliere_rms_angle(MaterialID mid, double E_MeV, double step_cm) {
    const Material *m = &MATERIALS[mid];
    if (m->density <= 0.0) return 0.0;
    double X0_cm  = radiation_length_g_cm2(mid) / fmax(m->density, 1e-10);
    double x_frac = step_cm / X0_cm;
    if (x_frac <= 0.0) return 0.0;
    /* p*beta ≈ sqrt(E^2 + 2*E*m_e) per elettroni relativistici */
    double pbeta  = sqrt(E_MeV * (E_MeV + 2.0 * M_E_MEV));
    double sigma  = (13.6 / pbeta) * sqrt(x_frac) * (1.0 + 0.038 * log(x_frac));
    return sigma;
}

/*
 * Probabilità di emissione Bremsstrahlung per step dx:
 * P_brem ≈ (4/3) * (dx/X_0) * [ln(2E/m_e) - 1/3]
 * L'energia del fotone emesso segue la distribuzione ∝ 1/k (a bassa energia).
 */
static double bremsstrahlung_prob(MaterialID mid, double E_MeV, double step_cm) {
    double X0_cm = radiation_length_g_cm2(mid) / fmax(MATERIALS[mid].density, 1e-10);
    if (X0_cm <= 0.0 || E_MeV <= 0.0) return 0.0;
    double tau  = E_MeV / M_E_MEV;
    double lnfact = log(2.0 * tau) - 1.0 / 3.0;
    if (lnfact < 0.0) lnfact = 0.0;
    return (4.0 / 3.0) * (step_cm / X0_cm) * lnfact;
}

/* ─────────────────────────────────────────────
   CODA SECONDARI
   ───────────────────────────────────────────── */

static void push_secondary(double x, double y, double dx, double dy,
                            double energy, ParticleType type, double weight,
                            int generation) {
    if (n_sec >= SECONDARY_QUEUE || energy < E_CUTOFF_PHOTON) return;
    Particle *p = &sec_queue[n_sec++];
    p->x = x; p->y = y; p->dx = dx; p->dy = dy;
    p->energy = energy; p->type = type; p->weight = weight;
    p->alive = 1; p->generation = generation;
    p->mat_idx = (uint8_t)get_material(x, y);
}

/* ─────────────────────────────────────────────
   TRACCIA: aggiunge un punto
   ───────────────────────────────────────────── */

static void track_add_point(Track *t, double x, double y, double e) {
    if (!t || t->n_pts >= MAX_TRACK_PTS) return;
    t->px[t->n_pts] = (float)x;
    t->py[t->n_pts] = (float)y;
    t->pe[t->n_pts] = (float)e;
    t->n_pts++;
}

/* forward declaration */
static inline MaterialID mid_for(Particle *p);

/* ─────────────────────────────────────────────
   TRASPORTO DI UN NEUTRONE
   ───────────────────────────────────────────── */

static void transport_neutron(Particle *p, Track *trk) {
    while (p->alive && p->energy > E_CUTOFF_NEUT) {
        MaterialID mid = get_material(p->x, p->y);
        if (mid == MAT_VACUUM && !in_domain(p->x, p->y)) { p->alive = 0; break; }

        double sig_s, sig_c, sig_f;
        neutron_xs(mid, p->energy, &sig_s, &sig_c, &sig_f);
        double sig_tot = sig_s + sig_c + sig_f;

        /* Sampling del cammino libero medio: d = -ln(ξ)/Σ_tot */
        double step;
        if (sig_tot < 1e-20) {
            /* nel vuoto o materiale trasparente: propaga fino al bordo */
            step = 1.0; /* 1 cm fisso nel vuoto */
        } else {
            step = -log(rng_double() + 1e-300) / sig_tot;
        }
        /* Clamp al bordo del dominio */
        step = fmin(step, 10.0);

        double new_x = p->x + p->dx * step;
        double new_y = p->y + p->dy * step;

        /* Deposita fluenza lungo il percorso */
        deposit_fluence((p->x + new_x) / 2.0, (p->y + new_y) / 2.0, PTYPE_NEUTRON, p->weight);
        track_add_point(trk, new_x, new_y, p->energy);

        p->x = new_x; p->y = new_y;

        if (!in_domain(p->x, p->y)) { p->alive = 0; break; }

        if (sig_tot < 1e-20) continue; /* vuoto */

        /* Selezione dell'interazione tramite roulette russa */
        double r = rng_double() * sig_tot;
        if (r < sig_s) {
            /* ── SCATTERING ELASTICO ── */
            double A = MATERIALS[mid].A_eff;
            double costh_lab;
            p->energy = neutron_elastic_scatter(p->energy, A, &costh_lab);
            double sinth_lab = sqrt(fmax(0.0, 1.0 - costh_lab * costh_lab));
            rotate_direction(&p->dx, &p->dy, costh_lab, sinth_lab);
            total_scatterings++;

        } else if (r < sig_s + sig_c) {
            /* ── CATTURA RADIATIVA ── */
            /* Il nucleo emette fotoni gamma (approssimazione: un singolo fotone) */
            double E_gamma = p->energy + 6.0; /* approssimazione: Q di cattura ~6 MeV */
            double gdx, gdy;
            sample_isotropic(&gdx, &gdy);
            push_secondary(p->x, p->y, gdx, gdy, E_gamma, PTYPE_PHOTON,
                           p->weight, p->generation + 1);
            /* Deposita energia di rinculo del nucleo come dose */
            deposit_dose(p->x, p->y, p->energy, MATERIALS[mid].density);
            p->alive = 0;
            total_captures++;

        } else {
            /* ── FISSIONE ── */
            /* Emette nu_fiss neutroni veloci (spettro di Watt) + fotoni prompt */
            double nu  = MATERIALS[mid].nu_fiss;
            int    n_nu = (int)nu + (rng_double() < (nu - (int)nu) ? 1 : 0);
            double E_release = MATERIALS[mid].E_fiss_MeV;

            for (int i = 0; i < n_nu; i++) {
                /* Spettro di fissione di Watt: P(E) ∝ sinh(√(2E)) exp(-E/1.0292) */
                /* Approssimazione: Maxwell con T=1.33 MeV */
                double E_watt;
                do {
                    double u = -log(rng_double() + 1e-300);
                    double v = -log(rng_double() + 1e-300);
                    double w = rng_double();
                    double c = u + 1.0;
                    if (w * w <= sin(PI * u / 2.0 / c) * sin(PI * u / 2.0 / c) + 1.0
                                 - 1.0 / c) {
                        E_watt = 1.33 * (v + u * 0.5);
                        break;
                    }
                    E_watt = 1.33 * v;
                    break;
                } while(0);
                E_watt = fmax(E_watt, 0.01);
                double ndx, ndy;
                sample_isotropic(&ndx, &ndy);
                push_secondary(p->x, p->y, ndx, ndy, E_watt, PTYPE_NEUTRON,
                               p->weight, p->generation + 1);
            }
            /* Fotoni prompt di fissione: ~7 fotoni con <E>=1 MeV */
            int n_gamma = 7;
            for (int i = 0; i < n_gamma; i++) {
                double Eg = -log(rng_double() + 1e-300) * 1.0;
                double gdx, gdy;
                sample_isotropic(&gdx, &gdy);
                push_secondary(p->x, p->y, gdx, gdy, Eg, PTYPE_PHOTON,
                               p->weight / n_gamma, p->generation + 1);
            }
            deposit_dose(p->x, p->y, E_release * 0.03, MATERIALS[mid].density);
            p->alive = 0;
            total_fissions++;
        }
    }
    if (p->energy <= E_CUTOFF_NEUT && p->alive) {
        /* Neutrone termico catturato */
        deposit_dose(p->x, p->y, p->energy, MATERIALS[mid_for(p)].density);
        p->alive = 0;
    }
}

/* helper per ottenere MaterialID dalla posizione corrente */
static inline MaterialID mid_for(Particle *p) {
    return get_material(p->x, p->y);
}

/* ─────────────────────────────────────────────
   TRASPORTO DI UN FOTONE
   ───────────────────────────────────────────── */

static void transport_photon(Particle *p, Track *trk) {
    while (p->alive && p->energy > E_CUTOFF_PHOTON) {
        MaterialID mid = get_material(p->x, p->y);
        if (mid == MAT_VACUUM && !in_domain(p->x, p->y)) { p->alive = 0; break; }

        double mu_pe, mu_c, mu_pp, mu_tot;
        photon_xs(mid, p->energy, &mu_pe, &mu_c, &mu_pp, &mu_tot);

        double step;
        if (mu_tot < 1e-20) step = 1.0;
        else step = -log(rng_double() + 1e-300) / mu_tot;
        step = fmin(step, 10.0);

        double new_x = p->x + p->dx * step;
        double new_y = p->y + p->dy * step;

        deposit_fluence((p->x + new_x) / 2.0, (p->y + new_y) / 2.0, PTYPE_PHOTON, p->weight);
        track_add_point(trk, new_x, new_y, p->energy);

        p->x = new_x; p->y = new_y;
        if (!in_domain(p->x, p->y)) { p->alive = 0; break; }
        if (mu_tot < 1e-20) continue;

        double r = rng_double() * mu_tot;

        if (r < mu_pe) {
            /* ── EFFETTO FOTOELETTRICO ── */
            /* L'elettrone viene emesso con E = hν - E_binding (appross. E_binding piccola) */
            double E_e = p->energy - MATERIALS[mid].I_eV * 1e-6 * MATERIALS[mid].Z_eff;
            if (E_e > E_CUTOFF_ELEC) {
                double edx, edy;
                sample_isotropic(&edx, &edy);
                push_secondary(p->x, p->y, edx, edy, E_e, PTYPE_ELECTRON,
                               p->weight, p->generation + 1);
            }
            deposit_dose(p->x, p->y, MATERIALS[mid].I_eV * 1e-6 * MATERIALS[mid].Z_eff,
                         MATERIALS[mid].density);
            p->alive = 0;

        } else if (r < mu_pe + mu_c) {
            /* ── SCATTERING COMPTON ── */
            double costh;
            double E_out = compton_scatter(p->energy, &costh);
            double sinth = sqrt(fmax(0.0, 1.0 - costh * costh));
            rotate_direction(&p->dx, &p->dy, costh, sinth);
            /* Elettrone di rinculo */
            double E_e = p->energy - E_out;
            p->energy  = E_out;
            if (E_e > E_CUTOFF_ELEC) {
                /* Direzione dell'elettrone di rinculo (approssimazione) */
                double edx = -(p->dy * sinth), edy = (p->dx * sinth);
                double enorm = sqrt(edx*edx + edy*edy);
                if (enorm > 1e-12) { edx /= enorm; edy /= enorm; }
                else sample_isotropic(&edx, &edy);
                push_secondary(p->x, p->y, edx, edy, E_e, PTYPE_ELECTRON,
                               p->weight, p->generation + 1);
            }

        } else {
            /* ── PRODUZIONE DI COPPIE ── */
            double E_avail = p->energy - E_PAIR_THRESH;
            /* Ripartizione energetica: approssimazione 50-50 */
            double E_e = E_avail * rng_double();
            double E_p = E_avail - E_e;
            double edx, edy, pdx, pdy;
            sample_isotropic(&edx, &edy);
            pdx = -edx; pdy = -edy; /* approssimazione: back-to-back nel lab */
            if (E_e > E_CUTOFF_ELEC)
                push_secondary(p->x, p->y, edx, edy, E_e, PTYPE_ELECTRON,
                               p->weight, p->generation + 1);
            if (E_p > E_CUTOFF_ELEC)
                push_secondary(p->x, p->y, pdx, pdy, E_p, PTYPE_POSITRON,
                               p->weight, p->generation + 1);
            deposit_dose(p->x, p->y, E_PAIR_THRESH, MATERIALS[mid].density);
            p->alive = 0;
        }
    }
}

/* ─────────────────────────────────────────────
   TRASPORTO DI UN ELETTRONE / POSITRONE
   ───────────────────────────────────────────── */

static void transport_electron(Particle *p, Track *trk) {
    double STEP_CM = 0.05; /* step fisso per condensed history */
    while (p->alive && p->energy > E_CUTOFF_ELEC) {
        MaterialID mid = get_material(p->x, p->y);
        if (!in_domain(p->x, p->y)) { p->alive = 0; break; }

        double dE_dx = bethe_bloch_MeV_per_cm(mid, p->energy);
        double step  = fmin(STEP_CM, p->energy / (dE_dx + 1e-30) * 0.5);
        if (step < 1e-6) { p->alive = 0; break; }

        /* Angolo di deflessione multiplo di Molière */
        double sigma_theta = moliere_rms_angle(mid, p->energy, step);
        if (sigma_theta > 0.0) {
            /* Gaussiana approssimata: campionamento Box-Muller */
            double u1 = rng_double() + 1e-300, u2 = rng_double();
            double theta = sigma_theta * sqrt(-2.0 * log(u1)) * cos(TWO_PI * u2);
            rotate_direction(&p->dx, &p->dy, cos(theta), sin(fabs(theta)));
        }

        double new_x = p->x + p->dx * step;
        double new_y = p->y + p->dy * step;

        deposit_fluence((p->x + new_x) / 2.0, (p->y + new_y) / 2.0, PTYPE_ELECTRON, p->weight);
        track_add_point(trk, new_x, new_y, p->energy);

        /* Perdita di energia continua */
        double dE = dE_dx * step;
        dE = fmin(dE, p->energy);
        deposit_dose(p->x, p->y, dE * p->weight, MATERIALS[mid].density);
        p->energy -= dE;

        /* Bremsstrahlung */
        double p_brem = bremsstrahlung_prob(mid, p->energy, step);
        if (rng_double() < p_brem && p->energy > 0.01) {
            /* Energia del fotone: campionamento ∝ 1/k con k ∈ [E_cutoff, E_max] */
            double k_max  = p->energy * 0.5;
            double k_min  = E_CUTOFF_PHOTON;
            double E_brem = k_min * exp(rng_double() * log(k_max / k_min));
            E_brem = fmin(E_brem, p->energy * 0.5);
            double gdx, gdy;
            gdx = p->dx; gdy = p->dy; /* approssimazione: in avanti */
            push_secondary(p->x, p->y, gdx, gdy, E_brem, PTYPE_PHOTON,
                           p->weight, p->generation + 1);
            p->energy -= E_brem;
        }

        p->x = new_x; p->y = new_y;

        /* Annichilazione positrone a riposo */
        if (p->type == PTYPE_POSITRON && p->energy < 0.01) {
            /* Due fotoni da 511 keV back-to-back */
            double gdx1 = p->dx, gdy1 = p->dy;
            push_secondary(p->x, p->y,  gdx1,  gdy1, 0.511, PTYPE_PHOTON, p->weight, p->generation + 1);
            push_secondary(p->x, p->y, -gdx1, -gdy1, 0.511, PTYPE_PHOTON, p->weight, p->generation + 1);
            p->alive = 0;
        }
    }
}

/* ─────────────────────────────────────────────
   DISPATCH TRASPORTO
   ───────────────────────────────────────────── */

static void transport_particle(Particle *p, Track *trk) {
    switch (p->type) {
        case PTYPE_NEUTRON:  transport_neutron(p, trk);  break;
        case PTYPE_PHOTON:   transport_photon(p, trk);   break;
        case PTYPE_ELECTRON:
        case PTYPE_POSITRON: transport_electron(p, trk); break;
    }
}

/* ─────────────────────────────────────────────
   API PUBBLICA
   ───────────────────────────────────────────── */

EXPORT void sim_init(void) {
    rng_seed(12345678901234ULL);
    memset(mat_grid,   0, sizeof(mat_grid));
    memset(fluence_n,  0, sizeof(fluence_n));
    memset(fluence_p,  0, sizeof(fluence_p));
    memset(fluence_e,  0, sizeof(fluence_e));
    memset(dose_map,   0, sizeof(dose_map));
    memset(particles,  0, sizeof(particles));
    memset(tracks,     0, sizeof(tracks));
    n_tracks = 0; track_write_idx = 0; n_sec = 0;
    total_histories = 0; total_fissions = 0;
    total_captures  = 0; total_scatterings = 0;

    /* Geometria default:
     *   sfondo = acqua
     *   blocco di piombo (schermatura) a destra
     *   nucleo di U-235 al centro-sinistra
     */
    for (int y = 0; y < GRID_H; y++)
        for (int x = 0; x < GRID_W; x++)
            mat_grid[y][x] = MAT_WATER;

    /* Blocco di Pb: x in [120,160], y in [60,140] */
    for (int y = 60; y < 140; y++)
        for (int x = 120; x < 160; x++)
            mat_grid[y][x] = MAT_LEAD;

    /* Nucleo U-235: centro (60,100), raggio 15 celle */
    int cx = 60, cy = 100, rad = 15;
    for (int y = cy - rad; y <= cy + rad; y++)
        for (int x = cx - rad; x <= cx + rad; x++)
            if ((x-cx)*(x-cx)+(y-cy)*(y-cy) <= rad*rad)
                mat_grid[y][x] = MAT_U235;

    /* Sorgente default: neutroni veloci, isotropici, al centro */
    src_x = DOMAIN_W_CM / 2.0;
    src_y = DOMAIN_H_CM / 2.0;
    src_energy   = 2.0;        /* 2 MeV */
    src_type     = PTYPE_NEUTRON;
    src_isotropic = 1;
    src_dx = 1.0; src_dy = 0.0;
}

EXPORT void sim_reset(void) {
    sim_init();
}

EXPORT void set_source(double x_cm, double y_cm, double energy_MeV,
                        int particle_type, int isotropic,
                        double dir_x, double dir_y) {
    src_x = x_cm; src_y = y_cm;
    src_energy    = energy_MeV;
    src_type      = (ParticleType)particle_type;
    src_isotropic = isotropic;
    double norm   = sqrt(dir_x*dir_x + dir_y*dir_y);
    if (norm > 1e-12) { src_dx = dir_x/norm; src_dy = dir_y/norm; }
    else { src_dx = 1.0; src_dy = 0.0; }
}

EXPORT void set_geometry(int cell_x, int cell_y, int material_id) {
    if (cell_x < 0 || cell_x >= GRID_W || cell_y < 0 || cell_y >= GRID_H) return;
    if (material_id < 0 || material_id >= MAT_COUNT) return;
    mat_grid[cell_y][cell_x] = (uint8_t)material_id;
}

/*
 * sim_step: esegue N storie primarie e le cascate secondarie associate.
 * Ritorna il numero di storie completate.
 */
EXPORT int sim_step(int n_primaries) {
    n_tracks = 0; /* azzera tracce per questo frame */

    for (int h = 0; h < n_primaries; h++) {
        /* Crea particella primaria */
        Particle primary;
        memset(&primary, 0, sizeof(primary));
        primary.x = src_x; primary.y = src_y;
        if (src_isotropic) sample_isotropic(&primary.dx, &primary.dy);
        else { primary.dx = src_dx; primary.dy = src_dy; }
        primary.energy = src_energy;
        primary.type   = src_type;
        primary.weight = 1.0;
        primary.alive  = 1;
        primary.generation = 0;

        n_sec = 0;

        /* Alloca traccia */
        Track *trk = NULL;
        if (track_write_idx < MAX_TRACKS) {
            trk = &tracks[track_write_idx];
            memset(trk, 0, sizeof(Track));
            trk->type = src_type;
            trk->generation = 0;
            track_add_point(trk, primary.x, primary.y, primary.energy);
            track_write_idx = (track_write_idx + 1) % MAX_TRACKS;
            n_tracks++;
        }

        transport_particle(&primary, trk);

        /* Processa secondari in cascata (BFS) */
        int sec_start = 0;
        while (sec_start < n_sec) {
            int sec_end = n_sec;
            for (int si = sec_start; si < sec_end && si < SECONDARY_QUEUE; si++) {
                Particle *sp = &sec_queue[si];
                Track *strk = NULL;
                if (track_write_idx < MAX_TRACKS) {
                    strk = &tracks[track_write_idx];
                    memset(strk, 0, sizeof(Track));
                    strk->type = sp->type;
                    strk->generation = sp->generation;
                    track_add_point(strk, sp->x, sp->y, sp->energy);
                    track_write_idx = (track_write_idx + 1) % MAX_TRACKS;
                    n_tracks++;
                }
                transport_particle(sp, strk);
            }
            sec_start = sec_end;
        }

        total_histories++;
    }
    track_write_idx = 0; /* reset per il prossimo frame */
    return n_primaries;
}

/* ─────────────────────────────────────────────
   GETTER PER BUFFER FLAT (per JS via WASM)
   ───────────────────────────────────────────── */

EXPORT float* get_fluence_n(void) { return &fluence_n[0][0]; }
EXPORT float* get_fluence_p(void) { return &fluence_p[0][0]; }
EXPORT float* get_fluence_e(void) { return &fluence_e[0][0]; }
EXPORT float* get_dose(void)      { return &dose_map[0][0];  }
EXPORT uint8_t* get_mat_grid(void){ return &mat_grid[0][0];  }

EXPORT int get_grid_w(void) { return GRID_W; }
EXPORT int get_grid_h(void) { return GRID_H; }
EXPORT int get_track_count(void) { return n_tracks; }

/* Ritorna array flat di tracce:
 * per ogni traccia: [type(f), n_pts(f), px0,py0,pe0, px1,py1,pe1, ..., (MAX_TRACK_PTS*3 floats)]
 * stride totale per traccia: 2 + MAX_TRACK_PTS*3 float
 */
EXPORT float* get_tracks_buffer(void) {
    static float tbuf[MAX_TRACKS * (2 + MAX_TRACK_PTS * 3)];
    int stride = 2 + MAX_TRACK_PTS * 3;
    for (int i = 0; i < n_tracks && i < MAX_TRACKS; i++) {
        Track *t = &tracks[i];
        float *tb = &tbuf[i * stride];
        tb[0] = (float)t->type;
        tb[1] = (float)t->n_pts;
        for (int j = 0; j < t->n_pts && j < MAX_TRACK_PTS; j++) {
            tb[2 + j*3 + 0] = t->px[j];
            tb[2 + j*3 + 1] = t->py[j];
            tb[2 + j*3 + 2] = t->pe[j];
        }
    }
    return tbuf;
}

EXPORT int get_track_stride(void) { return 2 + MAX_TRACK_PTS * 3; }

EXPORT long get_total_histories(void)   { return total_histories;   }
EXPORT long get_total_fissions(void)    { return total_fissions;     }
EXPORT long get_total_captures(void)    { return total_captures;     }
EXPORT long get_total_scatterings(void) { return total_scatterings;  }

/* ─────────────────────────────────────────────
   MAIN (solo per test nativo, non usato in WASM)
   ───────────────────────────────────────────── */

#ifndef __EMSCRIPTEN__
int main(void) {
    sim_init();
    for (int i = 0; i < 100; i++) sim_step(10);
    printf("Histories: %ld  Fissions: %ld  Captures: %ld  Scatterings: %ld\n",
           total_histories, total_fissions, total_captures, total_scatterings);
    return 0;
}
#endif
