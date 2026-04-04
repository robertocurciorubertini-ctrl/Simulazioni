/*
 * ============================================================
 *  KERR SPACETIME SIMULATOR — versione Emscripten/WebAssembly
 *
 *  Fisica: Metrica di Kerr in coordinate Boyer-Lindquist
 *  Unità geometrizzate: G = c = M = 1, r in unità di r_g
 *
 *  Rendering: Canvas HTML5 via emscripten_set_main_loop()
 *  Buffer pixel: RGBA 32bpp scritto direttamente in memoria
 *  e trasferito al canvas tramite CanvasRenderingContext2D.putImageData
 *
 *  Compilare con:
 *    emcc kerr.c -o kerr.html \
 *         -O3 -msimd128 \
 *         -s WASM=1 \
 *         -s USE_SDL=0 \
 *         -s EXPORTED_FUNCTIONS='["_main","_set_spin","_set_zoom","_set_mode","_set_nrays","_set_remit","_step_anim"]' \
 *         -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap"]' \
 *         -s ALLOW_MEMORY_GROWTH=1 \
 *         --shell-file shell.html
 * ============================================================
 */

#include <emscripten.h>
#include <emscripten/html5.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>

/* ------------------------------------------------------------------ */
/*  CONFIGURAZIONE                                                      */
/* ------------------------------------------------------------------ */
#define WIDTH        900
#define HEIGHT       700
#define PI           3.14159265358979323846
#define TWO_PI       6.28318530717958647692
#define MAX_STEPS    6000
#define MAX_PATH     5000
#define N_RAYS_MAX   120

typedef enum {
    MODE_GRID     = 0,
    MODE_GEODESIC = 1,
    MODE_LENS     = 2,
    MODE_REDSHIFT = 3,
    MODE_COUNT    = 4
} RenderMode;

/* ------------------------------------------------------------------ */
/*  STRUTTURE                                                           */
/* ------------------------------------------------------------------ */
typedef struct { uint8_t r, g, b, a; } Color;

typedef struct {
    double gtt, gtphi, grr, gphiphi;
    double Delta, Sigma, det;
} KerrMetric;

typedef struct {
    double rp, r_ergo, r_isco, r_ph, Omega_H;
} KerrProps;

typedef struct {
    double r[MAX_PATH];
    double phi[MAX_PATH];
    int    n, captured, escaped;
} GeodesicPath;

/* ------------------------------------------------------------------ */
/*  STATO GLOBALE                                                       */
/* ------------------------------------------------------------------ */
static uint8_t  pixels[WIDTH * HEIGHT * 4];   /* RGBA */
static double   g_a       = 0.60;             /* spin */
static double   g_zoom    = 1.0;
static int      g_nrays   = 40;
static double   g_remit   = 18.0;
static RenderMode g_mode  = MODE_GRID;
static double   g_phi_off = 0.0;
static int      g_animating = 0;
static KerrProps g_kp;

/* ------------------------------------------------------------------ */
/*  METRICA DI KERR (piano equatoriale θ=π/2, M=1)                    */
/* ------------------------------------------------------------------ */
static inline void kerr_metric(double r, double a, KerrMetric *m)
{
    double r2    = r * r;
    double a2    = a * a;
    m->Sigma     = r2;
    m->Delta     = r2 - 2.0*r + a2;
    m->gtt       = -(1.0 - 2.0*r/m->Sigma);
    m->gtphi     =  2.0*a*r/m->Sigma;
    m->grr       =  m->Sigma / m->Delta;
    m->gphiphi   =  r2 + a2 + 2.0*a2*r/m->Sigma;
    m->det       =  m->gtt * m->gphiphi - m->gtphi * m->gtphi;
}

/* ------------------------------------------------------------------ */
/*  QUANTITÀ FISICHE                                                    */
/* ------------------------------------------------------------------ */
static void kerr_props(double a, KerrProps *p)
{
    double sqrd  = sqrt(fmax(0.0, 1.0 - a*a));
    p->rp        = 1.0 + sqrd;
    p->r_ergo    = 2.0;

    double a2    = a*a;
    double cb    = cbrt(1.0 - a2);
    double z1    = 1.0 + cb*(cbrt(1.0+a) + cbrt(1.0-a));
    double z2    = sqrt(3.0*a2 + z1*z1);
    p->r_isco    = 3.0 + z2 - sqrt((3.0-z1)*(3.0+z1+2.0*z2));

    double arg   = (fabs(a) < 1e-10) ? 0.0 : -a;
    p->r_ph      = 2.0*(1.0 + cos(2.0/3.0 * acos(fmax(-1.0,fmin(1.0,arg)))));
    p->Omega_H   = a / (2.0 * p->rp);
}

/* ------------------------------------------------------------------ */
/*  POTENZIALE EFFETTIVO E DERIVATE                                     */
/* ------------------------------------------------------------------ */
static inline double Veff(double r, double a, double E, double L)
{
    KerrMetric m;
    kerr_metric(r, a, &m);
    return (E*E*m.gphiphi + 2.0*E*L*m.gtphi + L*L*m.gtt) / m.det;
}

static void geodesic_derivs(
    double r, double dr, double a, double E, double L,
    double *dr_out, double *ddr_out)
{
    KerrMetric m;
    kerr_metric(r, a, &m);

    const double eps = r * 5e-6;
    double Vp = Veff(r+eps, a, E, L);
    double Vm = Veff(r-eps, a, E, L);
    double dVdr = (Vp - Vm) / (2.0*eps);

    KerrMetric mp, mm;
    kerr_metric(r+eps, a, &mp);
    kerr_metric(r-eps, a, &mm);
    double dgrr = (mp.grr - mm.grr) / (2.0*eps);

    *dr_out  = dr;
    *ddr_out = (dVdr - dr*dr*dgrr) / (2.0*m.grr);
}

static inline double dphi_from_EL(double r, double a, double E, double L)
{
    KerrMetric m;
    kerr_metric(r, a, &m);
    return (E*m.gtphi + L*m.gtt) / m.det;
}

/* ------------------------------------------------------------------ */
/*  RK4 A PASSO ADATTIVO                                               */
/* ------------------------------------------------------------------ */
static void rk4_step(
    double *r, double *phi, double *dr,
    double a, double E, double L, double h)
{
    double k1r, k1dr, k2r, k2dr, k3r, k3dr, k4r, k4dr;
    double dp;

    geodesic_derivs(*r,          *dr,          a,E,L, &k1r,&k1dr);
    geodesic_derivs(*r+h*k1r/2, *dr+h*k1dr/2, a,E,L, &k2r,&k2dr);
    geodesic_derivs(*r+h*k2r/2, *dr+h*k2dr/2, a,E,L, &k3r,&k3dr);
    geodesic_derivs(*r+h*k3r,   *dr+h*k3dr,   a,E,L, &k4r,&k4dr);

    double r_new  = *r  + h*(k1r +2*k2r +2*k3r +k4r )/6.0;
    double dr_new = *dr + h*(k1dr+2*k2dr+2*k3dr+k4dr)/6.0;

    /* Aggiornamento phi con phi' medio lungo il passo */
    double dphi_mid = dphi_from_EL(*r + h*0.5*(k1r+k2r)/2.0, a, E, L);
    *phi += h * dphi_mid;
    *r   = r_new;
    *dr  = dr_new;
}

static void rk4_adaptive(
    double *r, double *phi, double *dr,
    double a, double E, double L,
    double *h, double h_min, double h_max, double tol)
{
    double r1=*r, p1=*phi, d1=*dr;
    double r2=*r, p2=*phi, d2=*dr;

    rk4_step(&r1,&p1,&d1, a,E,L, *h);

    double hh=*h*0.5;
    rk4_step(&r2,&p2,&d2, a,E,L, hh);
    rk4_step(&r2,&p2,&d2, a,E,L, hh);

    double err = fabs(r1-r2);
    if (err < 1e-15) err=1e-15;

    double factor = 0.9*pow(tol/err, 0.2);
    factor = fmax(0.1, fmin(5.0, factor));
    *h = fmax(h_min, fmin(h_max, (*h)*factor));

    if (err < tol || *h <= h_min) {
        *r=r1; *phi=p1; *dr=d1;
    }
}

/* ------------------------------------------------------------------ */
/*  INTEGRAZIONE GEODETICA                                             */
/* ------------------------------------------------------------------ */
static void integrate_geodesic(
    double r0, double phi0, double E, double L, double a,
    double r_max, double r_cap, GeodesicPath *path)
{
    double r=r0, phi=phi0;
    double V0=Veff(r,a,E,L);
    double dr = (V0>=0.0) ? -sqrt(V0) : 0.0;

    path->n=0; path->captured=0; path->escaped=0;

    double h=0.06, h_min=1e-5, h_max=0.15, tol=1e-7;

    for (int step=0; step<MAX_STEPS && path->n<MAX_PATH-1; step++) {
        if (r < r_cap)    { path->captured=1; break; }
        if (r > r_max)    { path->escaped=1;  break; }
        if (!isfinite(r)) break;

        path->r[path->n]   = r;
        path->phi[path->n] = phi;
        path->n++;

        rk4_adaptive(&r,&phi,&dr, a,E,L, &h, h_min,h_max, tol);

        double Vn = Veff(r,a,E,L);
        if (dr<0.0 && Vn<0.0) dr=fabs(dr);
    }
}

/* ------------------------------------------------------------------ */
/*  PIXEL BUFFER                                                        */
/* ------------------------------------------------------------------ */
static inline void put_pixel(int x, int y, uint8_t r, uint8_t g, uint8_t b)
{
    if ((unsigned)x>=(unsigned)WIDTH || (unsigned)y>=(unsigned)HEIGHT) return;
    int i=(y*WIDTH+x)*4;
    pixels[i]=r; pixels[i+1]=g; pixels[i+2]=b; pixels[i+3]=255;
}

static void draw_line(int x0,int y0,int x1,int y1,
                      uint8_t r,uint8_t g,uint8_t b)
{
    int dx=abs(x1-x0),dy=abs(y1-y0);
    int sx=x0<x1?1:-1, sy=y0<y1?1:-1, err=dx-dy;
    while(1){
        put_pixel(x0,y0,r,g,b);
        if(x0==x1&&y0==y1) break;
        int e2=2*err;
        if(e2>-dy){err-=dy;x0+=sx;}
        if(e2< dx){err+=dx;y0+=sy;}
    }
}

static void draw_circle(int cx,int cy,int radius,
                        uint8_t r,uint8_t g,uint8_t b,int dashed)
{
    if(radius<=0) return;
    int x=radius,y=0,err=0,dash=0;
    while(x>=y){
        if(!dashed||(dash/3)%2==0){
            put_pixel(cx+x,cy+y,r,g,b); put_pixel(cx+y,cy+x,r,g,b);
            put_pixel(cx-y,cy+x,r,g,b); put_pixel(cx-x,cy+y,r,g,b);
            put_pixel(cx-x,cy-y,r,g,b); put_pixel(cx-y,cy-x,r,g,b);
            put_pixel(cx+y,cy-x,r,g,b); put_pixel(cx+x,cy-y,r,g,b);
        }
        dash++;
        if(err<=0){y++;err+=2*y+1;}
        else      {x--;err-=2*x+1;}
    }
}

static void fill_circle(int cx,int cy,int radius,
                        uint8_t r,uint8_t g,uint8_t b)
{
    for(int dy=-radius;dy<=radius;dy++)
    for(int dx=-radius;dx<=radius;dx++)
        if(dx*dx+dy*dy<=radius*radius)
            put_pixel(cx+dx,cy+dy,r,g,b);
}

/* ------------------------------------------------------------------ */
/*  COORDINATE                                                          */
/* ------------------------------------------------------------------ */
static inline void w2s(double wx,double wy,double cx,double cy,
                       double scale,int *sx,int *sy)
{
    *sx=(int)(cx+wx*scale);
    *sy=(int)(cy-wy*scale);
}

/* ------------------------------------------------------------------ */
/*  COLORE DA LUNGHEZZA D'ONDA (CIE approssimata)                      */
/* ------------------------------------------------------------------ */
static void wavelength_rgb(double lam,
                           uint8_t *R,uint8_t *G,uint8_t *B)
{
    double r,g,b;
    if      (lam<380){r=0.5;g=0;b=0.5;}
    else if (lam<440){r=(440-lam)/60.0;g=0;b=1;}
    else if (lam<490){r=0;g=(lam-440)/50.0;b=1;}
    else if (lam<510){r=0;g=1;b=(510-lam)/20.0;}
    else if (lam<580){r=(lam-510)/70.0;g=1;b=0;}
    else if (lam<645){r=1;g=(645-lam)/65.0;b=0;}
    else if (lam<750){r=1;g=0;b=0;}
    else             {r=0.5;g=0;b=0;}
    *R=(uint8_t)(pow(r,0.8)*235);
    *G=(uint8_t)(pow(g,0.8)*235);
    *B=(uint8_t)(pow(b,0.8)*235);
}

/* ------------------------------------------------------------------ */
/*  RENDER 0: GRIGLIA / EMBEDDING DI FLAMM                             */
/* ------------------------------------------------------------------ */
static void render_grid(void)
{
    /* Sfondo */
    for(int i=0;i<WIDTH*HEIGHT*4;i+=4){
        pixels[i]   = 12;
        pixels[i+1] = 9;
        pixels[i+2] = 6;
        pixels[i+3] = 255;
    }

    const double cx = WIDTH  / 2.0;
    const double cy = HEIGHT / 2.0;

    const double a     = g_a;
    const double r_min = g_kp.rp * 1.01;
    const double r_max = 24.0 / fmax(g_zoom, 0.35);

    const int Nr = 180;   /* campionamento radiale per embedding */
    const int Np = 96;    /* campionamento angolare */

    /* Inclinazione della camera per vista pseudo-3D */
    const double incl = 0.95;   /* rad */
    const double ci   = cos(incl);
    const double si   = sin(incl);

    /* Array campionati su r */
    double rtab[Nr+1];
    double rho[Nr+1];
    double grr[Nr+1];
    double ztab[Nr+1];

    /* --------------------------------------------------------------
       1) Costruzione di rho(r) = sqrt(g_phiphi)
       -------------------------------------------------------------- */
    for(int i=0;i<=Nr;i++){
        double r = r_min + (r_max-r_min) * (double)i / (double)Nr;
        KerrMetric m;
        kerr_metric(r, a, &m);

        rtab[i] = r;
        grr[i]  = (m.Delta > 1e-12) ? m.grr : 1e12;
        rho[i]  = sqrt(fmax(m.gphiphi, 1e-12));
    }

    /* --------------------------------------------------------------
       2) Integrazione numerica di z(r) dalla relazione:
          (dz/dr)^2 = g_rr - (d rho / dr)^2
       -------------------------------------------------------------- */
    ztab[0] = 0.0;
    for(int i=1;i<=Nr;i++){
        double dr = rtab[i] - rtab[i-1];

        double drho_dr;
        if(i < Nr){
            drho_dr = (rho[i+1] - rho[i-1]) / (rtab[i+1] - rtab[i-1]);
        } else {
            drho_dr = (rho[i] - rho[i-1]) / dr;
        }

        double arg = grr[i] - drho_dr * drho_dr;
        if(arg < 0.0) arg = 0.0;

        double dzdr = sqrt(arg);

        /* clamp leggero per evitare cuspidi numeriche troppo violente */
        if(dzdr > 8.0) dzdr = 8.0;

        ztab[i] = ztab[i-1] + dzdr * dr;
    }

    /* Normalizzazione verticale per tenerla visibile ma non eccessiva */
    double zmax = ztab[Nr];
    double zscale = (zmax > 1e-9) ? (0.42 / zmax) * rho[Nr] : 1.0;

    /* Scala finale in pixel */
    double outer_extent = rho[Nr];
    double scale = 0.42 * fmin((double)WIDTH, (double)HEIGHT) / fmax(outer_extent, 1.0);
    scale *= pow(g_zoom, 0.92);

    /* --------------------------------------------------------------
       Proiezione 3D -> 2D:
       X = rho cos(phi)
       Y = rho sin(phi)
       Z = z(r)

       Rotazione intorno all'asse x:
       x_screen = X
       y_screen = Y cos(i) - Z sin(i)
       -------------------------------------------------------------- */
    #define PROJECT_POINT(RHO_, PHI_, Z_, SX_, SY_) do {            \
        double _X = (RHO_) * cos(PHI_);                             \
        double _Y = (RHO_) * sin(PHI_);                             \
        double _Z = (Z_) * zscale;                                  \
        double _xp = _X;                                            \
        double _yp = _Y * ci - _Z * si;                             \
        (SX_) = (int)(cx + scale * _xp);                            \
        (SY_) = (int)(cy - scale * _yp);                            \
    } while(0)

    /* --------------------------------------------------------------
       3) Righe iso-phi
       -------------------------------------------------------------- */
    for(int ip=0; ip<Np; ip++){
        double phi = TWO_PI * (double)ip / (double)Np;

        int first = 1;
        int px0 = 0, py0 = 0;

        for(int ir=0; ir<=Nr; ir++){
            int sx, sy;
            PROJECT_POINT(rho[ir], phi, ztab[ir], sx, sy);

            if(!first){
                double t = (double)ir / (double)Nr;

                /* enfasi della regione interna */
                double bright = 0.22 + 0.45 * (1.0 - t);
                uint8_t R = (uint8_t)(180.0 * bright);
                uint8_t G = (uint8_t)(125.0 * bright);
                uint8_t B = (uint8_t)( 82.0 * bright);

                draw_line(px0, py0, sx, sy, R, G, B);
            }

            px0 = sx;
            py0 = sy;
            first = 0;
        }
    }

    /* --------------------------------------------------------------
       4) Righe iso-r
       -------------------------------------------------------------- */
    for(int ir=0; ir<=Nr; ir += 6){
        double t = (double)ir / (double)Nr;
        double bright = 0.14 + 0.26 * (1.0 - t);

        uint8_t R = (uint8_t)(150.0 * bright);
        uint8_t G = (uint8_t)(115.0 * bright);
        uint8_t B = (uint8_t)( 95.0 * bright);

        int first = 1;
        int px0 = 0, py0 = 0;

        for(int ip=0; ip<=Np; ip++){
            double phi = TWO_PI * (double)ip / (double)Np;
            int sx, sy;
            PROJECT_POINT(rho[ir], phi, ztab[ir], sx, sy);

            if(!first){
                draw_line(px0, py0, sx, sy, R, G, B);
            }

            px0 = sx;
            py0 = sy;
            first = 0;
        }
    }

    /* --------------------------------------------------------------
       5) Curve caratteristiche proiettate sulla stessa superficie
       -------------------------------------------------------------- */
    typedef struct {
        double r;
        uint8_t R, G, B;
        int fill;
    } RingInfo;

    RingInfo rings[] = {
        { g_kp.r_ergo, 230, 115,  20, 0 },
        { g_kp.r_isco,  41, 128, 185, 0 },
        { g_kp.r_ph,    39, 174,  96, 0 },
        { g_kp.rp,     192,  57,  43, 1 }
    };

    for(int k=0; k<4; k++){
        double rr = rings[k].r;
        if(rr <= r_min || rr >= r_max) continue;

        KerrMetric m;
        kerr_metric(rr, a, &m);
        double rho_r = sqrt(fmax(m.gphiphi, 1e-12));

        /* z(rr) per interpolazione lineare su tabella */
        double z_r = 0.0;
        for(int i=1; i<=Nr; i++){
            if(rr <= rtab[i]){
                double u = (rr - rtab[i-1]) / (rtab[i] - rtab[i-1]);
                z_r = ztab[i-1] * (1.0-u) + ztab[i] * u;
                break;
            }
        }

        int first = 1;
        int px0 = 0, py0 = 0;
        for(int ip=0; ip<=Np; ip++){
            double phi = TWO_PI * (double)ip / (double)Np;
            int sx, sy;
            PROJECT_POINT(rho_r, phi, z_r, sx, sy);

            if(!first){
                draw_line(px0, py0, sx, sy, rings[k].R, rings[k].G, rings[k].B);
            }
            px0 = sx;
            py0 = sy;
            first = 0;
        }
    }

    /* --------------------------------------------------------------
       6) Disco interno nero per l'orizzonte, nella stessa proiezione
       -------------------------------------------------------------- */
    {
        KerrMetric mh;
        kerr_metric(g_kp.rp, a, &mh);
        double rho_h = sqrt(fmax(mh.gphiphi, 1e-12));
        int rad = (int)(rho_h * scale * 0.55);
        if(rad > 2){
            fill_circle((int)cx, (int)(cy + 0.02*HEIGHT), rad, 0, 0, 0);
        }
    }

    /* --------------------------------------------------------------
       7) Frecce di frame dragging, ancora come overlay qualitativo
       -------------------------------------------------------------- */
    if(a > 0.01){
        double rr = g_kp.r_ergo * 1.22;
        KerrMetric m;
        kerr_metric(rr, a, &m);
        double rho_r = sqrt(fmax(m.gphiphi, 1e-12));

        double z_r = 0.0;
        for(int i=1; i<=Nr; i++){
            if(rr <= rtab[i]){
                double u = (rr - rtab[i-1]) / (rtab[i] - rtab[i-1]);
                z_r = ztab[i-1] * (1.0-u) + ztab[i] * u;
                break;
            }
        }

        for(int i=0; i<12; i++){
            double phi = TWO_PI * (double)i / 12.0;
            int ax, ay;
            PROJECT_POINT(rho_r, phi, z_r, ax, ay);

            double tang = phi + PI/2.0;
            double al = 0.35 + a*0.55;
            uint8_t fc = (uint8_t)(230*al);
            uint8_t gc = (uint8_t)(110*al);
            uint8_t bc = (uint8_t)( 15*al);

            int ax2 = ax + (int)(10*cos(tang));
            int ay2 = ay - (int)( 7*sin(tang)*ci);

            draw_line(ax, ay, ax2, ay2, fc, gc, bc);
            put_pixel(ax2 + (int)(4*cos(tang + 2.5)), ay2 - (int)(3*sin(tang + 2.5)), fc, gc, bc);
            put_pixel(ax2 + (int)(4*cos(tang - 2.5)), ay2 - (int)(3*sin(tang - 2.5)), fc, gc, bc);
        }
    }

    #undef PROJECT_POINT
}
/* ------------------------------------------------------------------ */
/*  RENDER 1: GEODETICHE NULLE                                          */
/* ------------------------------------------------------------------ */
static void render_geodesics(void)
{
    for(int i=0;i<WIDTH*HEIGHT*4;i+=4){
        pixels[i]=8;pixels[i+1]=7;pixels[i+2]=5;pixels[i+3]=255;
    }

    double cx=WIDTH/2.0, cy=HEIGHT/2.0;
    double scale=g_zoom*19.0;
    double a=g_a;
    double r_max=g_remit*2.4;
    double r_cap=g_kp.rp*1.005;

    /* Griglia circolare tenue */
    int icx=(int)cx,icy=(int)cy;
    for(int rg=5;rg<=(int)r_max;rg+=5)
        draw_circle(icx,icy,(int)(rg*scale),30,26,20,0);

    GeodesicPath *path=malloc(sizeof(GeodesicPath));
    if(!path) return;

    double E=1.0;
    int n=g_nrays;

    for(int i=0;i<n;i++){
        double phi0=(double)i/n*TWO_PI+g_phi_off;
        double frac=((double)i/(n-1))*2.0-1.0;
        double b=frac*g_remit*0.90;
        double L=E*b;

        double V0=Veff(g_remit,a,E,L);
        if(V0<0.0) continue;

        integrate_geodesic(g_remit,phi0,E,L,a,r_max*1.6,r_cap,path);
        if(path->n<2) continue;

        /* Colore basato su parametro d'impatto e esito */
        uint8_t R,G,B;
        if(path->captured){
            double t=fmin(1.0,(double)path->n/500.0);
            R=(uint8_t)(220);G=(uint8_t)(150*(1-t*0.4));B=20;
        } else {
            double b_norm=(b+g_remit*0.90)/(2.0*g_remit*0.90);
            double lam=380.0+b_norm*370.0;
            wavelength_rgb(lam,&R,&G,&B);
        }

        for(int j=1;j<path->n;j++){
            double wx0=path->r[j-1]*cos(path->phi[j-1]);
            double wy0=path->r[j-1]*sin(path->phi[j-1]);
            double wx1=path->r[j  ]*cos(path->phi[j  ]);
            double wy1=path->r[j  ]*sin(path->phi[j  ]);
            int sx0,sy0,sx1,sy1;
            w2s(wx0,wy0,cx,cy,scale,&sx0,&sy0);
            w2s(wx1,wy1,cx,cy,scale,&sx1,&sy1);
            draw_line(sx0,sy0,sx1,sy1,R,G,B);
        }
    }
    free(path);

    draw_circle(icx,icy,(int)(g_kp.r_ergo*scale),230,115,20,1);
    draw_circle(icx,icy,(int)(g_kp.r_isco*scale), 41,128,185,1);
    draw_circle(icx,icy,(int)(g_kp.r_ph  *scale), 39,174, 96,1);
    draw_circle(icx,icy,(int)(g_kp.rp    *scale),192, 57, 43,0);
    fill_circle(icx,icy,(int)(g_kp.rp    *scale),  0,  0,  0);
}

/* ------------------------------------------------------------------ */
/*  RENDER 2: LENTE GRAVITAZIONALE                                     */
/* ------------------------------------------------------------------ */
static double source_sample(double tx,double ty,
                             uint8_t *R,uint8_t *G,uint8_t *B)
{
    double r=sqrt(tx*tx+ty*ty);
    double phi=atan2(ty,tx);
    double bulge=exp(-r*r*0.85)*1.3;
    double arm=0.0;
    for(int k=0;k<2;k++){
        double p=phi-PI*k-log(r+0.05)*1.9;
        double w=fmod(fabs(p),TWO_PI); if(w>PI) w=TWO_PI-w;
        arm+=exp(-w*w*3.2)*exp(-r*0.65)*0.85;
    }
    double fx=tx*13.0,fy=ty*13.0;
    double stars=0.0;
    for(int i=-1;i<=1;i++) for(int j=-1;j<=1;j++){
        double dx=fx-round(fx)-i*0.6, dy=fy-round(fy)-j*0.6;
        double d2=dx*dx+dy*dy;
        if(d2<0.025) stars+=0.55*exp(-d2*55.0);
    }
    double tot=fmin(1.0,bulge+arm+stars);
    *R=(uint8_t)(fmin(255.0,(tot*0.95+stars*0.2)*220));
    *G=(uint8_t)(fmin(255.0,(tot*0.75+stars*0.2)*220));
    *B=(uint8_t)(fmin(255.0,(tot*0.35+stars*0.4)*220));
    return tot;
}

static void render_lens(void)
{
    for(int i=0;i<WIDTH*HEIGHT*4;i+=4){
        pixels[i]=5;pixels[i+1]=4;pixels[i+2]=3;pixels[i+3]=255;
    }

    double cx=WIDTH/2.0, cy=HEIGHT/2.0;
    double scale=g_zoom*19.0;
    double a=g_a;
    double r_ph=g_kp.r_ph;

    int step=2;
    for(int sy=0;sy<HEIGHT;sy+=step){
        for(int sx=0;sx<WIDTH;sx+=step){
            double wx=(sx-cx)/scale, wy=-(sy-cy)/scale;
            double r=sqrt(wx*wx+wy*wy);
            double phi=atan2(wy,wx);

            uint8_t R=0,G=0,B=0;

            if(r<g_kp.rp*0.98){
                /* Dentro orizzonte */
            } else {
                double b=r;
                int prograde=(wy>=0.0);
                double delta;
                if(b < r_ph*1.05){
                    delta=PI;
                } else {
                    double bi=1.0/b;
                    delta=4.0*bi + (15.0*PI/4.0)*bi*bi;
                    delta+=(prograde?1.0:-1.0)*2.0*a*bi*bi;
                    if(delta>PI) delta=PI;
                }

                if(delta<PI*0.95){
                    double phi_s=phi - delta*(wy>=0.0?0.4:-0.4);
                    double tx=cos(phi_s)*(0.22+b*0.011);
                    double ty=sin(phi_s)*(0.22+b*0.011);
                    double brightness=source_sample(tx,ty,&R,&G,&B);
                    double b_cr=r_ph;
                    double mu=(b>b_cr+0.6)?fmin(5.0,b_cr/fabs(b-b_cr)):1.0;
                    double bright=fmin(1.0,brightness*(0.65+0.35*sqrt(mu)));
                    R=(uint8_t)(R*bright); G=(uint8_t)(G*bright); B=(uint8_t)(B*bright);
                } else {
                    R=8; G=4; B=0;
                }
            }

            for(int dy=0;dy<step&&sy+dy<HEIGHT;dy++)
            for(int dx=0;dx<step&&sx+dx<WIDTH;dx++)
                put_pixel(sx+dx,sy+dy,R,G,B);
        }
    }

    int icx=(int)cx,icy=(int)cy;
    draw_circle(icx,icy,(int)(g_kp.rp*scale),192,57,43,0);
    fill_circle(icx,icy,(int)(g_kp.rp*scale)-1,0,0,0);
}

/* ------------------------------------------------------------------ */
/*  RENDER 3: DILATAZIONE TEMPORALE / REDSHIFT                         */
/* ------------------------------------------------------------------ */
static void render_redshift(void)
{
    for(int i=0;i<WIDTH*HEIGHT*4;i+=4){
        pixels[i]=10;pixels[i+1]=8;pixels[i+2]=5;pixels[i+3]=255;
    }

    double a=g_a;
    double r_min=g_kp.rp*1.001, r_max=28.0;
    int Nr=800;

    /* ---- Mappa cromatica 2D (metà sinistra) ---- */
    int map_w=WIDTH/2-10, map_h=HEIGHT-60;
    int map_ox=10, map_oy=30;
    double map_cx=map_ox+map_w/2.0, map_cy=map_oy+map_h/2.0;
    double map_sc=fmin(map_w,map_h)/2.0/(r_max*0.46);
    double lam_emit=550.0;

    for(int sy=map_oy;sy<map_oy+map_h;sy++){
        for(int sx=map_ox;sx<map_ox+map_w;sx++){
            double wx=(sx-map_cx)/map_sc, wy=-(sy-map_cy)/map_sc;
            double r=sqrt(wx*wx+wy*wy);
            if(r>r_max*0.46) continue;
            if(r<g_kp.rp){pixels[(sy*WIDTH+sx)*4]=0;pixels[(sy*WIDTH+sx)*4+1]=0;pixels[(sy*WIDTH+sx)*4+2]=0;pixels[(sy*WIDTH+sx)*4+3]=255;continue;}

            KerrMetric m; kerr_metric(r,a,&m);
            double neg_gtt=-m.gtt;
            uint8_t R,G,B;
            if(neg_gtt<=0.0){
                /* Ergosfera */
                R=230;G=115;B=20;
            } else {
                double z=1.0/sqrt(neg_gtt)-1.0;
                double lam_obs=lam_emit*(1.0+z);
                lam_obs=fmax(350.0,fmin(780.0,lam_obs));
                wavelength_rgb(lam_obs,&R,&G,&B);
                double bright=fmin(1.0,0.55+0.45*exp(-r*0.14));
                R=(uint8_t)(R*bright); G=(uint8_t)(G*bright); B=(uint8_t)(B*bright);
            }
            put_pixel(sx,sy,R,G,B);
        }
    }

    /* Cerchi sulla mappa */
    int icx=(int)map_cx, icy=(int)map_cy;
    draw_circle(icx,icy,(int)(g_kp.rp    *map_sc),255,255,255,0);
    draw_circle(icx,icy,(int)(g_kp.r_ergo*map_sc),230,115, 20,1);
    draw_circle(icx,icy,(int)(g_kp.r_isco*map_sc), 41,128,185,1);
    fill_circle(icx,icy,(int)(g_kp.rp*map_sc)-1,0,0,0);

    /* ---- Grafici (metà destra) ---- */
    int gx=WIDTH/2+10, gw=WIDTH/2-20;
    int gy=30,  gh=HEIGHT-60;
    int p1h=(gh-30)/3, p2h=p1h, p3h=p1h;
    int p1y=gy, p2y=gy+p1h+15, p3y=gy+2*(p1h+15);

    /* Linee assi */
    draw_line(gx,p1y,gx,p1y+p1h,80,70,50);
    draw_line(gx,p1y+p1h,gx+gw,p1y+p1h,80,70,50);
    draw_line(gx,p2y,gx,p2y+p2h,80,70,50);
    draw_line(gx,p2y+p2h,gx+gw,p2y+p2h,80,70,50);
    draw_line(gx,p3y,gx,p3y+p3h,80,70,50);
    draw_line(gx,p3y+p3h,gx+gw,p3y+p3h,80,70,50);

    int px_prev=0,py_prev=0,first;
    double om_max=0.0;

    /* Pre-calcolo omega_max */
    for(int i=0;i<=Nr;i++){
        double r=r_min+(r_max-r_min)*i/Nr;
        KerrMetric m; kerr_metric(r,a,&m);
        double om=-m.gtphi/m.gphiphi;
        if(om>om_max) om_max=om;
    }
    if(om_max<1e-9) om_max=1.0;

    first=1;
    for(int i=0;i<=Nr;i++){
        double r=r_min+(r_max-r_min)*i/Nr;
        KerrMetric m; kerr_metric(r,a,&m);
        int px=gx+(int)((r-r_min)/(r_max-r_min)*gw);

        /* dτ/dt */
        double neg_gtt=-m.gtt;
        double dtau=(neg_gtt>0.0)?sqrt(neg_gtt):0.0;
        int py=p1y+p1h-(int)(fmin(1.0,dtau)*p1h);
        if(!first) draw_line(px_prev,py_prev,px,py,52,152,219);
        if(i==0) first=0;
        px_prev=px; py_prev=py;
    }
    first=1;
    for(int i=0;i<=Nr;i++){
        double r=r_min+(r_max-r_min)*i/Nr;
        KerrMetric m; kerr_metric(r,a,&m);
        int px=gx+(int)((r-r_min)/(r_max-r_min)*gw);
        double neg_gtt=-m.gtt;
        double dtau=(neg_gtt>0.0)?sqrt(neg_gtt):1e-4;
        double z=1.0/dtau-1.0; z=fmin(z,10.0);
        int py=p2y+p2h-(int)(fmin(1.0,z/10.0)*p2h);
        if(!first) draw_line(px_prev,py_prev,px,py,192,57,43);
        if(i==0) first=0;
        px_prev=px; py_prev=py;
    }
    first=1;
    for(int i=0;i<=Nr;i++){
        double r=r_min+(r_max-r_min)*i/Nr;
        KerrMetric m; kerr_metric(r,a,&m);
        int px=gx+(int)((r-r_min)/(r_max-r_min)*gw);
        double om=-m.gtphi/m.gphiphi;
        int py=p3y+p3h-(int)(fmin(1.0,om/om_max)*p3h);
        if(!first) draw_line(px_prev,py_prev,px,py,46,204,113);
        if(i==0) first=0;
        px_prev=px; py_prev=py;
    }

    /* Linee verticali r+, ergo, ISCO */
    typedef struct{double r;uint8_t R,G,B;}VL;
    VL vl[]={{g_kp.rp,192,57,43},{g_kp.r_ergo,230,115,20},{g_kp.r_isco,41,128,185},{g_kp.r_ph,39,174,96}};
    for(int k=0;k<4;k++){
        if(vl[k].r<r_min||vl[k].r>r_max) continue;
        int xv=gx+(int)((vl[k].r-r_min)/(r_max-r_min)*gw);
        draw_line(xv,p1y,xv,p3y+p3h,vl[k].R,vl[k].G,vl[k].B);
    }
}

/* ------------------------------------------------------------------ */
/*  FUNZIONI ESPORTATE (chiamate da JS)                                 */
/* ------------------------------------------------------------------ */
EMSCRIPTEN_KEEPALIVE void set_spin(double a)
{
    g_a=fmax(0.001,fmin(0.998,a));
    kerr_props(g_a,&g_kp);
}

EMSCRIPTEN_KEEPALIVE void set_zoom(double z)
{ g_zoom=fmax(0.2,fmin(4.0,z)); }

EMSCRIPTEN_KEEPALIVE void set_mode(int m)
{ g_mode=(RenderMode)(m%MODE_COUNT); }

EMSCRIPTEN_KEEPALIVE void set_nrays(int n)
{ g_nrays=fmax(4,fmin(N_RAYS_MAX,n)); }

EMSCRIPTEN_KEEPALIVE void set_remit(double r)
{ g_remit=fmax(5.0,fmin(40.0,r)); }

EMSCRIPTEN_KEEPALIVE void step_anim(void)
{ g_phi_off+=0.015; }

EMSCRIPTEN_KEEPALIVE double get_rp(void)    { return g_kp.rp; }
EMSCRIPTEN_KEEPALIVE double get_rergo(void) { return g_kp.r_ergo; }
EMSCRIPTEN_KEEPALIVE double get_risco(void) { return g_kp.r_isco; }
EMSCRIPTEN_KEEPALIVE double get_rph(void)   { return g_kp.r_ph; }
EMSCRIPTEN_KEEPALIVE double get_omH(void)   { return g_kp.Omega_H; }
EMSCRIPTEN_KEEPALIVE int    get_width(void) { return WIDTH; }
EMSCRIPTEN_KEEPALIVE int    get_height(void){ return HEIGHT; }

/* Puntatore al buffer pixel (letto da JS per putImageData) */
EMSCRIPTEN_KEEPALIVE uint8_t* get_pixel_buffer(void){ return pixels; }

/* Funzione di render chiamata dal main loop */
EMSCRIPTEN_KEEPALIVE void render_frame(void)
{
    switch(g_mode){
        case MODE_GRID:     render_grid();      break;
        case MODE_GEODESIC: render_geodesics(); break;
        case MODE_LENS:     render_lens();      break;
        case MODE_REDSHIFT: render_redshift();  break;
        default: break;
    }
    /* Notify JS che il buffer è pronto */
    EM_ASM( Module.onFrameReady && Module.onFrameReady(); );
}
/* ------------------------------------------------------------------ */
/*  MAIN                                                                */
/* ------------------------------------------------------------------ */
int main(void)
{
    kerr_props(g_a, &g_kp);
    /* Il loop è gestito da JS tramite requestAnimationFrame;
     * qui non usiamo emscripten_set_main_loop perché il controllo
     * del frame rate e della logica animazione è più pulito in JS. */
    return 0;
}
