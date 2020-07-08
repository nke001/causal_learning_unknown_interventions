/**
 * Includes
 */

#define  PY_SSIZE_T_CLEAN  /* So we get Py_ssize_t args. */
#include <Python.h>        /* Because of "reasons", the Python header must be first. */
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <immintrin.h>



/* Defines */
#define PRNG_BURNIN_TIME 256
#define ZERO8vf          (_mm256_setzero_ps())
#define ZERO8vi          (_mm256_setzero_si256())
#define ONE8vf           (_mm256_set1_ps(1.0))
#define ONE8vi           (_mm256_set1_epi32(1))
#define NINF8vf          (_mm256_castsi256_ps(_mm256_set1_epi32(0xFF800000)))



/* Extern "C" Guard */
#ifdef __cplusplus
extern "C" {
#endif



/**
 * Data Structure and Constant Definitions
 */

/**
 * @brief AVX2-accelerated quadruple-parallel xorshift128+ PRNG state.
 */

typedef struct PRNG_QUADXORSHIFT PRNG_QUADXORSHIFT;
struct PRNG_QUADXORSHIFT{
    __m256i a, b;
};



/* Globals Definitions */
static PRNG_QUADXORSHIFT _GLOBAL_PRNG, *GLOBAL_PRNG=&_GLOBAL_PRNG;



/* C Function Forward Declarations */



/* C Function Definitions */

/**
 * @brief Evaluate the logarithm ln(1+x) in base e, for x in the reduced range [-0.5,1].
 * 
 * This routine makes use of the Padé approximant (5/5) of ln(1+x) about x=0:
 * 
 *                   (137/7560)x^5 + (11/36)x^4 + (47/36)x^3 + 2x^2 + x
 *     ln(1+x) = ----------------------------------------------------------
 *               (1/252)x^5 + (5/42)x^4 + (5/6)x^3 + (20/9)x^2 + (5/2)x + 1
 * 
 * This approximant reaches a maximum absolute error at:
 *     <2.5e-8 @ x=1.0
 *     <1.2e-8 @ x=-0.5
 * 
 * Courtesy Wolfram Alpha: PadeApproximant[ln(1+x),{x,0,{5,5}}]
 * 
 * @param [in]  x  The vector of floating-point numbers to compute the logarithm of.
 * @return ln(1+x), not correctly rounded.
 */

static inline __m256   approx_log1p_reduced_8vf(__m256 x){
    /**
     * We use Horner's method to evaluate the two polynomials in parallel.
     * To double the available instruction parallelism and thus keep the FMA
     * units filled, we split the evaluation between the even and odd powers.
     */
    
    const __m256 P5 = _mm256_set1_ps(137./7560.),
                 P4 = _mm256_set1_ps( 11./36.),
                 P3 = _mm256_set1_ps( 47./36.),
                 P2 = _mm256_set1_ps(  2.),
                 P1 = _mm256_set1_ps(  1.),
                 Q5 = _mm256_set1_ps(  1./252.),
                 Q4 = _mm256_set1_ps(  5./42.),
                 Q3 = _mm256_set1_ps(  5./6.),
                 Q2 = _mm256_set1_ps( 20./9.),
                 Q1 = _mm256_set1_ps(  5./2.),
                 Q0 = _mm256_set1_ps(  1.);
    __m256 P, Q, Po, Pe, Qo, Qe, x2;
    
    x2 = _mm256_mul_ps(x, x);
    
    Po = P5;
    Pe = P4;
    Qo = Q5;
    Qe = Q4;
    
    Po = _mm256_fmadd_ps(Po, x2, P3);
    Pe = _mm256_fmadd_ps(Pe, x2, P2);
    Qo = _mm256_fmadd_ps(Qo, x2, Q3);
    Qe = _mm256_fmadd_ps(Qe, x2, Q2);
    
    Po = _mm256_fmadd_ps(Po, x2, P1);
    Pe = _mm256_mul_ps  (Pe, x2);
    Qo = _mm256_fmadd_ps(Qo, x2, Q1);
    Qe = _mm256_fmadd_ps(Qe, x2, Q0);
    
    P  = _mm256_fmadd_ps(Po, x,  Pe);
    Q  = _mm256_fmadd_ps(Qo, x,  Qe);
    
    return _mm256_div_ps(P, Q);
}

/**
 * @brief Evaluate the exponential pow2(x) in base 2, for x in the reduced range [-1,1].
 * 
 * This routine makes use of the Padé approximant (4/4) of pow2(x) about x=0:
 * 
 *               (k4/1680)x^4 + (k3/84)x^3 + (3k2/28)x^2 + (k1/2)x + 1
 *     pow2(x) = -----------------------------------------------------
 *               (k4/1680)x^4 - (k3/84)x^3 + (3k2/28)x^2 - (k1/2)x + 1
 * 
 *     , where k0=log^0(2),
 *             k1=log^1(2),
 *             k2=log^2(2),
 *             k3=log^3(2),
 *             k4=log^4(2)
 * 
 * This approximant reaches a maximum absolute error at:
 *     <1e-9 @ x=-1.0
 *     <3e-9 @ x=+1.0
 * 
 * Courtesy Wolfram Alpha: PadeApproximant[2**x,{x,0,{4,4}}]
 * 
 * @param [in]  x  The vector of floating-point numbers to compute the power of.
 * @return pow2(x), not correctly rounded.
 */

static inline __m256   approx_pow2_reduced_8vf(__m256 x){
    /**
     * We use Horner's method to evaluate the two polynomials in parallel.
     * Because of the extremely special structure of exp(), only two polynomials
     * need to be evaluated: The even and odd powers.
     */
    
    const __m256 P4 = _mm256_set1_ps(0.0001374018443946925),
                 P3 = _mm256_set1_ps(0.003964579190344398),
                 P2 = _mm256_set1_ps(0.051477108634093),
                 P1 = _mm256_set1_ps(0.34657359027997264),
                 P0 = _mm256_set1_ps(1.);
    __m256 P, Q, Po, Pe, x2;
    
    x2 = _mm256_mul_ps(x, x);
    
    Pe = P4;
    Po = P3;
    
    Pe = _mm256_fmadd_ps (Pe, x2, P2);
    Po = _mm256_fmadd_ps (Po, x2, P1);
    
    Pe = _mm256_fmadd_ps (Pe, x2, P0);
    
    P  = _mm256_fmadd_ps (Po, x,  Pe);
    Q  = _mm256_fnmadd_ps(Po, x,  Pe);
    
    return _mm256_div_ps(P, Q);
}

/**
 * @brief Compute approximate logarithm of x.
 */

static inline __m256   log8vf(__m256 x){
    const __m256 ONE       = _mm256_set1_ps(1.0),
                 INVLOG2E  = _mm256_castsi256_ps(_mm256_set1_epi32(0x3f317218));
    
    __m256  ilog2, floge;
    
    /* Compute integer portion of log2() */
    ilog2 = _mm256_cvtepi32_ps(
                _mm256_sub_epi32(
                    _mm256_srli_epi32(_mm256_castps_si256(x), 23),
                    _mm256_set1_epi32(127)));
    
    /* Compute fractional portion of log() using log1p() */
    floge = _mm256_or_ps(x, ONE);
    floge = _mm256_and_ps(floge, _mm256_castsi256_ps(_mm256_set1_epi32(0x3FFFFFFF)));
    floge = _mm256_sub_ps(floge, ONE);
    floge = approx_log1p_reduced_8vf(floge);
    
    /* Fuse and return */
    return _mm256_fmadd_ps(ilog2, INVLOG2E, floge);
}

/**
 * @brief Compute approximate negative logarithm of x.
 */

static inline __m256   nlog8vf(__m256 x){
    const __m256 ONE       = _mm256_set1_ps(1.0),
                 INVLOG2E  = _mm256_castsi256_ps(_mm256_set1_epi32(0x3f317218));
    
    __m256  ilog2, floge;
    
    /* Compute integer portion of log2() */
    ilog2 = _mm256_cvtepi32_ps(
                _mm256_sub_epi32(
                    _mm256_srli_epi32(_mm256_castps_si256(x), 23),
                    _mm256_set1_epi32(127)));
    
    /* Compute fractional portion of log() using log1p() */
    floge = _mm256_or_ps(x, ONE);
    floge = _mm256_and_ps(floge, _mm256_castsi256_ps(_mm256_set1_epi32(0x3FFFFFFF)));
    floge = _mm256_sub_ps(floge, ONE);
    floge = approx_log1p_reduced_8vf(floge);
    
    /* Fuse and return, negated */
    return _mm256_fnmsub_ps(ilog2, INVLOG2E, floge);
}

/**
 * @brief Compute approximate exponential of x.
 * @note  We use the approximation exp(x) ~ 2**(log2(e)*x)
 */

static inline __m256   exp8vf(__m256 x){
    __m256  v,r,f,p;
    __m256i i;
    
    const __m256 LOG2E = _mm256_set1_ps(1.4426950408889634);
    
    v = _mm256_mul_ps(x, LOG2E);                     /* Scale by log2(e) */
    r = _mm256_floor_ps(v);                          /* Round to integer */
    f = _mm256_fmsub_ps(x, LOG2E, r);                /* High-precision fractional part. */
    
    f = approx_pow2_reduced_8vf(f);                  /* Evaluate polynomial */
    
    i = _mm256_cvtps_epi32(r);                       /* Convert to int */
    i = _mm256_add_epi32(i, _mm256_set1_epi32(+127));/* Add exponent bias */
    i = _mm256_max_epi32(i, _mm256_setzero_si256()); /* Clamp below */
    i = _mm256_min_epi32(i, _mm256_set1_epi32(+255));/* Clamp above */
    i = _mm256_slli_epi32(i, 23);                    /* Materialize power of 2 */
    p = _mm256_castsi256_ps(i);                      /* Reinterpret as float */
    
    return _mm256_mul_ps(f,p);                       /* Rejoin integer/fractional parts */
}

/**
 * @brief Draw a scalar uint64_t uniformly-distributed on [0, 2**64).
 * @param S
 * @return Uniformly-random uint64_t.
 */

static inline uint64_t prng_draw_u64(PRNG_QUADXORSHIFT* S){
    __m256i a, b, r;
    a = _mm256_loadu_si256(&S->a);
    b = _mm256_loadu_si256(&S->b);
    a = _mm256_xor_si256(a, _mm256_slli_epi64(a, 23));
    a = _mm256_xor_si256(a, _mm256_srli_epi64(a, 17));
    a = _mm256_xor_si256(a, _mm256_xor_si256(b, _mm256_srli_epi64(b, 26)));
    _mm256_storeu_si256(&S->a, b);
    _mm256_storeu_si256(&S->b, a);
    r = _mm256_add_epi64(a,b);
    return _mm256_extract_epi64(r, 0);
}

/**
 * @brief Draw a vector of standard exp-distributed float32 from the
 *        PRNG state.
 * @param [in] S
 * @return A vector of standard exp-distributed float32. Their range is bounded
 *         on [2**-92, 63.76954061151497], and is finite and strictly positive.
 */

static inline __m256   prng_draw_exp8vf(PRNG_QUADXORSHIFT* S){
    __m256i a, b, c, d, e, f, g, h;
    __m256i m0, m1, m2, m3;
    __m256i m1e, m2e;
    __m256i fi1, fi2, fi3;
    __m256i hi, geo;
    __m256  hf, exg;
    __m256  x;
    __m256  n0, n1, n;
    
    /**
     * binary32: 1s+8e+23m
     * binary32: Exponent bias 127.
     *             - Biased exponent 255=infty+NaN
     *             - Biased exponent 0  =denormal
     * binary32: 2**-30 -> biased exponent -30+127 = 97
     * binary32: 2**-30 -> [0 01100001 00000000000000000000000]
     *                  -> 0x   3   0    8   0   0   0   0   0]
     */
    
    const __m256i TWOPOW30 = _mm256_set1_epi32(0x40000000);
    const __m256 ONE       = _mm256_set1_ps(1.0),
                 INVLOG2E  = _mm256_castsi256_ps(_mm256_set1_epi32(0x3f317218)),
                 TWOPOWM30 = _mm256_castsi256_ps(_mm256_set1_epi32(0x30800000));
    
    a  = _mm256_loadu_si256(&S->a);
    b  = _mm256_loadu_si256(&S->b);
    a  = _mm256_xor_si256(a, _mm256_slli_epi64(a, 23));
    a  = _mm256_xor_si256(a, _mm256_srli_epi64(a, 17));
    a  = _mm256_xor_si256(a, _mm256_xor_si256(b, _mm256_srli_epi64(b, 26)));
    c  = b;
    d  = a;
    c  = _mm256_xor_si256(c, _mm256_slli_epi64(c, 23));
    c  = _mm256_xor_si256(c, _mm256_srli_epi64(c, 17));
    c  = _mm256_xor_si256(c, _mm256_xor_si256(d, _mm256_srli_epi64(d, 26)));
    e  = d;
    f  = c;
    e  = _mm256_xor_si256(e, _mm256_slli_epi64(e, 23));
    e  = _mm256_xor_si256(e, _mm256_srli_epi64(e, 17));
    e  = _mm256_xor_si256(e, _mm256_xor_si256(f, _mm256_srli_epi64(f, 26)));
    g  = f;
    h  = e;
    g  = _mm256_xor_si256(g, _mm256_slli_epi64(g, 23));
    g  = _mm256_xor_si256(g, _mm256_srli_epi64(g, 17));
    g  = _mm256_xor_si256(g, _mm256_xor_si256(h, _mm256_srli_epi64(h, 26)));
    _mm256_storeu_si256(&S->a, h);
    _mm256_storeu_si256(&S->b, g);
    m0 = _mm256_add_epi64(a,b);
    m1 = _mm256_add_epi64(c,d);
    m2 = _mm256_add_epi64(e,f);
    m3 = _mm256_add_epi64(g,h);
    
    /**
     * Select whether we will generate from the HI or LO half of the interval:
     * HI: [0.5, 1.0)
     * LO: (0.0, 0.5]
     */
    
    hi  = _mm256_slli_epi32(m0, 31);/* Low bit of m0 is discarded in processing below. */
    
    /**
     * Select a mantissa by generating a U(1,2)-distributed float on the
     * range [-2,-1] or [1, 2] with 50-50 probability.
     * 
     * This is done by using one word of entropy to:
     * 
     *     1. Construct a random mantissa between 0x40000000 and 0x7FFFFFFF
     *        inclusive [2*30 to 2**31-1]
     *     2. Breaking all ties to even by ORing a 1 into the LSB.
     *     3. Converting to floating-point. This results in floating-point
     *        numbers rounded without bias on range [2**30, 2**31].
     *     4. Dividing by 0x40000000 (equivalently, multiplying by 2**-30).
     *        This results in floating-point numbers correctly rounded and
     *        with range [1.0, 2.0].
     *     5. Negating if we selected the high range by toggling the sign bit.
     */
    
    m0 = _mm256_srli_epi32(m0, 1);
    m0 = _mm256_or_si256(m0, _mm256_set1_epi32(0x40000001));
    n0 = _mm256_cvtepi32_ps(m0);
    n0 = _mm256_mul_ps(n0, TWOPOWM30);
    n0 = _mm256_xor_ps(n0, _mm256_castsi256_ps(hi));
    hi = _mm256_srai_epi32(hi, 31);
    hf = _mm256_castsi256_ps(hi);
    
    /**
     * Select a Geom(p=0.5)-distributed exponent on range [-2,-92]. This is done
     * by using three words of entropy to compute
     * 
     *     m1 |= 0x40000000; m1 &= -m1;
     *     m2 |= 0x40000000; m2 &= -m2;
     *     m3 |= 0x40000000; m3 &= -m3;
     * 
     * The results of these operations are *always* powers of 2 between 1
     * (most often) and 2**30 (most rarely). Abusing conversion to IEE Std 754
     * binary32 (single-precision) floating-point, it is possible to compute
     * their log2(x) using only bithacks exploiting their very specific IEEE
     * Std 754 structure.
     * 
     * These three log2()'s can then be combined together, effectively
     * implementing ctz(m1..m2..m3) on a 90-bit random bit-string.
     */
    
    m1  = _mm256_or_si256(m1, TWOPOW30);
    m1  = _mm256_and_si256(m1, _mm256_sub_epi32(_mm256_setzero_si256(), m1));
    m1e = _mm256_cmpeq_epi32(m1, TWOPOW30);
    fi1 = _mm256_castps_si256(_mm256_cvtepi32_ps(m1));
    m2  = _mm256_or_si256(m2, TWOPOW30);
    m2  = _mm256_and_si256(m2, _mm256_sub_epi32(_mm256_setzero_si256(), m2));
    m2e = _mm256_cmpeq_epi32(m2, TWOPOW30);
    fi2 = _mm256_castps_si256(_mm256_cvtepi32_ps(m2));
    m3  = _mm256_or_si256(m3, TWOPOW30);
    m3  = _mm256_and_si256(m3, _mm256_sub_epi32(_mm256_setzero_si256(), m3));
    fi3 = _mm256_castps_si256(_mm256_cvtepi32_ps(m3));
    
    fi2 = _mm256_blendv_epi8(_mm256_castps_si256(ONE), fi2, m1e);
    fi3 = _mm256_blendv_epi8(_mm256_castps_si256(ONE), fi3, _mm256_and_si256(m1e,m2e));
    geo = _mm256_add_epi32(fi1, fi2);
    geo = _mm256_add_epi32(geo, fi3);
    
    /**
     * At this point, geo has the following structure:
     * 
     *  SIGN |             EXPONENT          | MANTISSA
     * [  0  | ctz(90'randombitstring)+3*127 |   23'0  ]
     * 
     * with guaranteed overflow of the 8-bit exponent field into the sign.
     */
    
    exg = _mm256_castsi256_ps(
          _mm256_sub_epi32(_mm256_set1_epi32((4*127U-2U)<<23), geo));/* Subtract from four  exp-biases minus 2. */
    geo = _mm256_sub_epi32(_mm256_set1_epi32((3*127U-2U)<<23), geo); /* Subtract from three exp-biases minus 2. */
    geo = _mm256_srai_epi32(geo, 23);
    
    /**
     * We now have:
     *     hi  ~ Ber(p=0.5)
     *     n0  ~ Uniform(1,2) * (-1)**hi
     *     geo ~ -2-Geom(p=0.5)   Range: -2..-92
     *     exg ~ 2^geo
     * 
     * The input x to the first log1p(x) function is:
     *     x = n0-1         if !hi  (0.0-0.5; n0 is positive; x is non-negative [0,1])
     *     x = n0*exg       if  hi  (0.5-1.0; n0 is negative; x is negative [-0.5,0))
     * 
     * The bias  b to the output of the first log1p(x) function is:
     *     b = geo/log2(e)  if !hi  (0.0-0.5)
     *     b = 0            if  hi  (0.5-1.0)
     * 
     * We combine n0 (mantissa), exg (octave) and geo (base-2 log) into a single x
     * on the range [-0.5,+0.5] under control of hi/hf, and execute log1p(x).
     * 
     * We then conditionally add a high-precision geo/log2(e) if we are in the low
     * part.
     * 
     * Since we require the negative logarithm, the final FMADD is a FNMSUB.
     */
    
    x  = _mm256_fmsub_ps(n0, _mm256_blendv_ps(ONE, exg, hf),
                             _mm256_andnot_ps(hf, ONE));
    n1 = approx_log1p_reduced_8vf(x);
    n1 = _mm256_fnmsub_ps(_mm256_andnot_ps(hf, INVLOG2E),
                          _mm256_cvtepi32_ps(geo), n1);
    
    /**
     * We have a guarantee that the exp-distributed n1 will be:
     * 
     *   - n1 >= 2**-92                            {HI range, n0 = -1, geo = -92, x = 1-2**-92}
     *   - n1 <= -log(2**-92) = 63.76954061151497  {LO range, n0 = +1, geo = -92, x =   2**-92}
     * 
     * This is a finite, positive range that our consumers can rely on.
     */
    
    return n1;
}

/**
 * @brief Draw a vector of standard Gumbel-distributed float32 from the
 *        PRNG state.
 * @param S
 * @return A vector of standard Gumbel-distributed float32. Their range is
 *         bounded on [-4.155275656467376, 63.76954061151497]
 */

static inline __m256   prng_draw_gumbel8vf(PRNG_QUADXORSHIFT* S){
    /**
     * We must safely invoke -log() on an exp-distributed variate. We have a
     * guarantee from the exp-distributed sampler that the variate is finite
     * and positive. Therefore, we do not need to handle any special cases,
     * and feed directly into nlog().
     */
    
    return nlog8vf(prng_draw_exp8vf(S));
}

/**
 * @brief Seed the PRNG.
 * 
 * @param [in]  S  PRNG state to seed.
 * @param [in]  s  64-bit unsigned seed.
 */

static inline void     prng_seed(PRNG_QUADXORSHIFT* S, uint64_t seed){
    const uint64_t PHI_MINUS_1 = 0x9E3779B97F4A7C15;
    _mm256_storeu_si256(&S->a, _mm256_set_epi64x(seed+0*PHI_MINUS_1,
                                                 seed+2*PHI_MINUS_1,
                                                 seed+4*PHI_MINUS_1,
                                                 seed+6*PHI_MINUS_1));
    _mm256_storeu_si256(&S->b, _mm256_set_epi64x(seed+7*PHI_MINUS_1,
                                                 seed+5*PHI_MINUS_1,
                                                 seed+3*PHI_MINUS_1,
                                                 seed+1*PHI_MINUS_1));
}

/**
 * @brief Seed the PRNG and burn it in.
 * 
 * @param [in]  S  PRNG state to seed and burn in.
 * @param [in]  s  64-bit unsigned seed.
 */

static inline void     prng_seed_and_burnin(PRNG_QUADXORSHIFT* S, uint64_t seed){
    int i;
    
    prng_seed(S, seed);
    for(i=0;i<PRNG_BURNIN_TIME;i++)
        prng_draw_u64(GLOBAL_PRNG);
}

/**
 * @brief Perform 8x8 fp32 or int32 transposition.
 * 
 * Beginning from
 * 
 *        0  1  2  3  4  5  6  7
 *        8  9 10 11 12 13 14 15
 *       16 17 18 19 20 21 22 23
 *       24 25 26 27 28 29 30 31
 *       32 33 34 35 36 37 38 39
 *       40 41 42 43 44 45 46 47
 *       48 49 50 51 52 53 54 55
 *       56 57 58 59 60 61 62 63
 * 
 * In a first step (8 insns),
 * 
 *        0  8  1  9  4 12  5 13
 *        2 10  3 11  6 14  7 15
 *       16 24 17 25 20 28 21 29
 *       18 26 19 27 22 30 23 31
 *       32 40 33 41 36 44 37 45
 *       34 42 35 43 38 46 39 47
 *       48 56 49 57 52 60 53 61
 *       50 58 51 59 54 62 55 63
 * 
 * In a second step (8 insns),
 * 
 *        0  8 16 24  4 12 20 28
 *        1  9 17 25  5 13 21 29
 *        2 10 18 26  6 14 22 30
 *        3 11 19 27  7 15 23 31
 *       32 40 48 56 36 44 52 60
 *       33 41 49 57 37 45 53 61
 *       34 42 50 58 38 46 54 62
 *       35 43 51 59 39 47 55 63
 * 
 * And in the third step (8 insns),
 * 
 *        0  8 16 24 32 40 48 56
 *        1  9 17 25 33 41 49 57
 *        2 10 18 26 34 42 50 58
 *        3 11 19 27 35 43 51 59
 *        4 12 20 28 36 44 52 60
 *        5 13 21 29 37 45 53 61
 *        6 14 22 30 38 46 54 62
 *        7 15 23 31 39 47 55 63
 * 
 * PERFORMANCE:
 *   - On Haswell, 24 instructions all on Port 5, so 24cc.
 */

static inline void     transpose_8x8vf(__m256  i0, __m256  i1, __m256  i2, __m256  i3, __m256  i4, __m256  i5, __m256  i6, __m256  i7,
                                       __m256* o0, __m256* o1, __m256* o2, __m256* o3, __m256* o4, __m256* o5, __m256* o6, __m256* o7){
    __m256 j0 = _mm256_unpacklo_ps    (i0, i1);
    __m256 j1 = _mm256_unpackhi_ps    (i0, i1);
    __m256 j2 = _mm256_unpacklo_ps    (i2, i3);
    __m256 j3 = _mm256_unpackhi_ps    (i2, i3);
    __m256 j4 = _mm256_unpacklo_ps    (i4, i5);
    __m256 j5 = _mm256_unpackhi_ps    (i4, i5);
    __m256 j6 = _mm256_unpacklo_ps    (i6, i7);
    __m256 j7 = _mm256_unpackhi_ps    (i6, i7);
    __m256 k0 = _mm256_shuffle_ps     (j0, j2, 0b01000100);
    __m256 k1 = _mm256_shuffle_ps     (j0, j2, 0b11101110);
    __m256 k2 = _mm256_shuffle_ps     (j1, j3, 0b01000100);
    __m256 k3 = _mm256_shuffle_ps     (j1, j3, 0b11101110);
    __m256 k4 = _mm256_shuffle_ps     (j4, j6, 0b01000100);
    __m256 k5 = _mm256_shuffle_ps     (j4, j6, 0b11101110);
    __m256 k6 = _mm256_shuffle_ps     (j5, j7, 0b01000100);
    __m256 k7 = _mm256_shuffle_ps     (j5, j7, 0b11101110);
    *o0       = _mm256_permute2f128_ps(k0, k4, 0b00100000);
    *o1       = _mm256_permute2f128_ps(k1, k5, 0b00100000);
    *o2       = _mm256_permute2f128_ps(k2, k6, 0b00100000);
    *o3       = _mm256_permute2f128_ps(k3, k7, 0b00100000);
    *o4       = _mm256_permute2f128_ps(k0, k4, 0b00110001);
    *o5       = _mm256_permute2f128_ps(k1, k5, 0b00110001);
    *o6       = _mm256_permute2f128_ps(k2, k6, 0b00110001);
    *o7       = _mm256_permute2f128_ps(k3, k7, 0b00110001);
}

/**
 * @brief Batched Horizontal Sum of 8x8 float matrix to 8-element vector.
 * 
 * @return Packed horizontal sums of 8 vectors.
 * 
 * PERFORMANCE:
 *   - On Haswell, 9 instructions.
 *     THROUGHPUT:
 *       - 6x vhaddps     (p1 2p5)
 *       - 2x vperm2f128  (p5)
 *       - 1x vaddps      (p1)
 *       - TOTAL: (7p1 14p5)
 *     LATENCY:
 *       T= 0: First  four vhaddps proceed without stalls
 *       T= 8: Fifth  vhaddps proceeds without stall
 *       T=10: Sixth  vhaddps delayed 1cc
 *       T=11: Sixth  vhaddps begins execution
 *       T=16: First  vperm2f128 begins execution
 *       T=17: Second vperm2f128 begins execution
 *       T=20: First  vaddps begins execution
 *       T=23: Result available.
 *     PARALLELISM:
 *       - About two instances can be executed concurrently.
 */

static inline __m256   hsum_8x8vf(__m256  a, __m256  b, __m256  c, __m256  d, __m256  e, __m256  f, __m256  g, __m256  h){
    __m256 i = _mm256_hadd_ps(a, b);/* {a01, a23, b01, b23, a45, a67, b45, b67} */
    __m256 j = _mm256_hadd_ps(c, d);/* {c01, c23, d01, d23, c45, c67, d45, d67} */
    __m256 k = _mm256_hadd_ps(e, f);/* {e01, e23, f01, f23, e45, e67, f45, f67} */
    __m256 l = _mm256_hadd_ps(g, h);/* {g01, g23, h01, h23, g45, g67, h45, h67} */
    __m256 m = _mm256_hadd_ps(i, j);/* {a0123, b0123, c0123, d0123, a4567, b4567, c4567, d4567} */
    __m256 n = _mm256_hadd_ps(k, l);/* {e0123, f0123, g0123, h0123, e4567, f4567, g4567, h4567} */
    __m256 o = _mm256_permute2f128_ps(m, n, 0x20);/* {a0123, b0123, c0123, d0123, e0123, f0123, g0123, h0123} */
    __m256 p = _mm256_permute2f128_ps(m, n, 0x31);/* {a4567, b4567, c4567, d4567, e4567, f4567, g4567, h4567} */
    return _mm256_add_ps(o, p);/* {a01234567, b01234567, c01234567, d01234567, e01234567, f01234567, g01234567, h01234567} */
}

/**
 * @brief Single-precision LeakyReLU.
 * 
 * @param [in]      N       Number of floats to process. 
 * @param [in]      alpha   The slope of the negative part of the LeakyReLU.
 * @param [in]      x       The pointer to the source vector.
 * @param [in/out]  y       The pointer to the destination vector.
 * @note Assumes N is a multiple of 8 and no or perfect overlap!!!!
 */

static inline void     leakyrelu(uint64_t N, float alpha, const float* x, float* y){
    uint64_t i;
    
    __m256 va = _mm256_set1_ps(alpha);
    __m256 Z  = ZERO8vf;
    for(i=0;i<N;i+=8,x+=8,y+=8){
        __m256 vx = _mm256_loadu_ps(x);
        __m256 vp = _mm256_max_ps(vx,Z);
        __m256 vn = _mm256_min_ps(vx,Z);
        _mm256_storeu_ps(y, _mm256_fmadd_ps(va,vn,vp));
    }
}

/**
 * @brief Single-precision LeakyReLU derivative.
 * 
 * @param [in]      N       Number of floats to process. 
 * @param [in]      alpha   The slope of the negative part of the LeakyReLU.
 * @param [in/out]  x       The pointer to the input. Modified inplace with derivative.
 * @param [in]      dy      The pointer to the derivative of the loss with respect to the derivative.
 * @note Assumes N is a multiple of 8 and no or perfect overlap!!!!
 */

static inline void     dleakyreludx(uint64_t N, float alpha, float* x, const float* dy){
    uint64_t i;
    
    __m256 va = _mm256_set1_ps(alpha);
    for(i=0;i<N;i+=8,x+=8,dy+=8){
        __m256 vx  = _mm256_load_ps(x);
        __m256 vdy = _mm256_load_ps(dy);
        __m256 vs  = _mm256_blendv_ps(ONE8vf, va, vx);
        __m256 vdx = _mm256_mul_ps(vdy, vs);
        _mm256_store_ps(x, vdx);
    }
}

/**
 * @brief Perform the actual sampling from a causal graph.
 * 
 * @param [in]     bbuffer  BifGraph  None
 * @param [out]    bout     uint32    (M,bs,)
 * @param [in,out] prng     PRNG      None
 * @return 0 if successfully completed, !0 otherwise.
 */

static int             do_sample_cpt(Py_buffer* bbuffer, Py_buffer* bout,
                                     PRNG_QUADXORSHIFT* prng){
    /**
     * Temporaries
     */
    
    Py_ssize_t a,s,i,j,k,l;
    
    __m256i outmask8vi, argmax8vi, cnt8vi, sel8vi;
    __m256  prob8vf, max8vf, cmp8vf;
    const float* base0, *base1, *base2, *base3, *base4, *base5, *base6, *base7;
    
    int ret = 1;
    
    const Py_ssize_t M   = bout->shape[0];
    const Py_ssize_t BS  = bout->shape[1];
    const struct{
        uint64_t num_choices,
                 num_ancestors,
                 offset_to_ancestors_list,
                 offset_to_strides_list,
                 offset_to_cpt;
    }               *var = (void*)((uint64_t*)bbuffer->buf+4);
    
    
    /**
     * Definition of Indexer Macros.
     */
    
    #define BASE_ANCESTORS_INDEXER(k) \
        ((const uint64_t*)((char*)bbuffer->buf + var[(k)].offset_to_ancestors_list))
    #define BASE_STRIDES_INDEXER(k) \
        ((const uint64_t*)((char*)bbuffer->buf + var[(k)].offset_to_strides_list))
    #define BASE_OFFSET_INDEXER(k) \
        ((const float*)   ((char*)bbuffer->buf + var[(k)].offset_to_cpt))
    #define BASE_BUMP(p,s,v) \
        do{(p) = (const float*)((char*)(p) + (s)*(v));}while(0)
    #define OUT_INDEXER(k,l)  \
        ((uint32_t*)((char*)bout->buf + (k)*bout->strides[0] + (l)*sizeof(uint32_t)))
    
    
    /**
     * Setup.
     * 
     * Frontend computations, such as allocating temporaries and other junk.
     */
    
    if(M<=0 || BS<=0)
        return 0;
    
    for(i=0;i<M;i++)
        if(var[i].num_choices<=0)
            return !0;
    
    /**
     * Loop over batches of 8.
     * 
     * We break up the entire sampling procedure into minibatches of 8 that
     * are 8-way vectorized using AVX, which uses 256-bit (8x32) registers.
     */
    
    for(l=0;l<BS;l+=8){
        outmask8vi = _mm256_set_epi32(l+7<BS ? ~0 : 0, l+6<BS ? ~0 : 0,
                                      l+5<BS ? ~0 : 0, l+4<BS ? ~0 : 0,
                                      l+3<BS ? ~0 : 0, l+2<BS ? ~0 : 0,
                                      l+1<BS ? ~0 : 0, l+0<BS ? ~0 : 0);
        
        /**
         * Loop over the variables k.
         * 
         * Within every minibatch of 8, we sample ancestrally the categorical
         * variables, which are assumed to be sequentially sorted in
         * topological order.
         * 
         * This requires, for each variable,
         *   1. Retrieving the base pointer for variable k.
         *   2. Computing the 8 offsets for the 8 samples.
         *   3. Reading the probabilities from the 8 pointers+offsets.
         *   4. Online categorical sampling using Gumbel softmax trick.
         *   5. Store integer category into out int32 buffer.
         */
        
        for(k=0;k<M;k++){
            /* Base pointer. */
            base0=base1=base2=base3=base4=base5=base6=base7=BASE_OFFSET_INDEXER(k);
            
            
            /* Offset the base pointers according to the ancestors. */
            for(j=0;j<var[k].num_ancestors;j++){
                a = BASE_ANCESTORS_INDEXER(k)[j];
                s = BASE_STRIDES_INDEXER(k)[j];
                sel8vi = _mm256_maskload_epi32((int*)OUT_INDEXER(a,l), outmask8vi);
                BASE_BUMP(base0, s, _mm256_extract_epi32(sel8vi, 0));
                BASE_BUMP(base1, s, _mm256_extract_epi32(sel8vi, 1));
                BASE_BUMP(base2, s, _mm256_extract_epi32(sel8vi, 2));
                BASE_BUMP(base3, s, _mm256_extract_epi32(sel8vi, 3));
                BASE_BUMP(base4, s, _mm256_extract_epi32(sel8vi, 4));
                BASE_BUMP(base5, s, _mm256_extract_epi32(sel8vi, 5));
                BASE_BUMP(base6, s, _mm256_extract_epi32(sel8vi, 6));
                BASE_BUMP(base7, s, _mm256_extract_epi32(sel8vi, 7));
            }
            
            
            /* Loop over category j of variable k for a batch of 8 at a time. */
            max8vf = NINF8vf;
            cnt8vi = argmax8vi = ZERO8vi;
            for(j=0;j<var[k].num_choices;j++){
                /**
                 * Retrieve probabilities.
                 * 
                 * We perform the second layer of the MLP to compute the logits
                 * of variable k's category j, within this batch of 8.
                 */
                
                prob8vf = _mm256_set_ps(base7[j], base6[j], base5[j], base4[j],
                                        base3[j], base2[j], base1[j], base0[j]);
                
                /**
                 * Draw Exponential variates and update the categorical sample
                 * draw.
                 * 
                 * The Gumbel softmax trick consists in adding Gumbel variates to
                 * the logits and selecting their argmax. We can detect that a
                 * new maximum has been reached if logit+Gumbel == max, and if so
                 * we save the current counter value to argmax. The counter is
                 * then incremented regardless.
                 * 
                 * Because we are dealing with raw probabilities here, the
                 * "exponentiated" equivalent is to divide the raw probability
                 * by Exp-distributed variates and then keep the maximum. This
                 * is even faster.
                 * 
                 *     Let v ~ Exp, g=-log(v) ~ Gumbel. Then:
                 *         argmax(logit+g) == argmax(log(prob)-log(v))
                 *                         == argmax(log(prob/v))
                 *                         == argmax(prob/v)
                 * 
                 * Because prob in [0,1], v is finite and strictly positive and
                 * log() is monotonic, this is safe and numerically stable.
                 */
                
                prob8vf   = _mm256_div_ps(prob8vf, prng_draw_exp8vf(prng));
                max8vf    = _mm256_max_ps(prob8vf, max8vf);
                cmp8vf    = _mm256_cmp_ps(prob8vf, max8vf, _CMP_EQ_OQ);
                argmax8vi = _mm256_castps_si256(
                            _mm256_blendv_ps(_mm256_castsi256_ps(argmax8vi),
                                             _mm256_castsi256_ps(cnt8vi),
                                             cmp8vf));
                cnt8vi    = _mm256_add_epi32(cnt8vi, ONE8vi);
            }
            
            
            /**
             * Store out variable k for this batch of 8.
             * 
             * This is done by writing out 8 uint32 integers under control of a
             * store-mask, preventing us from overwriting data outside our range.
             */
            
            _mm256_maskstore_epi32((int*)OUT_INDEXER(k,l), outmask8vi, argmax8vi);
        }
    }
    ret = 0;
    
    
    /**
     * Undefinition of Indexer Macros.
     */
    
    #undef BASE_ANCESTORS_INDEXER
    #undef BASE_STRIDES_INDEXER
    #undef BASE_OFFSET_INDEXER
    #undef BASE_BUMP
    #undef OUT_INDEXER
    
    
    /**
     * Cleanup.
     */
    
    return ret;
}

/**
 * @brief Release exported buffers.
 * @param buffer
 * @param out
 * @return 0.
 */

static int             sample_cpt_release_buffers(Py_buffer* buffer,
                                                  Py_buffer* out){
    PyBuffer_Release(out);
    PyBuffer_Release(buffer);
    return 0;
}

/**
 * @brief Validate arguments to sample*() functions.
 * @param buffer
 * @param out
 * @return 0 if successfully validated all arguments; !0 otherwise.
 */

static int             sample_cpt_validate_args(Py_buffer* buffer,
                                                Py_buffer* out){
    uint64_t*  graph;
    Py_ssize_t buffer_0,
               out_0;
    
    if(out->readonly)
        return !PyErr_Format(PyExc_ValueError, "out array is read-only!");
    
    if(buffer->ndim != 1 ||
       out->ndim    != 2)
        return !PyErr_Format(PyExc_ValueError, "Arrays of incorrect ndim!");
    
    /**
     * We don't care about the item size or stride of "buffer", it's a
     * flattened homebrew data structure anyways.
     */
    
    if(out->itemsize != 4)
        return !PyErr_Format(PyExc_ValueError, "Arrays of incorrect dtype!");
    if(out->strides[1] != 4)
        return !PyErr_Format(PyExc_ValueError, "Arrays of incorrect and/or misaligned strides!");
    
    graph    = (uint64_t*)buffer->buf;
    buffer_0 = buffer->shape[0]*buffer->itemsize;
    out_0    = out->shape[0];
    
    if(buffer_0 < 8+8+8+8+5*out_0+64*8)
        return !PyErr_Format(PyExc_ValueError, "Flattened buffer is too small!");
    if(graph[1] != out_0)
        return !PyErr_Format(PyExc_ValueError, "Graph reports %llu variables, but \"out\" wants %llu!",
                             graph[1], out_0);
    
    return 0;
}

/**
 * @brief Get buffers for the Python objects.
 * @param obuffer
 * @param buffer
 * @param oout
 * @param out
 * @return 0 if successful; !0 otherwise.
 */

static int             sample_cpt_get_buffers(PyObject* obuffer, Py_buffer* buffer,
                                              PyObject* oout,    Py_buffer* out){
    int err = 0;
    
    if(!PyObject_CheckBuffer(obuffer) ||
       !PyObject_CheckBuffer(oout))
        return !PyErr_Format(PyExc_TypeError, "One of the arguments is not a buffer!");
    
    err |= PyObject_GetBuffer(obuffer, buffer, PyBUF_STRIDES);
    err |= PyObject_GetBuffer(oout,    out,    PyBUF_STRIDES|PyBUF_WRITABLE);
    
    if(!err)
        err = sample_cpt_validate_args(buffer, out);
    
    if(err)
        sample_cpt_release_buffers(buffer, out);
    
    return err;
}

/**
 * @brief Perform the actual sampling from a causal graph.
 * 
 * Intermediate buffers required:
 *              h        float32   (Hgt,8,)       Pre-/post-activation  of leaky ReLU.
 * 
 * 
 *        Dir     | Name   | Dtype   | Shape
 *        ==================================
 * @param [in]     bW0      float32   (M,Ns,Hgt,)
 * @param [in]     bB0      float32   (M,Hgt,)
 * @param [in]     bW1      float32   (Ns,Hgt,)
 * @param [in]     bB1      float32   (Ns,)
 * @param [in]     bN       int32     (M,)
 * @param [in]     bconfig  float32   (M,M,)
 * @param [out]    bout     uint32    (M,bs,)
 * @param [in]     alpha    float32   scalar
 * @param [in,out] prng     PRNG      None
 * @return 0 if successfully completed, !0 otherwise.
 */

static int             do_sample_mlp(Py_buffer* bW0, Py_buffer* bB0, Py_buffer* bW1, Py_buffer* bB1,
                                     Py_buffer* bN,  Py_buffer* bconfig, Py_buffer* bout,
                                     float      alpha,
                                     PRNG_QUADXORSHIFT* prng){
    /**
     * Temporaries
     */
    
    Py_ssize_t i,j,k,l;
    
    __m256i outmask8vi, argmax8vi, cnt8vi, sel8vi;
    __m256  logit8vf, max8vf, cmp8vf, B08vf, W18vf;
    __m256  acc0,   acc1,   acc2,   acc3,   acc4,   acc5,   acc6,   acc7;
    
    const uint32_t* const N   = (const uint32_t*)bN->buf;
    uint32_t*             Nc  = NULL;
    float*                H   = NULL;
    int                   ret = 1;
    
    const Py_ssize_t M   = bN  ->shape[0];
    const Py_ssize_t BS  = bout->shape[1];
    const Py_ssize_t Hgt = bW1 ->shape[1], Hgtu8 = (Hgt+7)&~7;
    
    
    /**
     * Definition of Indexer Macros.
     */
    
    #define W0_INDEXER(k,j,v)  \
        ((const float*)((char*)bW0->buf + (k)*bW0->strides[0] + (Nc[(j)]+(v))*bW0->strides[1]))
    #define B0_INDEXER(k)  \
        ((const float*)((char*)bB0->buf + (k)*bB0->strides[0]))
    #define W1_INDEXER(k,v)  \
        ((const float*)((char*)bW1->buf + (Nc[(k)]+(v))*bW1->strides[0]))
    #define B1_INDEXER(k,v)  \
        ((const float*)((char*)bB1->buf + (Nc[(k)]+(v))*sizeof(float)))
    #define CONFIG_INDEXER(k,j)  \
        ((const float*)((char*)bconfig->buf + (k)*bconfig->strides[0] + (j)*sizeof(float)))
    #define OUT_INDEXER(k,l)  \
        ((uint32_t*)((char*)bout->buf + (k)*bout->strides[0] + (l)*sizeof(uint32_t)))
    
    
    /**
     * Setup.
     * 
     * Frontend computations, such as allocating temporaries and other junk.
     */
    
    if(M<=0 || BS<=0)
        return 0;
    
    for(i=0;i<M;i++)
        if(N[i]<=0)
            return !0;
    
    Nc = _mm_malloc(M*sizeof(*Nc), 32);
    if(!Nc)
        goto earlyexit;
    for(i=1, Nc[0]=0; i<M; i++)
        Nc[i] = N[i]+Nc[i-1];/* Nc=cumsum(N) */
    
    H = _mm_malloc(8*Hgtu8*sizeof(*H), 32);
    if(!H)
        goto earlyexit;
    
    
    /**
     * Loop over batches of 8.
     * 
     * We break up the entire sampling procedure into minibatches of 8 that
     * are 8-way vectorized using AVX, which uses 256-bit (8x32) registers.
     */
    
    for(l=0;l<BS;l+=8){
        outmask8vi = _mm256_set_epi32(l+7<BS ? ~0 : 0, l+6<BS ? ~0 : 0,
                                      l+5<BS ? ~0 : 0, l+4<BS ? ~0 : 0,
                                      l+3<BS ? ~0 : 0, l+2<BS ? ~0 : 0,
                                      l+1<BS ? ~0 : 0, l+0<BS ? ~0 : 0);
        
        /**
         * Loop over the variables k.
         * 
         * Within every minibatch of 8, we sample ancestrally the categorical
         * variables, which are assumed to be sequentially sorted in
         * topological order.
         * 
         * This requires, for each variable,
         *   1. Computing 8 hidden-layer vectors from its ancestor variables and
         *      their selections into W0+B0.
         *   2. Applying the LeakyReLU(alpha=alpha) non-linearity
         *   3. Computing the logits sequentially by applying W1+B1.
         *   4. Online categorical sampling using Gumbel softmax trick.
         *   5. Store integer category into out int32 buffer.
         */
        
        for(k=0;k<M;k++){
            /* Compute the hidden layer for variable k for this batch of 8. */
            for(j=0;j<Hgt;j+=8){
                B08vf = _mm256_loadu_ps(B0_INDEXER(k)+j);
                _mm256_store_ps(H+j*8+ 0, B08vf);
                _mm256_store_ps(H+j*8+ 8, B08vf);
                _mm256_store_ps(H+j*8+16, B08vf);
                _mm256_store_ps(H+j*8+24, B08vf);
                _mm256_store_ps(H+j*8+32, B08vf);
                _mm256_store_ps(H+j*8+40, B08vf);
                _mm256_store_ps(H+j*8+48, B08vf);
                _mm256_store_ps(H+j*8+56, B08vf);
            }
            for(j=0;j<k;j++){
                if(!*CONFIG_INDEXER(k,j))
                    continue;
                
                sel8vi = _mm256_maskload_epi32((int*)OUT_INDEXER(j,l), outmask8vi);
                const float* W0vec0 = W0_INDEXER(k,j,_mm256_extract_epi32(sel8vi, 0));
                const float* W0vec1 = W0_INDEXER(k,j,_mm256_extract_epi32(sel8vi, 1));
                const float* W0vec2 = W0_INDEXER(k,j,_mm256_extract_epi32(sel8vi, 2));
                const float* W0vec3 = W0_INDEXER(k,j,_mm256_extract_epi32(sel8vi, 3));
                const float* W0vec4 = W0_INDEXER(k,j,_mm256_extract_epi32(sel8vi, 4));
                const float* W0vec5 = W0_INDEXER(k,j,_mm256_extract_epi32(sel8vi, 5));
                const float* W0vec6 = W0_INDEXER(k,j,_mm256_extract_epi32(sel8vi, 6));
                const float* W0vec7 = W0_INDEXER(k,j,_mm256_extract_epi32(sel8vi, 7));
                
                for(i=0;i<Hgt;i+=8){
                    _mm256_store_ps(H+i*8+ 0, _mm256_fmadd_ps(_mm256_loadu_ps(&W0vec0[i]), ONE8vf, _mm256_load_ps(H+i*8+ 0)));
                    _mm256_store_ps(H+i*8+ 8, _mm256_fmadd_ps(_mm256_loadu_ps(&W0vec1[i]), ONE8vf, _mm256_load_ps(H+i*8+ 8)));
                    _mm256_store_ps(H+i*8+16, _mm256_fmadd_ps(_mm256_loadu_ps(&W0vec2[i]), ONE8vf, _mm256_load_ps(H+i*8+16)));
                    _mm256_store_ps(H+i*8+24, _mm256_fmadd_ps(_mm256_loadu_ps(&W0vec3[i]), ONE8vf, _mm256_load_ps(H+i*8+24)));
                    _mm256_store_ps(H+i*8+32, _mm256_fmadd_ps(_mm256_loadu_ps(&W0vec4[i]), ONE8vf, _mm256_load_ps(H+i*8+32)));
                    _mm256_store_ps(H+i*8+40, _mm256_fmadd_ps(_mm256_loadu_ps(&W0vec5[i]), ONE8vf, _mm256_load_ps(H+i*8+40)));
                    _mm256_store_ps(H+i*8+48, _mm256_fmadd_ps(_mm256_loadu_ps(&W0vec6[i]), ONE8vf, _mm256_load_ps(H+i*8+48)));
                    _mm256_store_ps(H+i*8+56, _mm256_fmadd_ps(_mm256_loadu_ps(&W0vec7[i]), ONE8vf, _mm256_load_ps(H+i*8+56)));
                }
            }
            
            
            /* Apply non-linearity in-place. */
            leakyrelu(8*Hgtu8, alpha, H, H);
            
            
            /* Loop over category j of variable k for a batch of 8 at a time. */
            max8vf = NINF8vf;
            cnt8vi = argmax8vi = ZERO8vi;
            for(j=0;j<N[k];j++){
                /**
                 * Compute logits.
                 * 
                 * We perform the second layer of the MLP to compute the logits
                 * of variable k's category j, within this batch of 8.
                 */
                
                acc0=acc1=acc2=acc3=acc4=acc5=acc6=acc7=ZERO8vf;
                for(i=0;i<Hgt;i+=8){
                    W18vf = _mm256_loadu_ps(W1_INDEXER(k,j)+i);
                    acc0 = _mm256_fmadd_ps(W18vf, _mm256_load_ps(H+i*8+ 0), acc0);
                    acc1 = _mm256_fmadd_ps(W18vf, _mm256_load_ps(H+i*8+ 8), acc1);
                    acc2 = _mm256_fmadd_ps(W18vf, _mm256_load_ps(H+i*8+16), acc2);
                    acc3 = _mm256_fmadd_ps(W18vf, _mm256_load_ps(H+i*8+24), acc3);
                    acc4 = _mm256_fmadd_ps(W18vf, _mm256_load_ps(H+i*8+32), acc4);
                    acc5 = _mm256_fmadd_ps(W18vf, _mm256_load_ps(H+i*8+40), acc5);
                    acc6 = _mm256_fmadd_ps(W18vf, _mm256_load_ps(H+i*8+48), acc6);
                    acc7 = _mm256_fmadd_ps(W18vf, _mm256_load_ps(H+i*8+56), acc7);
                }
                logit8vf  = hsum_8x8vf(acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7);
                logit8vf  = _mm256_add_ps(logit8vf, _mm256_broadcast_ss(B1_INDEXER(k,j)));
                
                
                /**
                 * Draw Gumbels and update the categorical sample draw.
                 * 
                 * The Gumbel softmax trick consists in adding Gumbel variates to
                 * the logits and selecting their argmax. We can detect that a
                 * new maximum has been reached if logit+Gumbel == max, and if so
                 * we save the current counter value to argmax. The counter is
                 * then incremented regardless.
                 */
                
                logit8vf  = _mm256_add_ps(logit8vf, prng_draw_gumbel8vf(prng));
                max8vf    = _mm256_max_ps(logit8vf, max8vf);
                cmp8vf    = _mm256_cmp_ps(logit8vf, max8vf, _CMP_EQ_OQ);
                argmax8vi = _mm256_castps_si256(
                            _mm256_blendv_ps(_mm256_castsi256_ps(argmax8vi),
                                             _mm256_castsi256_ps(cnt8vi),
                                             cmp8vf));
                cnt8vi    = _mm256_add_epi32(cnt8vi, ONE8vi);
            }
            
            
            /**
             * Store out variable k for this batch of 8.
             * 
             * This is done by writing out 8 uint32 integers under control of a
             * store-mask, preventing us from overwriting data outside our range.
             */
            
            _mm256_maskstore_epi32((int*)OUT_INDEXER(k,l), outmask8vi, argmax8vi);
        }
    }
    ret = 0;
    
    
    /**
     * Undefinition of Indexer Macros.
     */
    
    #undef W0_INDEXER
    #undef B0_INDEXER
    #undef W1_INDEXER
    #undef B1_INDEXER
    #undef CONFIG_INDEXER
    #undef OUT_INDEXER
    
    
    /**
     * Cleanup.
     */
    
    earlyexit:
    _mm_free(Nc);
    _mm_free(H);
    return ret;
}

/**
 * @brief Release exported buffers.
 * @param W0
 * @param B0
 * @param W1
 * @param B1
 * @param N
 * @param config
 * @param out
 * @return 0.
 */

static int             sample_mlp_release_buffers(Py_buffer* W0,
                                                  Py_buffer* B0,
                                                  Py_buffer* W1,
                                                  Py_buffer* B1,
                                                  Py_buffer* N,
                                                  Py_buffer* config,
                                                  Py_buffer* out){
    PyBuffer_Release(out);
    PyBuffer_Release(config);
    PyBuffer_Release(N);
    PyBuffer_Release(B1);
    PyBuffer_Release(W1);
    PyBuffer_Release(B0);
    PyBuffer_Release(W0);
    return 0;
}

/**
 * @brief Validate arguments to sample*() functions.
 * @param W0
 * @param B0
 * @param W1
 * @param B1
 * @param N
 * @param config
 * @param out
 * @return 0 if successfully validated all arguments; !0 otherwise.
 */

static int             sample_mlp_validate_args(Py_buffer* W0,
                                                Py_buffer* B0,
                                                Py_buffer* W1,
                                                Py_buffer* B1, 
                                                Py_buffer* N,
                                                Py_buffer* config,
                                                Py_buffer* out){
    Py_ssize_t W0_0, W0_1, W0_2,
               B0_0, B0_1,
               W1_0, W1_1,
               B1_0,
               N_0,
               config_0, config_1,
               out_0;
    
    if(out->readonly){
        PyErr_SetString(PyExc_ValueError, "out array is read-only!");
        return !0;
    }
    
    if(W0->ndim     != 3 ||
       B0->ndim     != 2 ||
       W1->ndim     != 2 ||
       B1->ndim     != 1 ||
       N->ndim      != 1 ||
       config->ndim != 2 ||
       out->ndim    != 2){
        PyErr_SetString(PyExc_ValueError, "Arrays of incorrect ndim!");
        return !0;
    }
    
    if(W0->itemsize     != 4 ||
       B0->itemsize     != 4 ||
       W1->itemsize     != 4 ||
       B1->itemsize     != 4 ||
       N->itemsize      != 4 ||
       config->itemsize != 4 ||
       out->itemsize    != 4){
        PyErr_SetString(PyExc_ValueError, "Arrays of incorrect dtype!");
        return !0;
    }
    
    if(W0->strides[2]     != 4 || W0->strides[1]%32 || W0->strides[0]%32 ||
       B0->strides[1]     != 4 || B0->strides[0]%32 ||
       W1->strides[1]     != 4 || W1->strides[0]%32 ||
       B1->strides[0]     != 4 ||
       N->strides[0]      != 4 ||
       config->strides[1] != 4 ||
       out->strides[1]    != 4){
        PyErr_SetString(PyExc_ValueError, "Arrays of incorrect and/or misaligned strides!");
        return !0;
    }
    
    W0_0 = W0->shape[0]; W0_1 = W0->shape[1]; W0_2 = W0->shape[2];
    B0_0 = B0->shape[0]; B0_1 = B0->shape[1];
    W1_0 = W1->shape[0]; W1_1 = W1->shape[1];
    B1_0 = B1->shape[0];
    N_0 = N->shape[0];
    config_0 = config->shape[0]; config_1 = config->shape[1];
    out_0 = out->shape[0];
    
    if(N_0  != W0_0 || N_0 != B0_0 || N_0 != config_0 || N_0 != config_1 || N_0 != out_0 || /* M */
       W0_1 != W1_0 || W0_1 != B1_0 ||                                                      /* Ns */
       W0_2 != B0_1 || W0_2 != W1_1){                                                       /* H */
        PyErr_SetString(PyExc_ValueError, "Arrays of mismatched shapes!");
        return !0;
    }
    
    return 0;
}

/**
 * @brief Get buffers for the Python objects.
 * @param oW0
 * @param W0
 * @param oB0
 * @param B0
 * @param oW1
 * @param W1
 * @param oB1
 * @param B1
 * @param oN
 * @param N
 * @param oconfig
 * @param config
 * @param oout
 * @param out
 * @return 0 if successful; !0 otherwise.
 */

static int             sample_mlp_get_buffers(PyObject* oW0,     Py_buffer* W0,
                                              PyObject* oB0,     Py_buffer* B0,
                                              PyObject* oW1,     Py_buffer* W1,
                                              PyObject* oB1,     Py_buffer* B1, 
                                              PyObject* oN,      Py_buffer* N,
                                              PyObject* oconfig, Py_buffer* config,
                                              PyObject* oout,    Py_buffer* out){
    int err = 0;
    
    if(!PyObject_CheckBuffer(oW0)     ||
       !PyObject_CheckBuffer(oB0)     ||
       !PyObject_CheckBuffer(oW1)     ||
       !PyObject_CheckBuffer(oB1)     ||
       !PyObject_CheckBuffer(oN)      ||
       !PyObject_CheckBuffer(oconfig) ||
       !PyObject_CheckBuffer(oout)){
        PyErr_SetString(PyExc_TypeError, "One of the arguments is not a buffer!");
        return !0;
    }
    
    err |= PyObject_GetBuffer(oW0,     W0,     PyBUF_STRIDES);
    err |= PyObject_GetBuffer(oB0,     B0,     PyBUF_STRIDES);
    err |= PyObject_GetBuffer(oW1,     W1,     PyBUF_STRIDES);
    err |= PyObject_GetBuffer(oB1,     B1,     PyBUF_STRIDES);
    err |= PyObject_GetBuffer(oN,      N,      PyBUF_STRIDES);
    err |= PyObject_GetBuffer(oconfig, config, PyBUF_STRIDES);
    err |= PyObject_GetBuffer(oout,    out,    PyBUF_STRIDES|PyBUF_WRITABLE);
    
    if(!err)
        err = sample_mlp_validate_args(W0, B0, W1, B1, N, config, out);
    
    if(err)
        sample_mlp_release_buffers(W0, B0, W1, B1, N, config, out);
    
    return err;
}

/**
 * @brief Compute log-probabilities for a causal graph.
 * 
 * Parallelization Dimensions:
 * 
 *     BS  ⭱64
 *     Hgt ⭱ 8
 *     N[i]⭱ 1
 * 
 * Intermediate buffers required:
 * 
 *     hidden    float32     (BS⭱64/64, Hgt⭱8, 64)    Pre-/post-activation  of leaky ReLU.
 *     mlpout    float32     (BS⭱64/64,     Nm, 64)    The raw MLP output logits
 *     mlpexp    float32     (BS⭱64/64,     Nm, 64)    The stabilized, exponentiated MLP output logits.
 * 
 *              h        float32   (BS/8,Hgt,8,)       Pre-/post-activation  of leaky ReLU.
 *              o        float32   (2,Nm,8,)      The raw activations (1) and their exponential (1)
 * 
 * 
 *        Dir     | Name   | Dtype   | Shape
 *        ==================================
 * @param [in]     bW0      float32   (M,Ns,Hgt,)
 * @param [in]     bB0      float32   (M,Hgt,)
 * @param [in]     bW1      float32   (Ns,Hgt,)
 * @param [in]     bB1      float32   (Ns,)
 * @param [in]     bN       uint32    (M,)
 * @param [in]     bblock   uint32    (M,)
 * @param [in]     bbatch   uint32    (M,bs,)
 * @param [in]     bconfig  float32   (M,M,)
 * @param [out]    bout     float32   (M,bs,)
 * @param [out]    bdW0     float32   (M,Ns,Hgt,)
 * @param [out]    bdB0     float32   (M,Hgt,)
 * @param [out]    bdW1     float32   (Ns,Hgt,)
 * @param [out]    bdB1     float32   (Ns,)
 * @param [in]     alpha    float32   scalar
 * @param [in]     temp     float32   scalar
 * @return 0 if successfully completed, !0 otherwise.
 */

static int             do_logprob_mlp(Py_buffer* bW0,   Py_buffer* bB0,    Py_buffer* bW1,     Py_buffer* bB1,
                                      Py_buffer* bN,    Py_buffer* bblock, Py_buffer* bbatch,  Py_buffer* bconfig, Py_buffer* bout,
                                      Py_buffer* bdW0,  Py_buffer* bdB0,   Py_buffer* bdW1,    Py_buffer* bdB1,
                                      float      alpha, float      temp){
    /**
     * Temporaries
     */
    
    Py_ssize_t i,j,k,l;
    
    __m256i outmask8vi, batch8vi, cnt8vi, sel8vi, chk8vi;
    __m256  logit8vf, explogit8vf, max8vf, cmp8vf, sumexp8vf, logsumexp8vf, invexpsum8vf, logp8vf, B08vf, W18vf, dW18vf;
    __m256  acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
    __m256  tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
    __m256  invBS8vf, invtemp8vf;
    
    const uint32_t* const N     = (const uint32_t*)bN->buf;
    const uint32_t* const block = (const uint32_t*)bblock->buf;
    uint32_t        Nm  = 0;
    uint32_t*       Nc  = NULL;
    float*          H   = NULL;
    float*          buf = NULL;
    int             ret = 1;
    
    const Py_ssize_t M     = bN  ->shape[0];
    const Py_ssize_t BS    = bout->shape[1], BSu64 = (BS+63)&~63;
    const Py_ssize_t Hgt   = bW1 ->shape[1], Hgtu8 = (Hgt+7)&~7;
    
    
    /**
     * Definition of Indexer Macros.
     * 
     * Buffer allocations:
     *     - H:    (BSu8/8,Hgtu8/8,8,8)
     *     - buf: 
     *         - raw: (1,Nm,BSu64)                 (logits)
     *         - exp: (1,Nm,BSu64)                 (At first: exp(logits); Later: Kronecker-exp(logits))
     */
    
    #define W0_INDEXER(k,j,v)  \
        ((const float*)((char*)bW0->buf + (k)*bW0->strides[0] + (Nc[(j)]+(v))*bW0->strides[1]))
    #define B0_INDEXER(k)  \
        ((const float*)((char*)bB0->buf + (k)*bB0->strides[0]))
    #define W1_INDEXER(k,v)  \
        ((const float*)((char*)bW1->buf + (Nc[(k)]+(v))*bW1->strides[0]))
    #define B1_INDEXER(k,v)  \
        ((const float*)((char*)bB1->buf + (Nc[(k)]+(v))*sizeof(float)))
    #define CONFIG_INDEXER(k,j)  \
        ((const float*)((char*)bconfig->buf + (k)*bconfig->strides[0] + (j)*sizeof(float)))
    #define BATCH_INDEXER(k,l)  \
        ((const uint32_t*)((char*)bbatch->buf + (k)*bbatch->strides[0] + (l)*sizeof(uint32_t)))
    #define HPRE_INDEXER(l,i,lm8)  \
        ((float*)((char*)H + ((l)*Hgtu8 + (i)*8 + (lm8)*8)*sizeof(float)))
    #define HPOST_INDEXER(l,i,lm8)  \
        ((float*)((char*)H + (BSu64*Hgtu8 + (l)*Hgtu8 + (i)*8 + (lm8)*8)*sizeof(float)))
    #define RAW_INDEXER(l,j)  \
        ((float*)((char*)buf + ((j)*BSu64+(l))*sizeof(float)))
    #define EXP_INDEXER(l,j)  \
        ((float*)((char*)buf + (((j)+Nm)*BSu64+(l))*sizeof(float)))
    #define OUT_INDEXER(k,l)  \
        ((float*)((char*)bout->buf + (k)*bout->strides[0] + (l)*sizeof(float)))
    #define DW0_INDEXER(k,j,v)  \
        ((float*)((char*)bdW0->buf + (k)*bdW0->strides[0] + (Nc[(j)]+(v))*bdW0->strides[1]))
    #define DB0_INDEXER(k)  \
        ((float*)((char*)bdB0->buf + (k)*bdB0->strides[0]))
    #define DW1_INDEXER(k,v)  \
        ((float*)((char*)bdW1->buf + (Nc[(k)]+(v))*bdW1->strides[0]))
    #define DB1_INDEXER(k,v)  \
        ((float*)((char*)bdB1->buf + (Nc[(k)]+(v))*sizeof(float)))
    
    
    /**
     * Setup.
     * 
     * Frontend computations, such as allocating temporaries and other junk.
     */
    
    if(M<=0 || BS<=0 || temp<=0)
        return 0;
    invBS8vf   = _mm256_set1_ps(1.0/BS);
    invtemp8vf = _mm256_set1_ps(1.0/temp);
    
    for(i=0;i<M;i++)
        if(N[i]<=0)
            return !0;
    
    Nc = _mm_malloc(M*sizeof(*Nc), 32);
    if(!Nc)
        goto earlyexit;
    
    Nc[0] = 0;
    Nm    = N[0];
    for(i=1;i<M;i++){
        Nc[i] = N[i-1] + Nc[i-1];      /* Nc = cumsum(N) */
        Nm    = Nm >= N[i] ? Nm : N[i];/* Nm = max(N)    */
    }
    
    H = _mm_malloc(2*BSu64*Hgtu8*sizeof(float), 64);
    if(!H)
        goto earlyexit;
    memset(H, 0, 2*BSu64*Hgtu8*sizeof(float));
    
    buf = _mm_malloc(2*Nm*BSu64*sizeof(float), 64);
    if(!buf)
        goto earlyexit;
    memset(buf, 0, 2*Nm*BSu64*sizeof(float));
    
    
    /**
     * Loop over the variables k.
     * 
     * We evaluate the independent mechanism of each variable in agreement with
     * its configuration and optionally its gradient to obtain its log-probability.
     */
    
    for(k=0;k<M;k++){
        /* FORWARD PASS BEGINS */
        
        /**
         * Loop over batches of 8.
         * 
         * We break up the entire log-prob computation into minibatches of 8 that
         * are 8-way vectorized using AVX, which uses 256-bit (8x32) registers.
         */
        
        for(l=0;l<BS;l+=8){
            outmask8vi = _mm256_set_epi32(l+7<BS ? ~0 : 0, l+6<BS ? ~0 : 0,
                                          l+5<BS ? ~0 : 0, l+4<BS ? ~0 : 0,
                                          l+3<BS ? ~0 : 0, l+2<BS ? ~0 : 0,
                                          l+1<BS ? ~0 : 0, l+0<BS ? ~0 : 0);
            batch8vi   = _mm256_maskload_epi32((int*)BATCH_INDEXER(k,l), outmask8vi);
            
            /* Compute the hidden layer for variable k for this batch of 8. */
            for(i=0;i<Hgt;i+=8){
                B08vf = _mm256_loadu_ps(B0_INDEXER(k)+i);
                _mm256_store_ps(HPRE_INDEXER(l,i,0), B08vf);
                _mm256_store_ps(HPRE_INDEXER(l,i,1), B08vf);
                _mm256_store_ps(HPRE_INDEXER(l,i,2), B08vf);
                _mm256_store_ps(HPRE_INDEXER(l,i,3), B08vf);
                _mm256_store_ps(HPRE_INDEXER(l,i,4), B08vf);
                _mm256_store_ps(HPRE_INDEXER(l,i,5), B08vf);
                _mm256_store_ps(HPRE_INDEXER(l,i,6), B08vf);
                _mm256_store_ps(HPRE_INDEXER(l,i,7), B08vf);
            }
            for(j=0;j<M;j++){
                if(!*CONFIG_INDEXER(k,j))
                    continue;
                
                sel8vi = _mm256_maskload_epi32((int*)BATCH_INDEXER(j,l), outmask8vi);
                chk8vi = _mm256_add_epi32  (sel8vi, _mm256_set1_epi32(0x80000000));
                chk8vi = _mm256_cmpgt_epi32(chk8vi, _mm256_set1_epi32(0x7FFFFFFF+N[j]));
                if(_mm256_movemask_epi8(chk8vi)){
                    ret = 2;/* Have out-of-bounds value for variable j. Abort. */
                    goto earlyexit;
                }
                const float* W0vec0 = W0_INDEXER(k,j,_mm256_extract_epi32(sel8vi, 0));
                const float* W0vec1 = W0_INDEXER(k,j,_mm256_extract_epi32(sel8vi, 1));
                const float* W0vec2 = W0_INDEXER(k,j,_mm256_extract_epi32(sel8vi, 2));
                const float* W0vec3 = W0_INDEXER(k,j,_mm256_extract_epi32(sel8vi, 3));
                const float* W0vec4 = W0_INDEXER(k,j,_mm256_extract_epi32(sel8vi, 4));
                const float* W0vec5 = W0_INDEXER(k,j,_mm256_extract_epi32(sel8vi, 5));
                const float* W0vec6 = W0_INDEXER(k,j,_mm256_extract_epi32(sel8vi, 6));
                const float* W0vec7 = W0_INDEXER(k,j,_mm256_extract_epi32(sel8vi, 7));
                
                for(i=0;i<Hgt;i+=8){
                    _mm256_store_ps(HPRE_INDEXER(l,i,0), _mm256_fmadd_ps(_mm256_loadu_ps(&W0vec0[i]), ONE8vf, _mm256_load_ps(HPRE_INDEXER(l,i,0))));
                    _mm256_store_ps(HPRE_INDEXER(l,i,1), _mm256_fmadd_ps(_mm256_loadu_ps(&W0vec1[i]), ONE8vf, _mm256_load_ps(HPRE_INDEXER(l,i,1))));
                    _mm256_store_ps(HPRE_INDEXER(l,i,2), _mm256_fmadd_ps(_mm256_loadu_ps(&W0vec2[i]), ONE8vf, _mm256_load_ps(HPRE_INDEXER(l,i,2))));
                    _mm256_store_ps(HPRE_INDEXER(l,i,3), _mm256_fmadd_ps(_mm256_loadu_ps(&W0vec3[i]), ONE8vf, _mm256_load_ps(HPRE_INDEXER(l,i,3))));
                    _mm256_store_ps(HPRE_INDEXER(l,i,4), _mm256_fmadd_ps(_mm256_loadu_ps(&W0vec4[i]), ONE8vf, _mm256_load_ps(HPRE_INDEXER(l,i,4))));
                    _mm256_store_ps(HPRE_INDEXER(l,i,5), _mm256_fmadd_ps(_mm256_loadu_ps(&W0vec5[i]), ONE8vf, _mm256_load_ps(HPRE_INDEXER(l,i,5))));
                    _mm256_store_ps(HPRE_INDEXER(l,i,6), _mm256_fmadd_ps(_mm256_loadu_ps(&W0vec6[i]), ONE8vf, _mm256_load_ps(HPRE_INDEXER(l,i,6))));
                    _mm256_store_ps(HPRE_INDEXER(l,i,7), _mm256_fmadd_ps(_mm256_loadu_ps(&W0vec7[i]), ONE8vf, _mm256_load_ps(HPRE_INDEXER(l,i,7))));
                }
            }
            
            
            /* Apply non-linearity out-of-place. */
            leakyrelu(8*Hgtu8, alpha, HPRE_INDEXER(l,0,0), HPOST_INDEXER(l,0,0));
            
            
            /* Loop over category j of variable k for a batch of 8 at a time. */
            max8vf = NINF8vf;
            for(j=0;j<N[k];j++){
                /**
                 * Compute logits.
                 * 
                 * We perform the second layer of the MLP to compute the logits
                 * of variable k's category j, within this batch of 8.
                 * 
                 * We also track the maximum of the logits, for normalization purposes.
                 */
                
                acc0=acc1=acc2=acc3=acc4=acc5=acc6=acc7=ZERO8vf;
                for(i=0;i<Hgt;i+=8){
                    W18vf = _mm256_loadu_ps(W1_INDEXER(k,j)+i);
                    acc0 = _mm256_fmadd_ps(W18vf, _mm256_load_ps(HPOST_INDEXER(l,i,0)), acc0);
                    acc1 = _mm256_fmadd_ps(W18vf, _mm256_load_ps(HPOST_INDEXER(l,i,1)), acc1);
                    acc2 = _mm256_fmadd_ps(W18vf, _mm256_load_ps(HPOST_INDEXER(l,i,2)), acc2);
                    acc3 = _mm256_fmadd_ps(W18vf, _mm256_load_ps(HPOST_INDEXER(l,i,3)), acc3);
                    acc4 = _mm256_fmadd_ps(W18vf, _mm256_load_ps(HPOST_INDEXER(l,i,4)), acc4);
                    acc5 = _mm256_fmadd_ps(W18vf, _mm256_load_ps(HPOST_INDEXER(l,i,5)), acc5);
                    acc6 = _mm256_fmadd_ps(W18vf, _mm256_load_ps(HPOST_INDEXER(l,i,6)), acc6);
                    acc7 = _mm256_fmadd_ps(W18vf, _mm256_load_ps(HPOST_INDEXER(l,i,7)), acc7);
                }
                logit8vf  = hsum_8x8vf(acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7);
                logit8vf  = _mm256_add_ps(logit8vf, _mm256_broadcast_ss(B1_INDEXER(k,j)));
                logit8vf  = _mm256_mul_ps(logit8vf, invtemp8vf);
                max8vf    = _mm256_max_ps(logit8vf, max8vf);
                _mm256_store_ps(RAW_INDEXER(l,j), logit8vf);
            }
            
            
            /* Stabilized exponentiation */
            sumexp8vf = ZERO8vf;
            for(j=0;j<N[k];j++){
                logit8vf    = _mm256_load_ps(RAW_INDEXER(l,j));
                logit8vf    = _mm256_sub_ps(logit8vf, max8vf);
                explogit8vf = exp8vf(logit8vf);
                sumexp8vf   = _mm256_add_ps(sumexp8vf, explogit8vf);
                _mm256_store_ps(EXP_INDEXER(l,j), explogit8vf);
            }
            
            
            /* Normalization & Store Out */
            invexpsum8vf = _mm256_div_ps(ONE8vf, sumexp8vf);
            logsumexp8vf = _mm256_add_ps(max8vf, log8vf(sumexp8vf));
            logp8vf      = NINF8vf;
            cnt8vi       = ZERO8vi;
            for(j=0;j<N[k];j++){
                logit8vf    = _mm256_load_ps(RAW_INDEXER(l,j));
                explogit8vf = _mm256_load_ps(EXP_INDEXER(l,j));
                logit8vf    = _mm256_sub_ps(logit8vf,    logsumexp8vf);
                explogit8vf = _mm256_mul_ps(explogit8vf, invexpsum8vf);
                
                cmp8vf      = _mm256_castsi256_ps(_mm256_cmpeq_epi32(batch8vi, cnt8vi));
                cnt8vi      = _mm256_add_epi32(cnt8vi, ONE8vi);
                
                logp8vf     = _mm256_blendv_ps(logp8vf, logit8vf, cmp8vf);
                explogit8vf = _mm256_sub_ps(_mm256_and_ps(ONE8vf, cmp8vf), explogit8vf);
                explogit8vf = _mm256_mul_ps(explogit8vf, invtemp8vf);
                
                _mm256_store_ps(EXP_INDEXER(l,j), explogit8vf);
            }
            _mm256_maskstore_ps(OUT_INDEXER(k,l), outmask8vi, logp8vf);
        }
        
        /* FORWARD PASS ENDS */
        
        /* BACKWARD PASS BEGINS */
        
        /**
         * If there is no derivative to take, skip.
         * 
         * If variable k is blocked, skip.
         * 
         * Otherwise, zero out the derivative of invalid elements (from then on
         * the derivatives derived from them will stay zero, by the chain rule)
         * and begin backpropagating.
         */
        
        if(!bdW0 && !bdB0 && !bdW1 && !bdB1)
            continue;
        
        if(block[k])
            continue;
        
        if(BSu64 > BS)
            for(j=0;j<N[k];j++)
                memset(EXP_INDEXER(BS,j), 0, (BSu64-BS)*sizeof(float));
        
        /**
         * The first step is propagating the gradient across the only active
         * element of the logsoftmax, namely the one corresponding to the
         * retained digit i=batch[k,l]. This derivative is
         * 
         *     d logsoftmax(o)_i / d o_k = Kroneker_{ik} - softmax(o)_k
         * 
         * Call this derivative do (shape: N[k],BS).
         * 
         * 
         * LAYER 2
         * 
         * The derivative w.r.t. B1 (shape: N[k]) is simple:
         * 
         *     dB1_{a} = do_{ab} / BS
         * 
         * The derivative w.r.t. W1 (shape: N[k],Hgt), assuming h
         * (shape: Hgt,BS) is:
         * 
         *     dW1_{ca} = h_{ab} do_{cb} / BS
         * 
         * Lastly, the derivative w.r.t. h (shape: Hgt,BS) is
         * 
         *     dh_{ab} = W1_{ca} do_{cb}
         * 
         * which concludes the second layer.
         * 
         * 
         * LAYER 1
         * 
         * The derivative w.r.t. B0 (shape: Hgt) is exactly:
         * 
         *     dB0_{a} = dh_{ab} / BS
         * 
         * The derivatives w.r.t. all individually-selected additive rows
         * of W0 (shape: Hgt) are
         * 
         *     dW0[sel[b]]_{a} = dh_{ab} / BS
         * 
         * which concludes the first layer.
         */
        
        if(bdB1){
            /**
             *     dB1_{a} = do_{ab} / BS
             * 
             * Computed one scalar at a time.
             */
            
            for(j=0;j<N[k];j++){
                acc0=acc1=acc2=acc3=acc4=acc5=acc6=acc7=ZERO8vf;
                for(l=0;l<BS;l+=64){
                    acc0 = _mm256_fmadd_ps(invBS8vf, _mm256_load_ps(EXP_INDEXER(l+ 0,j)), acc0);
                    acc1 = _mm256_fmadd_ps(invBS8vf, _mm256_load_ps(EXP_INDEXER(l+ 8,j)), acc1);
                    acc2 = _mm256_fmadd_ps(invBS8vf, _mm256_load_ps(EXP_INDEXER(l+16,j)), acc2);
                    acc3 = _mm256_fmadd_ps(invBS8vf, _mm256_load_ps(EXP_INDEXER(l+24,j)), acc3);
                    acc4 = _mm256_fmadd_ps(invBS8vf, _mm256_load_ps(EXP_INDEXER(l+32,j)), acc4);
                    acc5 = _mm256_fmadd_ps(invBS8vf, _mm256_load_ps(EXP_INDEXER(l+40,j)), acc5);
                    acc6 = _mm256_fmadd_ps(invBS8vf, _mm256_load_ps(EXP_INDEXER(l+48,j)), acc6);
                    acc7 = _mm256_fmadd_ps(invBS8vf, _mm256_load_ps(EXP_INDEXER(l+56,j)), acc7);
                }
                acc0 = _mm256_fmadd_ps(ONE8vf, acc0, acc4);
                acc1 = _mm256_fmadd_ps(ONE8vf, acc1, acc5);
                acc2 = _mm256_fmadd_ps(ONE8vf, acc2, acc6);
                acc3 = _mm256_fmadd_ps(ONE8vf, acc3, acc7);
                acc0 = _mm256_fmadd_ps(ONE8vf, acc0, acc2);
                acc1 = _mm256_fmadd_ps(ONE8vf, acc1, acc3);
                acc0 = _mm256_add_ps  (acc0, acc1);
                acc0 = _mm256_add_ps  (acc0, _mm256_permute2f128_ps(acc0, acc0, 0x01));/* Lane-crossing. */
                acc0 = _mm256_add_ps  (acc0, _mm256_permute_ps(acc0, 0x4E));/* {1,0,3,2} */
                acc0 = _mm256_add_ps  (acc0, _mm256_permute_ps(acc0, 0xB1));/* {2,3,0,1} */
                *DB1_INDEXER(k,j) += _mm_cvtss_f32(_mm256_castps256_ps128(acc0));
            }
        }
        
        if(bdW1){
            /**
             *     dW1_{ca} = h_{ab} do_{cb} / BS
             * 
             * Computed 8 Hgt's at a time.
             */
            
            for(j=0;j<N[k];j++){
                for(i=0;i<Hgt;i+=8){
                    outmask8vi = _mm256_set_epi32(i+7<Hgt ? ~0 : 0, i+6<Hgt ? ~0 : 0,
                                                  i+5<Hgt ? ~0 : 0, i+4<Hgt ? ~0 : 0,
                                                  i+3<Hgt ? ~0 : 0, i+2<Hgt ? ~0 : 0,
                                                  i+1<Hgt ? ~0 : 0, i+0<Hgt ? ~0 : 0);
                    acc0=acc1=acc2=acc3=acc4=acc5=acc6=acc7=ZERO8vf;
                    for(l=0;l<BS;l+=8){
                        acc0 = _mm256_fmadd_ps(_mm256_load_ps(HPOST_INDEXER(l,i,0)), _mm256_broadcast_ss(EXP_INDEXER(l+0,j)), acc0);
                        acc1 = _mm256_fmadd_ps(_mm256_load_ps(HPOST_INDEXER(l,i,1)), _mm256_broadcast_ss(EXP_INDEXER(l+1,j)), acc1);
                        acc2 = _mm256_fmadd_ps(_mm256_load_ps(HPOST_INDEXER(l,i,2)), _mm256_broadcast_ss(EXP_INDEXER(l+2,j)), acc2);
                        acc3 = _mm256_fmadd_ps(_mm256_load_ps(HPOST_INDEXER(l,i,3)), _mm256_broadcast_ss(EXP_INDEXER(l+3,j)), acc3);
                        acc4 = _mm256_fmadd_ps(_mm256_load_ps(HPOST_INDEXER(l,i,4)), _mm256_broadcast_ss(EXP_INDEXER(l+4,j)), acc4);
                        acc5 = _mm256_fmadd_ps(_mm256_load_ps(HPOST_INDEXER(l,i,5)), _mm256_broadcast_ss(EXP_INDEXER(l+5,j)), acc5);
                        acc6 = _mm256_fmadd_ps(_mm256_load_ps(HPOST_INDEXER(l,i,6)), _mm256_broadcast_ss(EXP_INDEXER(l+6,j)), acc6);
                        acc7 = _mm256_fmadd_ps(_mm256_load_ps(HPOST_INDEXER(l,i,7)), _mm256_broadcast_ss(EXP_INDEXER(l+7,j)), acc7);
                    }
                    acc0 = _mm256_fmadd_ps(ONE8vf, acc0, acc4);
                    acc1 = _mm256_fmadd_ps(ONE8vf, acc1, acc5);
                    acc2 = _mm256_fmadd_ps(ONE8vf, acc2, acc6);
                    acc3 = _mm256_fmadd_ps(ONE8vf, acc3, acc7);
                    acc0 = _mm256_fmadd_ps(ONE8vf, acc0, acc2);
                    acc1 = _mm256_fmadd_ps(ONE8vf, acc1, acc3);
                    acc0 = _mm256_add_ps  (acc0, acc1);
                    float* dW1vec = DW1_INDEXER(k,j)+i;
                    dW18vf = _mm256_maskload_ps(dW1vec, outmask8vi);
                    dW18vf = _mm256_fmadd_ps(invBS8vf, acc0, dW18vf);
                    _mm256_maskstore_ps(dW1vec, outmask8vi, dW18vf);
                }
            }
        }
        
        if(!bdW0 && !bdB0)
            continue;
        
        if(1){
            /**
             *     dh_{ab} = W1_{ca} do_{cb}
             * 
             * Computed 8 Hgt x 8 BS at a time.
             */
            
            for(l=0;l<BS;l+=8){
                for(i=0;i<Hgt;i+=8){
                    acc0=acc1=acc2=acc3=acc4=acc5=acc6=acc7=ZERO8vf;
                    for(j=0;j<N[k];j++){
                        tmp0 = _mm256_load_ps(EXP_INDEXER(l,j));
                        acc0 = _mm256_fmadd_ps(_mm256_broadcast_ss(W1_INDEXER(k,j)+i+0), tmp0, acc0);
                        acc1 = _mm256_fmadd_ps(_mm256_broadcast_ss(W1_INDEXER(k,j)+i+1), tmp0, acc1);
                        acc2 = _mm256_fmadd_ps(_mm256_broadcast_ss(W1_INDEXER(k,j)+i+2), tmp0, acc2);
                        acc3 = _mm256_fmadd_ps(_mm256_broadcast_ss(W1_INDEXER(k,j)+i+3), tmp0, acc3);
                        acc4 = _mm256_fmadd_ps(_mm256_broadcast_ss(W1_INDEXER(k,j)+i+4), tmp0, acc4);
                        acc5 = _mm256_fmadd_ps(_mm256_broadcast_ss(W1_INDEXER(k,j)+i+5), tmp0, acc5);
                        acc6 = _mm256_fmadd_ps(_mm256_broadcast_ss(W1_INDEXER(k,j)+i+6), tmp0, acc6);
                        acc7 = _mm256_fmadd_ps(_mm256_broadcast_ss(W1_INDEXER(k,j)+i+7), tmp0, acc7);
                    }
                    transpose_8x8vf(acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7,
                                    (__m256*)HPOST_INDEXER(l,i,0),
                                    (__m256*)HPOST_INDEXER(l,i,1),
                                    (__m256*)HPOST_INDEXER(l,i,2),
                                    (__m256*)HPOST_INDEXER(l,i,3),
                                    (__m256*)HPOST_INDEXER(l,i,4),
                                    (__m256*)HPOST_INDEXER(l,i,5),
                                    (__m256*)HPOST_INDEXER(l,i,6),
                                    (__m256*)HPOST_INDEXER(l,i,7));
                }
            }
            
            /* Apply gradient of non-linearity partly out-of-place. */
            dleakyreludx(BSu64*Hgtu8, alpha, HPRE_INDEXER(0,0,0), HPOST_INDEXER(0,0,0));
        }
        
        if(bdB0){
            /**
             *     dB0_{a} = dh_{ab} / BS
             * 
             * Computed 8 Hgt at a time.
             */
            
            for(i=0;i<Hgt;i+=8){
                outmask8vi = _mm256_set_epi32(i+7<Hgt ? ~0 : 0, i+6<Hgt ? ~0 : 0,
                                              i+5<Hgt ? ~0 : 0, i+4<Hgt ? ~0 : 0,
                                              i+3<Hgt ? ~0 : 0, i+2<Hgt ? ~0 : 0,
                                              i+1<Hgt ? ~0 : 0, i+0<Hgt ? ~0 : 0);
                acc0=acc1=acc2=acc3=acc4=acc5=acc6=acc7=ZERO8vf;
                for(l=0;l<BS;l+=8){
                    acc0 = _mm256_fmadd_ps(_mm256_load_ps(HPRE_INDEXER(l,i,0)), invBS8vf, acc0);
                    acc1 = _mm256_fmadd_ps(_mm256_load_ps(HPRE_INDEXER(l,i,1)), invBS8vf, acc1);
                    acc2 = _mm256_fmadd_ps(_mm256_load_ps(HPRE_INDEXER(l,i,2)), invBS8vf, acc2);
                    acc3 = _mm256_fmadd_ps(_mm256_load_ps(HPRE_INDEXER(l,i,3)), invBS8vf, acc3);
                    acc4 = _mm256_fmadd_ps(_mm256_load_ps(HPRE_INDEXER(l,i,4)), invBS8vf, acc4);
                    acc5 = _mm256_fmadd_ps(_mm256_load_ps(HPRE_INDEXER(l,i,5)), invBS8vf, acc5);
                    acc6 = _mm256_fmadd_ps(_mm256_load_ps(HPRE_INDEXER(l,i,6)), invBS8vf, acc6);
                    acc7 = _mm256_fmadd_ps(_mm256_load_ps(HPRE_INDEXER(l,i,7)), invBS8vf, acc7);
                }
                acc0 = _mm256_fmadd_ps(ONE8vf, acc0, acc4);
                acc1 = _mm256_fmadd_ps(ONE8vf, acc1, acc5);
                acc2 = _mm256_fmadd_ps(ONE8vf, acc2, acc6);
                acc3 = _mm256_fmadd_ps(ONE8vf, acc3, acc7);
                acc0 = _mm256_fmadd_ps(ONE8vf, acc0, acc2);
                acc1 = _mm256_fmadd_ps(ONE8vf, acc1, acc3);
                acc0 = _mm256_add_ps  (acc0, acc1);
                _mm256_maskstore_ps(DB0_INDEXER(k)+i, outmask8vi, acc0);
            }
        }
        
        if(bdW0){
            /**
             *     dW0[sel[b]]_{a} = dh_{ab} / BS
             * 
             * Computed 8 Hgt at a time.
             */
            
            for(j=0;j<M;j++){
                if(!*CONFIG_INDEXER(k,j))
                    continue;
                
                for(l=0;l<BS;l+=8){
                    sel8vi = _mm256_maskload_epi32((int*)BATCH_INDEXER(j,l), outmask8vi);
                    float* DW0vec0 = DW0_INDEXER(k,j,_mm256_extract_epi32(sel8vi, 0));
                    float* DW0vec1 = DW0_INDEXER(k,j,_mm256_extract_epi32(sel8vi, 1));
                    float* DW0vec2 = DW0_INDEXER(k,j,_mm256_extract_epi32(sel8vi, 2));
                    float* DW0vec3 = DW0_INDEXER(k,j,_mm256_extract_epi32(sel8vi, 3));
                    float* DW0vec4 = DW0_INDEXER(k,j,_mm256_extract_epi32(sel8vi, 4));
                    float* DW0vec5 = DW0_INDEXER(k,j,_mm256_extract_epi32(sel8vi, 5));
                    float* DW0vec6 = DW0_INDEXER(k,j,_mm256_extract_epi32(sel8vi, 6));
                    float* DW0vec7 = DW0_INDEXER(k,j,_mm256_extract_epi32(sel8vi, 7));
                    
                    for(i=0;i<Hgt;i+=8){
                        _mm256_storeu_ps(&DW0vec0[i], _mm256_fmadd_ps(invBS8vf, _mm256_load_ps(HPRE_INDEXER(l,i,0)), _mm256_loadu_ps(&DW0vec0[i])));
                        _mm256_storeu_ps(&DW0vec1[i], _mm256_fmadd_ps(invBS8vf, _mm256_load_ps(HPRE_INDEXER(l,i,1)), _mm256_loadu_ps(&DW0vec1[i])));
                        _mm256_storeu_ps(&DW0vec2[i], _mm256_fmadd_ps(invBS8vf, _mm256_load_ps(HPRE_INDEXER(l,i,2)), _mm256_loadu_ps(&DW0vec2[i])));
                        _mm256_storeu_ps(&DW0vec3[i], _mm256_fmadd_ps(invBS8vf, _mm256_load_ps(HPRE_INDEXER(l,i,3)), _mm256_loadu_ps(&DW0vec3[i])));
                        _mm256_storeu_ps(&DW0vec4[i], _mm256_fmadd_ps(invBS8vf, _mm256_load_ps(HPRE_INDEXER(l,i,4)), _mm256_loadu_ps(&DW0vec4[i])));
                        _mm256_storeu_ps(&DW0vec5[i], _mm256_fmadd_ps(invBS8vf, _mm256_load_ps(HPRE_INDEXER(l,i,5)), _mm256_loadu_ps(&DW0vec5[i])));
                        _mm256_storeu_ps(&DW0vec6[i], _mm256_fmadd_ps(invBS8vf, _mm256_load_ps(HPRE_INDEXER(l,i,6)), _mm256_loadu_ps(&DW0vec6[i])));
                        _mm256_storeu_ps(&DW0vec7[i], _mm256_fmadd_ps(invBS8vf, _mm256_load_ps(HPRE_INDEXER(l,i,7)), _mm256_loadu_ps(&DW0vec7[i])));
                    }
                }
            }
        }
        
        /* BACKWARD PASS ENDS */
    }
    ret = 0;
    
    
    /**
     * Undefinition of Indexer Macros.
     */
    
    #undef W0_INDEXER
    #undef B0_INDEXER
    #undef W1_INDEXER
    #undef B1_INDEXER
    #undef CONFIG_INDEXER
    #undef BATCH_INDEXER
    #undef HPRE_INDEXER
    #undef HPOST_INDEXER
    #undef RAW_INDEXER
    #undef EXP_INDEXER
    #undef OUT_INDEXER
    #undef DW0_INDEXER
    #undef DB0_INDEXER
    #undef DW1_INDEXER
    #undef DB1_INDEXER
    
    
    /**
     * Cleanup.
     */
    
    earlyexit:
    _mm_free(Nc);
    _mm_free(H);
    _mm_free(buf);
    return ret;
}

/**
 * @brief Release exported buffers.
 * 
 * @return 0.
 */

static int             logprob_mlp_release_buffers(Py_buffer* W0,
                                                   Py_buffer* B0,
                                                   Py_buffer* W1,
                                                   Py_buffer* B1,
                                                   Py_buffer* N,
                                                   Py_buffer* block,
                                                   Py_buffer* batch,
                                                   Py_buffer* config,
                                                   Py_buffer* out,
                                                   Py_buffer* dW0,
                                                   Py_buffer* dB0,
                                                   Py_buffer* dW1,
                                                   Py_buffer* dB1){
    PyBuffer_Release(dB1);
    PyBuffer_Release(dW1);
    PyBuffer_Release(dB0);
    PyBuffer_Release(dW0);
    PyBuffer_Release(out);
    PyBuffer_Release(config);
    PyBuffer_Release(batch);
    PyBuffer_Release(block);
    PyBuffer_Release(N);
    PyBuffer_Release(B1);
    PyBuffer_Release(W1);
    PyBuffer_Release(B0);
    PyBuffer_Release(W0);
    return 0;
}

/**
 * @brief Validate arguments to logprob*() functions.
 * 
 * @return 0 if successfully validated all arguments; !0 otherwise.
 */

static int             logprob_mlp_validate_args(Py_buffer* W0,
                                                 Py_buffer* B0,
                                                 Py_buffer* W1,
                                                 Py_buffer* B1,
                                                 Py_buffer* N,
                                                 Py_buffer* block,
                                                 Py_buffer* batch,
                                                 Py_buffer* config,
                                                 Py_buffer* out,
                                                 Py_buffer* dW0,
                                                 Py_buffer* dB0,
                                                 Py_buffer* dW1,
                                                 Py_buffer* dB1){
    Py_ssize_t W0_0, W0_1, W0_2,  dW0_0, dW0_1, dW0_2,
               B0_0, B0_1,        dB0_0, dB0_1,
               W1_0, W1_1,        dW1_0, dW1_1,
               B1_0,              dB1_0,
               N_0,
               block_0,
               batch_0, batch_1,
               config_0, config_1,
               out_0, out_1;
    
    if(out->readonly               ||
       (dW0->obj && dW0->readonly) ||
       (dB0->obj && dB0->readonly) ||
       (dW1->obj && dW1->readonly) ||
       (dB1->obj && dB1->readonly)){
        PyErr_SetString(PyExc_ValueError, "out, dW0, dB0, dW1 or dB1 array(s) are read-only!");
        return !0;
    }
    
    if(W0->ndim     != 3 || (dW0->obj && W0->ndim != dW0->ndim) ||
       B0->ndim     != 2 || (dB0->obj && B0->ndim != dB0->ndim) ||
       W1->ndim     != 2 || (dW1->obj && W1->ndim != dW1->ndim) ||
       B1->ndim     != 1 || (dB1->obj && B1->ndim != dB1->ndim) ||
       N->ndim      != 1 ||
       block->ndim  != 1 ||
       batch->ndim  != 2 ||
       config->ndim != 2 ||
       out->ndim    != 2){
        PyErr_SetString(PyExc_ValueError, "Arrays of incorrect ndim!");
        return !0;
    }
    
    if(W0->itemsize     != 4 || (dW0->obj && W0->itemsize != dW0->itemsize) ||
       B0->itemsize     != 4 || (dB0->obj && B0->itemsize != dB0->itemsize) ||
       W1->itemsize     != 4 || (dW1->obj && W1->itemsize != dW1->itemsize) ||
       B1->itemsize     != 4 || (dB1->obj && B1->itemsize != dB1->itemsize) ||
       N->itemsize      != 4 ||
       block->itemsize  != 4 ||
       batch->itemsize  != 4 ||
       config->itemsize != 4 ||
       out->itemsize    != 4){
        PyErr_SetString(PyExc_ValueError, "Arrays of incorrect dtype!");
        return !0;
    }
    
    if(W0->strides[2]     != 4 || (dW0->obj && dW0->strides[2] != W0->strides[2]) ||
       B0->strides[1]     != 4 || (dB0->obj && dB0->strides[1] != B0->strides[1]) ||
       W1->strides[1]     != 4 || (dW1->obj && dW1->strides[1] != W1->strides[1]) ||
       B1->strides[0]     != 4 || (dB1->obj && dB1->strides[0] != B1->strides[0]) ||
       W0->strides[1]%32       || (dW0->obj && dW0->strides[1]%32) ||
       W0->strides[0]%32       || (dW0->obj && dW0->strides[0]%32) ||
       B0->strides[0]%32       || (dB0->obj && dB0->strides[0]%32) ||
       W1->strides[0]%32       || (dW1->obj && dW1->strides[0]%32) ||
       N->strides[0]      != 4 ||
       block->strides[0]  != 4 ||
       batch->strides[1]  != 4 ||
       config->strides[1] != 4 ||
       out->strides[1]    != 4){
        PyErr_SetString(PyExc_ValueError, "Arrays of incorrect and/or misaligned strides!");
        return !0;
    }
    
    W0_0 = W0->shape[0]; W0_1 = W0->shape[1]; W0_2 = W0->shape[2];
    B0_0 = B0->shape[0]; B0_1 = B0->shape[1];
    W1_0 = W1->shape[0]; W1_1 = W1->shape[1];
    B1_0 = B1->shape[0];
    
    dW0_0 = dW0->obj ? dW0->shape[0] : W0_0; dW0_1 = dW0->obj ? dW0->shape[1] : W0_1; dW0_2 = dW0->obj ? dW0->shape[2] : W0_2;
    dB0_0 = dB0->obj ? dB0->shape[0] : B0_0; dB0_1 = dB0->obj ? dB0->shape[1] : B0_1;
    dW1_0 = dW1->obj ? dW1->shape[0] : W1_0; dW1_1 = dW1->obj ? dW1->shape[1] : W1_1;
    dB1_0 = dB1->obj ? dB1->shape[0] : B1_0;
    
    N_0 = N->shape[0];
    block_0 = block->shape[0];
    batch_0 = batch->shape[0]; batch_1 = batch->shape[1];
    config_0 = config->shape[0]; config_1 = config->shape[1];
    out_0 = out->shape[0]; out_1 = out->shape[1];
    
    if(N_0  != W0_0 || N_0 != B0_0 || N_0 != block_0 || N_0 != batch_0 || N_0 != config_0 || N_0 != config_1 || N_0 != out_0 || /* M */
       batch_1 != out_1 ||                                                                  /* BS */
       W0_1 != W1_0 || W0_1 != B1_0 ||                                                      /* Ns */
       W0_2 != B0_1 || W0_2 != W1_1){                                                       /* H */
        PyErr_SetString(PyExc_ValueError, "Arrays of mismatched shapes!");
        return !0;
    }
    if(dW0_0 != W0_0 || dW0_1 != W0_1 || dW0_2 != W0_2 ||
       dB0_0 != B0_0 || dB0_1 != B0_1 ||
       dW1_0 != W1_0 || dW1_1 != W1_1 ||
       dB1_0 != B1_0){
        PyErr_SetString(PyExc_ValueError, "Gradient array shape does not match parameter shape!");
        return !0;
    }
    
    return 0;
}

/**
 * @brief Get buffers for Python objects.
 * 
 * @return 0 if successful; !0 otherwise.
 */

static int             logprob_mlp_get_buffers(PyObject* oW0,     Py_buffer* W0,
                                               PyObject* oB0,     Py_buffer* B0,
                                               PyObject* oW1,     Py_buffer* W1,
                                               PyObject* oB1,     Py_buffer* B1, 
                                               PyObject* oN,      Py_buffer* N,
                                               PyObject* oblock,  Py_buffer* block,
                                               PyObject* obatch,  Py_buffer* batch,
                                               PyObject* oconfig, Py_buffer* config,
                                               PyObject* oout,    Py_buffer* out,
                                               PyObject* odW0,    Py_buffer* dW0,
                                               PyObject* odB0,    Py_buffer* dB0,
                                               PyObject* odW1,    Py_buffer* dW1,
                                               PyObject* odB1,    Py_buffer* dB1){
    int err = 0;
    
    if(!PyObject_CheckBuffer(oW0)     ||
       !PyObject_CheckBuffer(oB0)     ||
       !PyObject_CheckBuffer(oW1)     ||
       !PyObject_CheckBuffer(oB1)     ||
       !PyObject_CheckBuffer(oN)      ||
       !PyObject_CheckBuffer(oblock)  ||
       !PyObject_CheckBuffer(obatch)  ||
       !PyObject_CheckBuffer(oconfig) ||
       !PyObject_CheckBuffer(oout)){
        PyErr_SetString(PyExc_TypeError, "One of the arguments is not a buffer!");
        return !0;
    }
    
    err |= PyObject_GetBuffer(oW0,     W0,     PyBUF_STRIDES);
    err |= PyObject_GetBuffer(oB0,     B0,     PyBUF_STRIDES);
    err |= PyObject_GetBuffer(oW1,     W1,     PyBUF_STRIDES);
    err |= PyObject_GetBuffer(oB1,     B1,     PyBUF_STRIDES);
    err |= PyObject_GetBuffer(oN,      N,      PyBUF_STRIDES);
    err |= PyObject_GetBuffer(oblock,  block,  PyBUF_STRIDES);
    err |= PyObject_GetBuffer(obatch,  batch,  PyBUF_STRIDES);
    err |= PyObject_GetBuffer(oconfig, config, PyBUF_STRIDES);
    err |= PyObject_GetBuffer(oout,    out,    PyBUF_STRIDES|PyBUF_WRITABLE);
    if(PyObject_CheckBuffer(odW0))
        err |= PyObject_GetBuffer(odW0,    dW0,    PyBUF_STRIDES|PyBUF_WRITABLE);
    else
        dW0->obj = NULL;
    
    if(PyObject_CheckBuffer(odB0))
        err |= PyObject_GetBuffer(odB0,    dB0,    PyBUF_STRIDES|PyBUF_WRITABLE);
    else
        dB0->obj = NULL;
    
    if(PyObject_CheckBuffer(odW1))
        err |= PyObject_GetBuffer(odW1,    dW1,    PyBUF_STRIDES|PyBUF_WRITABLE);
    else
        dW1->obj = NULL;
    
    if(PyObject_CheckBuffer(odB1))
        err |= PyObject_GetBuffer(odB1,    dB1,    PyBUF_STRIDES|PyBUF_WRITABLE);
    else
        dB1->obj = NULL;
    
    if(!err)
        err = logprob_mlp_validate_args(W0, B0, W1, B1, N, block, batch,
                                        config, out, dW0, dB0, dW1, dB1);
    
    if(err)
        logprob_mlp_release_buffers(W0, B0, W1, B1, N, block, batch,
                                    config, out, dW0, dB0, dW1, dB1);
    
    return err;
}




/* Python Method Definitions */

/**
 * @brief Seed module-global PRNG.
 * @return None
 */

static PyObject* py_seed(PyObject* self, PyObject* args, PyObject* kwargs){
    /* Seed */
    unsigned long long seed=0;
    
    /* Parse arguments */
    static char* kwargs_list[] = {"seed", NULL};
    if(!PyArg_ParseTupleAndKeywords(args, kwargs, "K", kwargs_list, &seed))
        return NULL;
    
    /* Seed the PRNG with it and burn in the generator. */
    prng_seed_and_burnin(GLOBAL_PRNG, seed);
    
    /* Return */
    Py_INCREF(Py_None);
    return Py_None;
}

/**
 * @brief Sample from the (ground-truth) distribution, defined by CPT.
 */

static PyObject* py_sample_cpt(PyObject* self, PyObject* args, PyObject* kwargs){
    PyObject* obif, *obuffer, *oout;
    Py_buffer bbuffer, bout;
    PRNG_QUADXORSHIFT _PRNG, *PRNG=&_PRNG;
    int       err   = 0;
    PyObject* ret   = NULL;
    
    
    /* Parse arguments */
    static char* kwargs_list[] = {"bif", "out", NULL};
    if(!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwargs_list,
                                    &obif, &oout))
        return NULL;
    
    
    /* Get .buffer attribute */
    obuffer = PyObject_GetAttrString(obif, "buffer");
    if(!obuffer)
        return NULL;
    
    
    /* Retrieve exported buffers and validate */
    if(sample_cpt_get_buffers(obuffer, &bbuffer,
                              oout,    &bout) != 0){
        Py_DecRef(obuffer);
        return NULL;
    }
    
    
    /* Seed local PRNG from global PRNG while remaining under GIL */
    prng_seed(PRNG, prng_draw_u64(GLOBAL_PRNG));
    
    
    /* Drop GIL and perform sampling. */
    Py_BEGIN_ALLOW_THREADS
    err = do_sample_cpt(&bbuffer, &bout, PRNG);
    Py_END_ALLOW_THREADS
    if(!err){
        Py_INCREF(Py_None);
        ret = Py_None;
    }else{
        PyErr_SetString(PyExc_RuntimeError, "Error during sampling!");
    }
    
    
    /* Release exported buffers */
    sample_cpt_release_buffers(&bbuffer, &bout);
    Py_DecRef(obuffer);
    
    
    /* Exit */
    return ret;
}

/**
 * @brief Sample from the (ground-truth) distribution, defined by MLP.
 */

static PyObject* py_sample_mlp(PyObject* self, PyObject* args, PyObject* kwargs){
    PyObject* oW0, *oB0, *oW1, *oB1, *oN, *oconfig, *oout;
    Py_buffer bW0, bB0,  bW1,  bB1,  bN,  bconfig,  bout;
    PRNG_QUADXORSHIFT _PRNG, *PRNG=&_PRNG;
    float     alpha = 0.1;
    int       err   = 0;
    PyObject* ret   = NULL;
    
    
    /* Parse arguments */
    static char* kwargs_list[] = {"W0", "B0", "W1", "B1", "N",
                                  "config", "out", "alpha", NULL};
    if(!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOOOOO|f", kwargs_list,
                                    &oW0, &oB0, &oW1, &oB1, &oN,
                                    &oconfig, &oout, &alpha))
        return NULL;
    
    
    /* Retrieve exported buffers and validate */
    if(sample_mlp_get_buffers(oW0,     &bW0,
                              oB0,     &bB0,
                              oW1,     &bW1,
                              oB1,     &bB1,
                              oN,      &bN,
                              oconfig, &bconfig,
                              oout,    &bout) != 0)
        return NULL;
    
    
    /* Seed local PRNG from global PRNG while remaining under GIL */
    prng_seed(PRNG, prng_draw_u64(GLOBAL_PRNG));
    
    
    /* Drop GIL and perform sampling. */
    Py_BEGIN_ALLOW_THREADS
    err = do_sample_mlp(&bW0, &bB0, &bW1, &bB1, &bN, &bconfig, &bout, alpha, PRNG);
    Py_END_ALLOW_THREADS
    if(!err){
        Py_INCREF(Py_None);
        ret = Py_None;
    }else{
        PyErr_SetString(PyExc_RuntimeError, "Error during sampling!");
    }
    
    
    /* Release exported buffers */
    sample_mlp_release_buffers(&bW0, &bB0, &bW1, &bB1, &bN, &bconfig, &bout);
    
    
    /* Exit */
    return ret;
}

/**
 * @brief Compute the log-probabilites for the distribution.
 */

static PyObject* py_logprob_mlp(PyObject* self, PyObject* args, PyObject* kwargs){
    PyObject* oW0,  *oB0,  *oW1,  *oB1, *oN, *oblock, *obatch, *oconfig, *oout;
    Py_buffer bW0,  bB0,   bW1,   bB1,  bN,  bblock,  bbatch,  bconfig,  bout;
    Py_buffer bdW0, bdB0,  bdW1,  bdB1;
    float     alpha = 0.1, temp = 1.0;
    int       err   = 0;
    PyObject* odW0  = Py_None, *odB0=Py_None, *odW1=Py_None, *odB1=Py_None;
    PyObject* ret   = NULL;
    
    
    /* Parse arguments */
    static char* kwargs_list[] = {"W0", "B0", "W1", "B1", "N", "block", "batch",
                                  "config", "out", "dW0", "dB0", "dW1", "dB1",
                                  "alpha", "temp", NULL};
    if(!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOOOOOOO|OOOOff", kwargs_list,
                                    &oW0, &oB0, &oW1, &oB1, &oN, &oblock, &obatch,
                                    &oconfig, &oout, &odW0, &odB0, &odW1, &odB1,
                                    &alpha, &temp))
        return NULL;
    
    
    /* Retrieve exported buffers and validate */
    if(logprob_mlp_get_buffers(oW0,     &bW0,
                               oB0,     &bB0,
                               oW1,     &bW1,
                               oB1,     &bB1,
                               oN,      &bN,
                               oblock,  &bblock,
                               obatch,  &bbatch,
                               oconfig, &bconfig,
                               oout,    &bout,
                               odW0,    &bdW0,
                               odB0,    &bdB0,
                               odW1,    &bdW1,
                               odB1,    &bdB1) != 0)
        return NULL;
    
    
    /* Drop GIL and compute log-probabilities. */
    Py_BEGIN_ALLOW_THREADS
    err = do_logprob_mlp(&bW0, &bB0, &bW1, &bB1, &bN, &bblock, &bbatch,
                         &bconfig, &bout,
                         bdW0.obj ? &bdW0 : NULL,
                         bdB0.obj ? &bdB0 : NULL,
                         bdW1.obj ? &bdW1 : NULL,
                         bdB1.obj ? &bdB1 : NULL,
                         alpha, temp);
    Py_END_ALLOW_THREADS
    if(!err){
        Py_INCREF(Py_None);
        ret = Py_None;
    }else if(err == 2){
        PyErr_SetString(PyExc_RuntimeError, "Out-of-bounds discrete variable during log-probability computation!");
    }else{
        PyErr_SetString(PyExc_RuntimeError, "Error during log-probability computation!");
    }
    
    
    /* Release exported buffers */
    logprob_mlp_release_buffers(&bW0, &bB0, &bW1, &bB1, &bN, &bblock, &bbatch,
                                &bconfig, &bout, &bdW0, &bdB0, &bdW1, &bdB1);
    
    
    /* Exit */
    return ret;
}



/**
 * Python Module Definition
 */

static const char  _causal_MODULE_DOC[] = "C implementation of categorical samplers.";
static PyMethodDef _causal_METHODS[] = {
    {"seed", (PyCFunction)py_seed, METH_VARARGS|METH_KEYWORDS,
     "seed(uint64 seed)"},
    {"sample_cpt", (PyCFunction)py_sample_cpt, METH_VARARGS|METH_KEYWORDS,
     "sample_cpt(bif, out)"},
    {"sample_mlp", (PyCFunction)py_sample_mlp, METH_VARARGS|METH_KEYWORDS,
     "sample_mlp(W0, B0, W1, B1, N, config, out, alpha)"},
    {"logprob_mlp", (PyCFunction)py_logprob_mlp, METH_VARARGS|METH_KEYWORDS,
     "logprob_mlp(W0, B0, W1, B1, N, block, batch, config, out, "
     "dW0, dB0, dW1, dB1, alpha, temp)"},
    {NULL},  /* Sentinel */
};
static PyModuleDef _causal_MODULE_DEF = {
    PyModuleDef_HEAD_INIT,
    "_causal",           /* m_name */
    _causal_MODULE_DOC,  /* m_doc */
    -1,                  /* m_size */
    _causal_METHODS,     /* m_methods */
    NULL,                /* m_reload */
    NULL,                /* m_traverse */
    NULL,                /* m_clear */
    NULL,                /* m_free */
};
PyMODINIT_FUNC PyInit__causal(void){
    PyObject* m;
    
    #if !defined(__APPLE__) || (__clang_major__ > 8)
    /* Check CPU features */
    __builtin_cpu_init();
    if(!__builtin_cpu_supports("sse")  ||
       !__builtin_cpu_supports("sse2") ||
       !__builtin_cpu_supports("avx")  ||
       !__builtin_cpu_supports("avx2") ||
       !__builtin_cpu_supports("fma")){
        PyErr_SetString(PyExc_ImportError, "Rejecting load of the module \"causal._causal\" on this CPU. Requires SSE, SSE2, AVX, AVX2 and FMA enabled.");
        return NULL;
    }
    #endif
    
    
    /* Initialize module-private PRNG */
    prng_seed_and_burnin(GLOBAL_PRNG, 0);
    
    
    /* Initialize Python module structures */
    m = PyModule_Create(&_causal_MODULE_DEF);
    if(!m){
        return NULL;
    }else{
        return m;
    }
}


/* End Extern "C" Guard */
#ifdef __cplusplus
}
#endif
