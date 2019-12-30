# -*- coding: utf-8 -*-
import torch


def uniform_ultra_(x):
    """
    Ultra-stable Uniform(0, 1) sampling on the closed interval [0, 1].
    
    For each element, draw a random exponent from the geometric distribution
    [0, 1, 2, ...) with p=0.5 and a random mantissa on the range [0.5, 1.0]
    and multiply the mantissa by 2 raised to the selected exponent.
    
    This routine can return
    
       - Every            float16 on the range [0, 1].
       - Every normalized float32 on the range [0, 1].
       - Every            float64 on the range [2**-126, 1], plus 0.
    
    The true probability of sampling an exact IEEE Std 754 0.0 from a U(0,1)
    distribution is much lower than that of sampling an exact IEEE Std 754 1.0.
    The reason why is that the floating-point numbers are far denser near 0
    than near 1, and this explains why rounding collapses far more real numbers
    to a single representable floating-point number near 1.
    
    Because a probability of 2**-126 is considered essentially an impossibility
    even by cryptographers and an acceptable risk, we will be satisfied with
    only returning floating-point numbers equal to 0 or greater than 2**-126,
    on the theory that the probability of actually sampling a number in-between
    is so low that the event will never occur within multiple human lifetimes,
    and so their probability can be assigned to exact 0.
    
    The true probabilities of drawing a 0 and 1 for a given floating-point
    format, and the probability of this routine in fact drawing them are
    given in the table below. Probabilities below 2**-100 can generally be
    considered equal to 0 for all intents and purposes.
    
         IEEE754 |        True         |     This Routine    |
         Format  |  P(x=0)  |  P(x=1)  |  P(x=0)  |  P(x=1)  |
         ========+==========+==========+==========+==========|
         float16 | 2**  -25 |  2**-12  | 2**  -25 |  2**-12  |
         float32 | 2** -150 |  2**-25  | 2** -126 |  2**-25  |
         float64 | 2**-1075 |  2**-54  | 2** -126 |  2**-54  |
    """
    
    m0 = torch.empty_like(x, dtype=torch.int64).random_(-2**62, +2**62)
    m1 = torch.empty_like(x, dtype=torch.int64).random_(-2**62, +2**62)
    m2 = torch.empty_like(x, dtype=torch.int64).random_()
    m0 = (m0&-m0).float()
    m1 = (m1&-m1).float()
    z0 = m0.eq(0)
    z1 = m1.eq(0)
    m0 = torch.where(z0, torch.full_like(m0, 2.**-63), m0.reciprocal())
    m1 = torch.where(z1, m1, m1.reciprocal())
    m1 = torch.where(z0, m1, torch.ones_like(m1))
    m  = m0*m1
    
    """
    At this point, the exponent is determined; m can be zero, or any power of
    2 from 2**0 down to 2**-125 inclusive.
    
    We are yet, however, to elect a normalized mantissa on the range [.5, 1].
    This is not entirely trivial because we must trigger rounding-to-nearest
    without a tie-to-even; Ties-to-even induce a small bias for 0 in the LSB.
    At the same time, the extremes (.5 and 1) must be generated with half
    the probability of the other numbers, as they will receive "votes" from
    roundings in the octaves above and below as well.
    
    The trick to do this is to set both the MSB and the LSB of the random
    integer to 1 before conversion to floating-point. The MSB set to 1
    normalizes the number to the form 1.xxxxxxx, and the LSB equal to 1
    prevents any ties from occuring, ensuring that all pseudo-ties between a
    lower-even and upper-odd mantissa break towards the upper-odd mantissa,
    thereby avoiding the bias for even mantissas (lower-odd and upper-even
    already rounded upwards).
    """
    
    if   x.dtype == torch.float16:
        m2 = (m2 >> 48) | +0x8001            # Sets bit 15 and 0
        x.copy_(m2).mul_(.5**16).mul_(m.to(x))
    elif x.dtype == torch.float32:
        m2 = (m2 >> 32) | +0x80000001        # Sets bit 31 and 0
        x.copy_(m2).mul_(.5**32).mul_(m.to(x))
    elif x.dtype == torch.float64:
        m2 = (m2 >>  4) | +0x800000000000001 # Sets bit 59 and 0
        x.copy_(m2).mul_(.5**60).mul_(m.to(x))
    else:
        raise ValueError("Only floating-point types supported!")
    
    return x


def loguniform_ultra_(x):
    """
    Ultra-stable log-Uniform(0, 1) sampling on the semi-open interval (0, 1].
    
    This routine will never return a logarithm greater than 0 or lesser than
    log(2**--127), for the same reasons that uniform_ultra_() never returns a
    non-zero number between 0 and 2**-126: We consider it a practical
    impossibility. Although it is possible, returning a 0 from this routine has
    exactly half the probability of returning a 0 from uniform_ultra_().
    
    The justification for never returning an infinity is that the probability
    of drawing any given *real* number from a uniform distribution is 0, and
    this includes the extremas 0 and 1. However, whereas "many" *real* numbers
    very near 1 have a *floating-point* logarithm of exactly 0.0 (even if they
    are extremely unlikely), the probability of drawing a *real* number that
    has a *floating-point* logarithm below FLT_MIN is, for float32, equal to
    exp(-2**127), which is so much smaller than 2**-126 that it will
    NEVER occur in the known Universe, within as many lifetimes as you care to
    name, with all compute power within it dedicated to random number
    generation.
    """
    
    uniform_ultra_(x).mul_(-0.5).log1p_()
    m0 = torch.empty_like(x, dtype=torch.int64).random_(-2**62, +2**62)
    m1 = torch.empty_like(x, dtype=torch.int64).random_(-2**62, +2**62)
    m2 = torch.empty_like(x, dtype=torch.int64).random_()
    m0 = (m0&-m0).float()
    m1 = (m1&-m1).float()
    z0 = m0.eq(0)
    z1 = m1.eq(0)
    m0 = torch.where(z0, torch.full_like(m0, 2.**-63), m0.reciprocal())
    m1 = torch.where(z1, torch.full_like(m1, 2.**-63), m1.reciprocal())
    m1 = torch.where(z0, m1, torch.ones_like(m1))
    x.add_(m0.mul_(m1).log_().to(x))
    return x


def log2uniform_ultra_(x):
    """
    Ultra-stable log2-Uniform(0, 1) sampling on the semi-open interval (0, 1].
    """
    return loguniform_ultra_(x).mul_(1.4426950408889634) # 1/log(2)


