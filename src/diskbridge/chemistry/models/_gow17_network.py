from __future__ import annotations

import numpy as np

I_HEP = 0
I_OHX = 1
I_CHX = 2
I_CO = 3
I_CP = 4
I_HCOP = 5
I_H2 = 6
I_HP = 7
I_H3P = 8
I_H2P = 9
I_SP = 10
I_SIP = 11
I_OP = 12

I_CO_ICE = 13

N_SPEC = 13
N_Y = 14

G_SI = 0
G_S = 1
G_C = 2
G_O = 3
G_HE = 4
G_E = 5
G_H = 6
N_GHOST = 7

GHOST_OFFSET = N_Y

X_HE_STD = 0.1
X_S_STD = 0.0
X_SI_STD = 1.7e-6

TEMP_COLL_K = 7.0e2

N_CR = 8
ICR_H2 = 0
ICR_HE = 1
ICR_H = 2

IN_CR = np.array([I_H2, G_HE + GHOST_OFFSET, G_H + GHOST_OFFSET, G_C + GHOST_OFFSET, I_CO, I_CO, G_S + GHOST_OFFSET, G_SI + GHOST_OFFSET], dtype=np.int64)
OUT_CR = np.array([I_H2P, I_HEP, I_HP, I_CP, G_O + GHOST_OFFSET, I_HCOP, I_SP, I_SIP], dtype=np.int64)
K_CR_BASE = np.array([2.0, 1.1, 1.0, 520.0, 92.0, 6.52, 2040.0, 8400.0], dtype=np.float64)

N_2BODY = 35
I2BODY_H2_H = 15
I2BODY_H2_H2 = 16
I2BODY_H_E = 17

IN_2BODY1 = np.array(
    [
        I_H3P,
        I_H3P,
        I_H3P,
        I_HEP,
        I_HEP,
        I_CP,
        I_CP,
        I_CHX,
        I_OHX,
        I_HEP,
        I_H3P,
        I_CP,
        I_HCOP,
        I_H2P,
        I_HP,
        I_H2,
        I_H2,
        G_H + GHOST_OFFSET,
        I_H3P,
        I_HEP,
        I_CHX,
        I_OHX,
        I_CP,
        I_SP,
        I_CP,
        I_SIP,
        I_CP,
        I_H3P,
        I_HEP,
        I_H2P,
        I_HP,
        I_OP,
        I_OP,
        I_OP,
        I_CP,
    ],
    dtype=np.int64,
)

IN_2BODY2 = np.array(
    [
        G_C + GHOST_OFFSET,
        G_O + GHOST_OFFSET,
        I_CO,
        I_H2,
        I_CO,
        I_H2,
        I_OHX,
        G_O + GHOST_OFFSET,
        G_C + GHOST_OFFSET,
        G_E + GHOST_OFFSET,
        G_E + GHOST_OFFSET,
        G_E + GHOST_OFFSET,
        G_E + GHOST_OFFSET,
        I_H2,
        G_E + GHOST_OFFSET,
        G_H + GHOST_OFFSET,
        I_H2,
        G_E + GHOST_OFFSET,
        G_E + GHOST_OFFSET,
        I_H2,
        G_H + GHOST_OFFSET,
        G_O + GHOST_OFFSET,
        I_H2,
        G_E + GHOST_OFFSET,
        G_S + GHOST_OFFSET,
        G_E + GHOST_OFFSET,
        G_SI + GHOST_OFFSET,
        G_O + GHOST_OFFSET,
        I_OHX,
        G_H + GHOST_OFFSET,
        G_O + GHOST_OFFSET,
        G_H + GHOST_OFFSET,
        I_H2,
        I_H2,
        I_H2,
    ],
    dtype=np.int64,
)

OUT_2BODY1 = np.array(
    [
        I_CHX,
        I_OHX,
        I_HCOP,
        I_HP,
        I_CP,
        I_CHX,
        I_HCOP,
        I_CO,
        I_CO,
        G_HE + GHOST_OFFSET,
        I_H2,
        G_C + GHOST_OFFSET,
        I_CO,
        I_H3P,
        G_H + GHOST_OFFSET,
        G_H + GHOST_OFFSET,
        I_H2,
        I_HP,
        G_H + GHOST_OFFSET,
        I_H2P,
        I_H2,
        G_O + GHOST_OFFSET,
        G_C + GHOST_OFFSET,
        G_S + GHOST_OFFSET,
        G_S + GHOST_OFFSET,
        G_SI + GHOST_OFFSET,
        I_SIP,
        I_H2,
        I_OP,
        I_HP,
        I_OP,
        I_HP,
        I_OHX,
        G_O + GHOST_OFFSET,
        I_CHX,
    ],
    dtype=np.int64,
)

OUT_2BODY2 = np.array(
    [
        I_H2,
        I_H2,
        I_H2,
        G_HE + GHOST_OFFSET,
        G_O + GHOST_OFFSET,
        G_H + GHOST_OFFSET,
        G_H + GHOST_OFFSET,
        G_H + GHOST_OFFSET,
        G_H + GHOST_OFFSET,
        G_H + GHOST_OFFSET,
        G_H + GHOST_OFFSET,
        G_H + GHOST_OFFSET,
        G_H + GHOST_OFFSET,
        G_H + GHOST_OFFSET,
        G_H + GHOST_OFFSET,
        G_H + GHOST_OFFSET,
        G_H + GHOST_OFFSET,
        G_H + GHOST_OFFSET,
        G_E + GHOST_OFFSET,
        G_H + GHOST_OFFSET,
        G_HE + GHOST_OFFSET,
        G_C + GHOST_OFFSET,
        G_H + GHOST_OFFSET,
        G_H + GHOST_OFFSET,
        G_H + GHOST_OFFSET,
        G_C + GHOST_OFFSET,
        G_H + GHOST_OFFSET,
        G_C + GHOST_OFFSET,
        G_O + GHOST_OFFSET,
        G_HE + GHOST_OFFSET,
        I_H2,
        G_H + GHOST_OFFSET,
        G_O + GHOST_OFFSET,
        G_H + GHOST_OFFSET,
        G_H + GHOST_OFFSET,
        G_H + GHOST_OFFSET,
    ],
    dtype=np.int64,
)

K2_TEXP = np.array(
    [
        0.0,
        -0.190,
        0.0,
        0.0,
        0.0,
        -1.3,
        0.0,
        0.0,
        -0.339,
        -0.5,
        -0.52,
        0.0,
        -0.64,
        0.042,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.52,
        0.0,
        0.26,
        0.0,
        -1.3,
        -0.59,
        0.0,
        -0.62,
        0.0,
        -0.190,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float64,
)

K2_BASE = np.array(
    [
        1.00,
        1.99e-9,
        1.7e-9,
        1.26e-13,
        1.6e-9,
        3.3e-13 * 0.7,
        1.00,
        7.0e-11,
        7.95e-10,
        1.0e-11,
        4.54e-7,
        1.00,
        1.06e-5,
        1.76e-9,
        2.753e-14,
        1.00,
        1.00,
        1.00,
        8.46e-7,
        7.20e-15,
        2.81e-11,
        3.5e-11,
        3.3e-13 * 0.3,
        1.6e-10,
        5e-11,
        1.46e-10,
        2.1e-9,
        1.99e-9,
        1.00,
        6.4e-10,
        1.00,
        1.00,
        1.6e-9,
        1.6e-9,
        0.0,
    ],
    dtype=np.float64,
)

A_KCHX = 1.04e-9
N_KCHX = 2.31e-3
C_KCHX = np.array([3.4e-8, 6.97e-9, 1.31e-7, 1.51e-4], dtype=np.float64)
TI_KCHX = np.array([7.62, 1.38, 2.66e1, 8.11e3], dtype=np.float64)

N_PH = 7
IPH_C = 0
IPH_CO = 2
IPH_H2 = 4

IN_PH = np.array([G_C + GHOST_OFFSET, I_CHX, I_CO, I_OHX, I_H2, G_S + GHOST_OFFSET, G_SI + GHOST_OFFSET], dtype=np.int64)
OUT_PH1 = np.array([I_CP, G_C + GHOST_OFFSET, G_C + GHOST_OFFSET, G_O + GHOST_OFFSET, G_H + GHOST_OFFSET, I_SP, I_SIP], dtype=np.int64)
KPH_BASE = np.array([3.5e-10, 9.1e-10, 2.4e-10, 3.8e-10, 5.7e-11, 6e-10, 4.5e-9], dtype=np.float64)
KPH_AVFAC = np.array([3.76, 2.12, 3.88, 2.66, 4.18, 3.10, 2.61], dtype=np.float64)

N_GR = 6
IGR_H = 0

IN_GR = np.array([G_H + GHOST_OFFSET, I_HP, I_CP, I_HEP, I_SP, I_SIP], dtype=np.int64)
OUT_GR = np.array([I_H2, G_H + GHOST_OFFSET, G_C + GHOST_OFFSET, G_HE + GHOST_OFFSET, G_S + GHOST_OFFSET, G_SI + GHOST_OFFSET], dtype=np.int64)

C_HP = np.array([12.25, 8.074e-6, 1.378, 5.087e2, 1.586e-2, 0.4723, 1.102e-5], dtype=np.float64)
C_CP = np.array([45.58, 6.089e-3, 1.128, 4.331e2, 4.845e-2, 0.8120, 1.333e-4], dtype=np.float64)
C_HEP = np.array([5.572, 3.185e-7, 1.512, 5.115e3, 3.903e-7, 0.4956, 5.494e-7], dtype=np.float64)
C_SP = np.array([3.064, 7.769e-5, 1.319, 1.087e2, 3.475e-1, 0.4790, 4.689e-2], dtype=np.float64)
C_SIP = np.array([2.166, 5.678e-8, 1.874, 4.375e4, 1.635e-6, 0.8964, 7.538e-5], dtype=np.float64)
