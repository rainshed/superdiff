from dataclasses import dataclass
import math


@dataclass
class PottsParams:
    z: float
    J: list
    h: list


def matvec2(M, v):
    return [M[0][0] * v[0] + M[0][1] * v[1], M[1][0] * v[0] + M[1][1] * v[1]]


def self_consistency_map(m, p, beta):
    Jm = matvec2(p.J, m)
    B = [p.h[0] + p.z * Jm[0], p.h[1] + p.z * Jm[1]]
    bmax = max(B[0], B[1], 0.0)
    w1 = math.exp(beta * (B[0] - bmax))
    w2 = math.exp(beta * (B[1] - bmax))
    w3 = math.exp(beta * (0.0 - bmax))
    Z = w1 + w2 + w3
    return [w1 / Z, w2 / Z]


def free_energy_density(m, p, beta):
    Jm = matvec2(p.J, m)
    e_corr = 0.5 * p.z * (m[0] * Jm[0] + m[1] * Jm[1])
    B = [p.h[0] + p.z * Jm[0], p.h[1] + p.z * Jm[1]]
    energies = [-B[0], -B[1], 0.0]
    min_e = min(energies)
    z_shift = sum(math.exp(-beta * (e - min_e)) for e in energies)
    log_z = -beta * min_e + math.log(z_shift)
    return e_corr - log_z / beta


def solve_fixed_points_for_T(p, T):
    beta = 1.0 / T
    roots = []
    steps = 10
    inits = []
    for i in range(steps + 1):
        for j in range(steps + 1 - i):
            inits.append([i / steps, j / steps])
    inits.extend([[0.9, 0.05], [0.05, 0.9], [0.05, 0.05]])

    for m0 in inits:
        m = m0[:]
        for _ in range(2500):
            rhs = self_consistency_map(m, p, beta)
            if math.hypot(rhs[0] - m[0], rhs[1] - m[1]) < 1e-10:
                m = rhs
                break
            m = [0.9 * m[0] + 0.1 * rhs[0], 0.9 * m[1] + 0.1 * rhs[1]]
        rhs = self_consistency_map(m, p, beta)
        if math.hypot(rhs[0] - m[0], rhs[1] - m[1]) < 1e-7:
            if not any(math.hypot(m[0] - u[0], m[1] - u[1]) < 1e-4 for u in roots):
                roots.append(m)

    if not roots:
        roots = [[1 / 3, 1 / 3]]

    out = []
    for m in roots:
        Jm = matvec2(p.J, m)
        B = [p.h[0] + p.z * Jm[0], p.h[1] + p.z * Jm[1]]
        energies = [-B[0], -B[1], 0.0]
        min_e = min(energies)
        weights = [math.exp(-beta * (e - min_e)) for e in energies]
        Z = sum(weights)
        probs = [w / Z for w in weights]
        S = -sum(pr * math.log(pr) for pr in probs if pr > 1e-12)
        F = free_energy_density(m, p, beta)
        U = F + S / beta
        out.append({"m": m, "S": S, "F": F, "U": U, "T": T})

    fmin = min(d["F"] for d in out)
    for d in out:
        d["status"] = "stable" if abs(d["F"] - fmin) < 1e-6 else "metastable"
    return out


def generate_svg(data, outfile):
    stable = [d for d in data if d["status"] == "stable"]
    metastable = [d for d in data if d["status"] == "metastable"]
    all_u = [d["U"] for d in data]
    all_s = [d["S"] for d in data]
    umin, umax = min(all_u), max(all_u)
    smin, smax = min(all_s), max(all_s)
    padx = (umax - umin) * 0.07 or 0.2
    pady = (smax - smin) * 0.07 or 0.2
    umin -= padx
    umax += padx
    smin -= pady
    smax += pady

    W, H = 920, 640
    left, right, top, bottom = 90, 40, 70, 90
    pw, ph = W - left - right, H - top - bottom

    def tx(u):
        return left + (u - umin) / (umax - umin) * pw

    def ty(s):
        return top + (smax - s) / (smax - smin) * ph

    lines = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">')
    lines.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')
    lines.append(f'<text x="{W/2}" y="36" text-anchor="middle" font-size="24">E-S Curve (3-state Potts Mean-Field)</text>')
    lines.append(f'<line x1="{left}" y1="{H-bottom}" x2="{W-right}" y2="{H-bottom}" stroke="black" stroke-width="2"/>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{H-bottom}" stroke="black" stroke-width="2"/>')

    for i in range(6):
        xv = umin + i * (umax - umin) / 5
        x = tx(xv)
        lines.append(f'<line x1="{x:.1f}" y1="{H-bottom}" x2="{x:.1f}" y2="{H-bottom+8}" stroke="black"/>')
        lines.append(f'<text x="{x:.1f}" y="{H-bottom+28}" text-anchor="middle" font-size="13">{xv:.2f}</text>')

        yv = smin + i * (smax - smin) / 5
        y = ty(yv)
        lines.append(f'<line x1="{left-8}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="black"/>')
        lines.append(f'<text x="{left-14}" y="{y+4:.1f}" text-anchor="end" font-size="13">{yv:.2f}</text>')

    lines.append(f'<text x="{W/2}" y="{H-30}" text-anchor="middle" font-size="18">Internal Energy U</text>')
    lines.append(f'<text x="30" y="{H/2}" transform="rotate(-90,30,{H/2})" text-anchor="middle" font-size="18">Entropy S</text>')

    for d in metastable:
        lines.append(f'<circle cx="{tx(d["U"]):.2f}" cy="{ty(d["S"]):.2f}" r="3" fill="#1f77b4" opacity="0.85"/>')
    for d in stable:
        lines.append(f'<circle cx="{tx(d["U"]):.2f}" cy="{ty(d["S"]):.2f}" r="3" fill="black" opacity="0.95"/>')

    lx, ly = W - 220, top + 15
    lines.append(f'<rect x="{lx}" y="{ly}" width="180" height="62" fill="white" stroke="#ddd"/>')
    lines.append(f'<circle cx="{lx+18}" cy="{ly+20}" r="4" fill="black"/><text x="{lx+34}" y="{ly+25}" font-size="14">Stable</text>')
    lines.append(f'<circle cx="{lx+18}" cy="{ly+45}" r="4" fill="#1f77b4"/><text x="{lx+34}" y="{ly+50}" font-size="14">Metastable</text>')

    lines.append('</svg>')
    with open(outfile, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    params = PottsParams(
        z=2.0,
        J=[[1.0, -1.0], [-1.0, 1.0]],
        h=[-0.95, -0.95],
    )
    T_vals = [0.05 + 0.03 * i for i in range(99)]

    data = []
    for T in T_vals:
        data.extend(solve_fixed_points_for_T(params, T))

    out = 'MeanField/code/es_curve_param_z2_h-0.95.svg'
    generate_svg(data, out)
    print(out)


if __name__ == '__main__':
    main()
