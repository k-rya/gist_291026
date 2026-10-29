import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/tmp/claude-0/-home-claude/9abe3760-06c7-5b7c-acc5-e31dd6412d9c/scratchpad"
rng = np.random.default_rng(7)
DELTA = 1.0
dt = 0.05

def lorentz_freqs(N, d=DELTA, clip=40.0):
    q = (np.arange(N) + 0.5) / N
    return np.clip(d*np.tan(np.pi*(q-0.5)), -clip*d, clip*d)

def order_global(phi):      return abs(np.mean(np.exp(1j*phi)))
def order_block(phi, m):    return abs(np.mean(np.exp(1j*phi[m])))

def part1():
    N = 1000
    isA = np.zeros(N, bool); isA[:N//2] = True; isB = ~isA
    a, b = 0.8, 0.2
    lam_max = 0.5*(a+b)
    Kc = 2*DELTA/lam_max
    OMEGA = rng.permutation(lorentz_freqs(N))
    print(f"[1] SBM a={a} b={b}:  λ_max(W)={lam_max:.3f}  K_c=2Δ/λ_max={Kc:.3f}")

    def rhs(phi, K1, K2):
        z1 = np.exp(1j*phi); z2 = z1*z1
        sA1, sB1 = z1[isA].sum(), z1[isB].sum()
        sA2, sB2 = z2[isA].sum(), z2[isB].sum()
        Wz1 = np.where(isA, a*sA1+b*sB1, b*sA1+a*sB1)
        Wz2 = np.where(isA, a*sA2+b*sB2, b*sA2+a*sB2)
        pair = (K1/N)   * np.imag(np.conj(z1)*Wz1)
        hi   = (K2/N**2)* np.imag(np.conj(z1)*(Wz2*np.conj(Wz1)))
        return OMEGA + pair + hi

    def step(phi, K1, K2):
        k1 = rhs(phi,K1,K2); k2 = rhs(phi+0.5*dt*k1,K1,K2)
        k3 = rhs(phi+0.5*dt*k2,K1,K2); k4 = rhs(phi+dt*k3,K1,K2)
        return phi + (dt/6)*(k1+2*k2+2*k3+k4)

    def settle(phi, K1, K2, Ts=14, Ta=5):
        for _ in range(int(Ts/dt)): phi = step(phi,K1,K2)
        acc=0.0
        for _ in range(int(Ta/dt)): phi = step(phi,K1,K2); acc += order_global(phi)
        return acc/int(Ta/dt), phi

    def sweep(K1seq, K2, phi):
        R=[]
        for K1 in K1seq:
            r, phi = settle(phi, K1, K2); R.append(r)
        return np.array(R), phi

    K1s = np.linspace(1.0, 9.0, 16)
    K2grid = np.linspace(0, 26, 8)
    want = {0.0, K2grid[3], K2grid[-1]}
    curves = {}; widths = []
    for K2 in K2grid:
        phi0 = rng.uniform(0, 2*np.pi, N)
        Ru, phitop = sweep(K1s, K2, phi0)
        Rd, _      = sweep(K1s[::-1], K2, phitop); Rd = Rd[::-1]
        w = np.trapezoid(np.abs(Rd-Ru), K1s); widths.append(w)
        if K2 in want: curves[K2] = dict(Ru=Ru, Rd=Rd, width=w)
        print(f"    K2={K2:4.1f}:  ширина петли гистерезиса = {w:.3f}")
    widths = np.array(widths)
    K2_list = sorted(curves)

    fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.6))
    pcols = ["0.35", "tab:blue", "crimson"]
    for K2, c in zip(K2_list, pcols):
        cc = curves[K2]
        ax[0].plot(K1s, cc["Ru"], "-o", ms=3, lw=1.9, color=c, label=fr"$K_2={K2:.0f}$ ↑")
        ax[0].plot(K1s, cc["Rd"], "--", lw=1.3, color=c)
    ax[0].axvline(Kc, color="green", ls=":", lw=1.8, label=fr"$K_c={Kc:.1f}$ (спектр $W$)")
    ax[0].set_xlabel(r"парная связь $K_1$"); ax[0].set_ylabel(r"параметр порядка $R$")
    ax[0].set_title(r"Тройки открывают взрывную петлю, привязанную к $K_c(W)$")
    ax[0].legend(fontsize=8); ax[0].set_ylim(-0.02, 1.02)

    ax[1].plot(K2grid, widths, "o-", color="seagreen", ms=6, lw=1.8)
    ax[1].set_xlabel(r"сила троек $K_2$"); ax[1].set_ylabel(r"ширина петли $\int|R_\downarrow-R_\uparrow|\,dK_1$")
    ax[1].set_title(r"Взрывность растёт с $K_2$ (флаг-комплекс $T_{ijk}=W_{ij}W_{ik}$)")
    ax[1].set_ylim(bottom=-0.01)
    fig.suptitle(fr"Графон + тройки, конечное $N={N}$: порог из спектра $W$, тройки задают взрывность",
                 fontsize=12)
    plt.tight_layout(rect=[0,0,1,0.95])
    f1 = f"{OUT}/triples_fig1_anchor.png"; plt.savefig(f1, dpi=150); plt.close()
    print("    ->", f1)
    return dict(Kc=Kc, N=N, widths=widths, K2grid=K2grid, curves=curves, K1s=K1s, K2_list=K2_list)

def part2():
    N = 1000
    isA = np.zeros(N, bool); isA[:N//2] = True
    fvec = np.where(isA, 0.4, 1.0)
    lam_max = np.mean(fvec**2)
    Kc = 2*DELTA/lam_max
    OMEGA = rng.permutation(lorentz_freqs(N))
    print(f"[2] rank-1 W=f⊗f, f∈{{0.4,1.0}}:  λ_max=⟨f²⟩={lam_max:.3f}  K_c={Kc:.3f}")

    def make_rhs(tvec):
        def rhs(phi, K1, K2):
            z1 = np.exp(1j*phi); z2 = z1*z1
            Sf = (fvec*z1).sum()
            Wz1 = fvec*Sf
            pair = (K1/N)*np.imag(np.conj(z1)*Wz1)
            S1 = (tvec*z1).sum(); S2 = (tvec*z2).sum()
            hi = K2*tvec*np.imag(np.conj(z1)*(S2/N)*np.conj(S1/N))
            return OMEGA + pair + hi
        return rhs

    def integrate(rhs, phi, K1, K2, T):
        for _ in range(int(T/dt)):
            k1 = rhs(phi,K1,K2); k2 = rhs(phi+0.5*dt*k1,K1,K2)
            k3 = rhs(phi+0.5*dt*k2,K1,K2); k4 = rhs(phi+dt*k3,K1,K2)
            phi = phi + (dt/6)*(k1+2*k2+2*k3+k4)
        return phi

    def settle(rhs, phi, K1, K2, Ts=14, Ta=5):
        phi = integrate(rhs, phi, K1, K2, Ts)
        acc=0.0; nav=int(Ta/dt)
        for _ in range(nav):
            phi = integrate(rhs, phi, K1, K2, dt); acc += order_global(phi)
        return acc/nav, phi

    def loop_width(rhs, K1s, K2):
        phi = rng.uniform(0, 2*np.pi, N)
        Ru=[]
        for K1 in K1s: r,phi = settle(rhs,phi,K1,K2); Ru.append(r)
        Rd=[]
        for K1 in K1s[::-1]: r,phi = settle(rhs,phi,K1,K2); Rd.append(r)
        Ru=np.array(Ru); Rd=np.array(Rd[::-1])
        return np.trapezoid(np.abs(Rd-Ru), K1s), Ru, Rd

    tensors = {
        "согласованный  t=f":      make_rhs(fvec.copy()),
        "рассогласованный t=flip(f)": make_rhs(np.where(isA, 1.0, 0.4)),
    }
    cols = {"согласованный  t=f":"crimson", "рассогласованный t=flip(f)":"navy"}

    def cross(K1s, R, thr=0.5):
        for i in range(1, len(R)):
            if R[i-1] < thr <= R[i]:
                u = (thr-R[i-1])/(R[i]-R[i-1]); return K1s[i-1]+u*(K1s[i]-K1s[i-1])
        return np.nan

    K1s = np.linspace(Kc-2.0, Kc+3.4, 14)
    K2grid = np.linspace(0, 40, 8)
    K2show = 26.0
    Kup = {}; demo = {}
    for name, rhs in tensors.items():
        kups = []; best = None; bestd = 1e9
        for K2 in K2grid:
            w, Ru, Rd = loop_width(rhs, K1s, K2)
            kups.append(cross(K1s, Ru))
            if abs(K2-K2show) < bestd: bestd = abs(K2-K2show); best = (Ru, Rd, K2)
        Kup[name] = np.array(kups); demo[name] = best
        jump_below = np.nanmin(Kup[name])
        print(f"    {name:26s}: мин. положение скачка K1↑={jump_below:.2f}  (K_c={Kc:.2f})")
    K2show = demo[list(tensors)[0]][2]

    fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.6))
    for name in tensors:
        Ru, Rd, _ = demo[name]
        ax[0].plot(K1s, Ru, "-o", ms=3, lw=1.9, color=cols[name], label=f"{name} ↑")
        ax[0].plot(K1s, Rd, "--", lw=1.3, color=cols[name])
    ax[0].axvline(Kc, color="green", ls=":", lw=1.8, label=fr"$K_c={Kc:.1f}$")
    ax[0].set_xlabel(r"парная связь $K_1$"); ax[0].set_ylabel(r"параметр порядка $R$")
    ax[0].set_title(fr"$R(K_1)$ при $K_2={K2show:.0f}$: согласованный тензор уже взрывной")
    ax[0].legend(fontsize=8); ax[0].set_ylim(-0.02, 1.02)

    for name in tensors:
        ax[1].plot(K2grid, Kup[name], "o-", ms=6, lw=1.8, color=cols[name], label=name)
    ax[1].axhline(Kc, color="green", ls=":", lw=1.8, label=fr"$K_c={Kc:.1f}$")
    ax[1].set_xlabel(r"сила троек $K_2$"); ax[1].set_ylabel(r"положение скачка $K_1^{\uparrow}$ (где $R{=}0.5$)")
    ax[1].set_title(r"Согласованный $T$ уводит скачок ПОД $K_c$ (взрывная нуклеация)")
    ax[1].legend(fontsize=8)
    fig.suptitle(r"Конкуренция «спектра $W$» и структуры тензора $T$: взрывность определяет "
                 r"выравнивание $T$ с ведущей модой $v_1$", fontsize=12)
    plt.tight_layout(rect=[0,0,1,0.95])
    f2 = f"{OUT}/triples_fig2_competition.png"; plt.savefig(f2, dpi=150); plt.close()
    print("    ->", f2)
    return dict(Kc=Kc, N=N, K2grid=K2grid, Kup=Kup, K2show=K2show)

if __name__ == "__main__":
    print("="*72)
    r1 = part1()
    print("-"*72)
    r2 = part2()
    print("="*72)
    print("Готово. Рисунки: triples_fig1_anchor.png, triples_fig2_competition.png")
