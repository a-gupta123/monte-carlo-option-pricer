import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def mc_call(S0, K, r, vol, T, N):
    # Monte Carlo with no variance reduction
    Z = np.random.normal(size=N)
    ST = S0 * np.exp((r - 0.5*vol**2)*T + vol*np.sqrt(T)*Z)
    payoffs = np.maximum(ST - K, 0)
    discounted_payoffs = np.exp(-r*T) * payoffs
    
    price = discounted_payoffs.mean()
    se = discounted_payoffs.std(ddof=1) / np.sqrt(N)
    ci_low = price - 1.96*se
    ci_high = price + 1.96*se

    return price, ci_low, ci_high, se

def mc_call_with_antithetic_variates(S0, K, r, vol, T, N):
    # Monte Carlo with antithetic variates

    Z = np.random.normal(size=N)
    ST_pos = S0 * np.exp((r - 0.5*vol**2)*T + vol*np.sqrt(T)*Z)
    ST_neg =  S0 * np.exp((r - 0.5*vol**2)*T - vol*np.sqrt(T)*Z)

    pos_payoffs = np.maximum(ST_pos - K, 0)
    neg_payoffs = np.maximum(ST_neg - K, 0)
    discounted_payoffs = np.exp(-r*T) * (pos_payoffs + neg_payoffs)/2
    
    price = discounted_payoffs.mean()
    se = discounted_payoffs.std(ddof=1) / np.sqrt(N)

    ci_low = price -1.96 *se
    ci_high = price + 1.96*se

    return price, ci_low, ci_high, se

def black_scholes_call(S0, K, r, vol, T):
    # Black-Scholes option pricing

    if T <= 0:
        return max(S0 - K, 0.0)
    if vol <= 0:
        ST = S0 * np.exp(r*T)
        return np.exp(-r*T) * max(ST - K, 0.0)

    d1 = (np.log(S0/K) + (r + 0.5*vol**2)*T) / (vol*np.sqrt(T))
    d2 = d1 - vol*np.sqrt(T)

    return S0 * norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def convergence_plot(S0, K, r, vol, T, N):
    bs = black_scholes_call(S0, K, r, vol, T)

    original_prices = []
    antithetic_prices = []
    antithetic_ci_lows = []
    antithetic_ci_highs = []
    for n in N:
        p_original, _, _, _ = mc_call(S0, K, r, vol, T, n)
        p_anti, lo_anti, hi_anti, _ = mc_call_with_antithetic_variates(S0, K, r, vol, T, n)
        original_prices.append(p_original)
        antithetic_prices.append(p_anti)
        antithetic_ci_lows.append(lo_anti)
        antithetic_ci_highs.append(hi_anti)

    plt.figure()
    plt.plot(N_values_for_convergence_plot, antithetic_prices, marker="o", label="MC (antithetic variates)")
    plt.fill_between(N_values_for_convergence_plot, antithetic_ci_lows, antithetic_ci_highs, alpha=0.25, label="95% CI (antithetic)")
    plt.plot(N_values_for_convergence_plot, original_prices, marker="o", label="MC (no variance reduction)")
    plt.axhline(bs, linestyle="--", label="Black–Scholes price")

    plt.xscale("log")
    plt.xlabel("Number of simulations (N)")
    plt.ylabel("Option price")
    plt.title("Convergence Plot")
    plt.legend()
    plt.tight_layout()
    plt.savefig("convergence_plot.png", dpi=200, bbox_inches="tight")
    plt.show()

# Parameters
S0 = 100   # stock price
K = 100    # strike price
r = 0.05   # risk-free rate (%)
vol = 0.2  # volatility (%)
T = 1      # time to maturity in years

# Number of Monte Carlo simulations used for price estimation in terminal output
N = 10000

original_price, original_lo, original_hi, original_se = mc_call(S0, K, r, vol, T, N)
anti_price, anti_lo, anti_hi, anti_se = mc_call_with_antithetic_variates(S0, K, r, vol, T, N)
bs = black_scholes_call(S0, K, r, vol, T)

print("Original MC:", original_price, "CI:", (float(original_lo), float(original_hi)), "SE:", original_se)
print("Antithetic MC:", anti_price, "CI:", (float(anti_lo), float(anti_hi)), "SE:", anti_se)
print("Black–Scholes:", bs)

# Convergence plot
N_values_for_convergence_plot = [200, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000]
convergence_plot(S0, K, r, vol, T, N_values_for_convergence_plot)
