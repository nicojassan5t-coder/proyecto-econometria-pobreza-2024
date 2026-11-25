from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

base = pd.read_csv(Path(__file__).with_name("base_2024.csv"))
assert sorted(base.cve_ent) == list(range(1, 33))
assert not base.isna().any().any()
base["pibpc"] = base.pib_mmdp_2018 * 1_000_000 / base.habitantes
base["ln_pibpc"] = np.log(base.pibpc)

for variable in ["pibpc", "ln_pibpc"]:
    cols = [variable, "escolaridad", "informalidad"]
    y = base.pobreza.to_numpy()
    X = np.column_stack([np.ones(len(base)), base[cols]])
    b, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    assert rank == X.shape[1]
    n, k = X.shape
    A = np.linalg.pinv(X)
    h = np.sum(X * A.T, axis=1)
    fitted = X @ b
    e = y - fitted
    se = np.sqrt(np.sum((A * (e / (1 - h)))**2, axis=1))
    df = n - k
    p = 2 * stats.t.sf(abs(b / se), df)
    margin = stats.t.ppf(.975, df) * se
    print(variable, pd.DataFrame(dict(b=b, EE=se, p=p, inferior=b-margin, superior=b+margin),
                                index=["Constante"] + cols))
    r2 = 1 - (e @ e) / sum((y - y.mean())**2)
    print("R2 y R2 ajustada", r2, 1-(1-r2)*(n-1)/df)
    corr = base[cols].corr()
    print(corr, "VIF", np.diag(np.linalg.inv(corr)))
    sq = e**2
    aux = X @ np.linalg.lstsq(X, sq, rcond=None)[0]
    lm = n * (1 - sum((sq - aux)**2) / sum((sq - sq.mean())**2))
    print("Koenker-Breusch-Pagan", lm, stats.chi2.sf(lm, k-1))
    print("Shapiro-Wilk", stats.shapiro(e))
    cook = e**2 / (k * (e @ e / df)) * h / (1-h)**2
    print(pd.DataFrame(dict(entidad=base.entidad, ajustado=fitted, residuo=e,
                            leverage=h, cook=cook)).nlargest(5, "cook"))
    keep = base.cve_ent != 4
    print("Sin Campeche", np.linalg.lstsq(X[keep], y[keep], rcond=None)[0])
print(base[["pobreza","pibpc","escolaridad","informalidad"]].describe())