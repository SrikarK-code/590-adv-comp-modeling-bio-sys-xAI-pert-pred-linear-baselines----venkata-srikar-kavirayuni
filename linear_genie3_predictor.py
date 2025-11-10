# complete perturbation-predictor pipeline
# minimal comments, lowercase

import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from joblib import Parallel, delayed, dump, load
from sklearn.utils import resample
import itertools
import json
import time
import warnings
warnings.filterwarnings("ignore")

# ---------- user paths / params ----------
Y_PATH = "Y_pseudobulk.csv"          # genes x conditions, index=gene names, columns=condition ids
PERT_PATH = "perturbations.csv"      # columns: condition, perturbed_gene
SCGPT_EMB_PATH = "scgpt_gene_emb.npy"  # optional npy aligned to genes in Y
OUTDIR = "results"
SEED = 42

# model/hyperparams
K = 10       # gene embedding dim (pca for G)
L = 10       # perturbation pca dim (for linear P)
RIDGE_LAMBDA = 0.1
TOP_K_GENES = 1000
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 6
RF_MIN_SAMPLES_LEAF = 1
N_JOBS = -1

# experiment params
N_REPEATS = 5          # number of random train/test splits
TEST_FRAC = 0.2
BOOTSTRAP_ITERS = 1000
TOP_K_FOR_STABILITY = 20

os.makedirs(OUTDIR, exist_ok=True)

# ---------- helpers ----------
def load_data(y_path, pert_path):
    y = pd.read_csv(y_path, index_col=0)
    p = pd.read_csv(pert_path)
    p = p.set_index("condition").reindex(y.columns)
    return y, p

def gene_embedding_pca(Ytrain, k=K, random_state=SEED):
    X = (Ytrain - Ytrain.mean(axis=1).values.reshape(-1,1))
    pca = PCA(n_components=k, svd_solver="randomized", random_state=random_state)
    pca.fit(X.T)  # fit on conditions as samples for gene loadings
    G = pca.components_.T  # genes x k
    return G, pca

def perturbation_embedding_pca(Ytrain, l=L, random_state=SEED):
    pca = PCA(n_components=l, svd_solver="randomized", random_state=random_state)
    P = pca.fit_transform((Ytrain - Ytrain.mean(axis=0)).T)  # n_train x l
    return P, pca

def fit_linear_baseline(Y_train, G, P_train, ridge_lambda=RIDGE_LAMBDA):
    b = Y_train.mean(axis=1).values.reshape(-1,1)
    A = G.T @ (Y_train.values - b)        # K x n_train
    PtP = P_train.T @ P_train
    inv = np.linalg.inv(PtP + ridge_lambda * np.eye(PtP.shape[0]))
    W = A @ P_train @ inv                 # K x L
    return W, b

def predict_linear(G, W, b, P_test):
    Yhat_centered = G @ W @ P_test.T
    Yhat = Yhat_centered + b
    return Yhat

def build_perturbation_feature_matrix(pert_df, method, genes, scgpt_emb=None, pcaP=None, Ytrain=None):
    if method == "binary":
        rows = []
        for cond, row in pert_df.iterrows():
            vec = np.zeros(len(genes), dtype=float)
            tg = str(row['perturbed_gene'])
            if tg in genes:
                vec[genes.index(tg)] = 1.0
            rows.append(vec)
        X = np.vstack(rows)
        return X
    if method == "scgpt":
        if scgpt_emb is None:
            raise ValueError("scgpt emb required")
        rows = []
        for cond, row in pert_df.iterrows():
            tg = str(row['perturbed_gene'])
            if tg in genes:
                rows.append(scgpt_emb[genes.index(tg)])
            else:
                rows.append(np.zeros(scgpt_emb.shape[1]))
        return np.vstack(rows)
    if method == "pca":
        if pcaP is None or Ytrain is None:
            raise ValueError("pcaP and Ytrain required")
        # transform using pca fitted on Ytrain
        # note: pcaP expects samples as conditions; use mean-centered columns
        mat = (Ytrain - Ytrain.mean(axis=0))
        # get condition order same as pert_df.index
        order = list(pert_df.index)
        train_order = list(Ytrain.columns)
        # if pert_df is subset of train cols, transform directly else use mapping
        X = pcaP.transform((Ytrain[order] - Ytrain.mean(axis=0)).T)
        return X
    raise ValueError("unknown method")

def fit_rf_for_gene(y, X, rf_params):
    rf = RandomForestRegressor(n_estimators=rf_params['n_estimators'],
                               max_depth=rf_params['max_depth'],
                               min_samples_leaf=rf_params['min_samples_leaf'],
                               n_jobs=1,
                               random_state=rf_params['random_state'])
    rf.fit(X, y)
    return rf

def fit_genie3_rf_parallel(Y_train, X_train, rf_params, n_jobs=N_JOBS):
    genes = Y_train.index.tolist()
    X = X_train
    def fit_one(i):
        y = Y_train.values[i,:]
        return fit_rf_for_gene(y, X, rf_params)
    models = Parallel(n_jobs=n_jobs)(delayed(fit_one)(i) for i in range(len(genes)))
    return models

def predict_genie3_rf(models, X_test):
    n_genes = len(models)
    n_test = X_test.shape[0]
    Yhat = np.zeros((n_genes, n_test))
    for i, rf in enumerate(models):
        Yhat[i,:] = rf.predict(X_test)
    return Yhat

def rmse_topk(Y_true_df, Y_pred_arr, top_k=TOP_K_GENES):
    means = Y_true_df.mean(axis=1).values
    idx = np.argsort(means)[-min(top_k, len(means)):]
    return np.sqrt(np.mean((Y_true_df.values[idx,:] - Y_pred_arr[idx,:])**2))

def pearson_delta(Y_true_arr, Y_pred_arr):
    a = Y_true_arr.flatten()
    b = Y_pred_arr.flatten()
    if np.std(a)==0 or np.std(b)==0:
        return 0.0
    r, _ = pearsonr(a, b)
    return r

def bootstrap_ci(metric_vals, iters=BOOTSTRAP_ITERS, alpha=0.05, random_state=SEED):
    rng = np.random.RandomState(random_state)
    n = len(metric_vals)
    boots = []
    for _ in range(iters):
        sample = rng.choice(metric_vals, size=n, replace=True)
        boots.append(np.mean(sample))
    lo = np.percentile(boots, 100*alpha/2)
    hi = np.percentile(boots, 100*(1-alpha/2))
    return np.mean(metric_vals), lo, hi

def topk_features_from_rf(models, k=TOP_K_FOR_STABILITY, genes=None, feature_names=None):
    # extract top-k features per target gene from rf.feature_importances_
    out = {}
    for i, rf in enumerate(models):
        imp = rf.feature_importances_
        idx = np.argsort(imp)[-k:][::-1]
        names = [feature_names[j] for j in idx] if feature_names is not None else idx.tolist()
        out[genes[i]] = names
    return out

def jaccard_set(a, b):
    sa = set(a); sb = set(b)
    if len(sa|sb)==0: return 0.0
    return len(sa & sb) / len(sa | sb)

def stability_topk_across_splits(list_of_topk_dicts, k=TOP_K_FOR_STABILITY):
    # input: list of dicts {target: [top features]}
    genes = list(list_of_topk_dicts[0].keys())
    gene_scores = {}
    for g in genes:
        sets = [list_of_topk_dicts[s][g][:k] for s in range(len(list_of_topk_dicts))]
        pairs = list(itertools.combinations(sets, 2))
        if not pairs:
            gene_scores[g] = 0.0
            continue
        jaccs = [jaccard_set(a,b) for (a,b) in pairs]
        gene_scores[g] = np.mean(jaccs)
    return gene_scores  # per-gene mean jaccard

def calibration_slope_intercept(y_true_vec, y_pred_vec):
    lr = LinearRegression().fit(y_pred_vec.reshape(-1,1), y_true_vec.reshape(-1,1))
    slope = lr.coef_[0][0]
    intercept = lr.intercept_[0]
    return slope, intercept

# ---------- main experiment ----------
def run_experiment():
    Y, pert = load_data(Y_PATH, PERT_PATH)
    genes = Y.index.tolist()
    conds = list(Y.columns)
    rng = np.random.RandomState(SEED)

    # load scgpt emb if present
    use_scgpt = False
    if os.path.exists(SCGPT_EMB_PATH):
        try:
            scgpt_emb = np.load(SCGPT_EMB_PATH, allow_pickle=False)
            if scgpt_emb.shape[0] != len(genes):
                print("scgpt emb row mismatch; ignoring scgpt")
                use_scgpt = False
            else:
                use_scgpt = True
                print("scgpt embeddings loaded")
        except Exception as e:
            print("scgpt load failed:", e)
            use_scgpt = False

    rf_params = dict(n_estimators=RF_N_ESTIMATORS,
                     max_depth=RF_MAX_DEPTH,
                     min_samples_leaf=RF_MIN_SAMPLES_LEAF,
                     random_state=SEED)

    metrics_across_repeats = []
    linear_metrics = []
    rf_metrics = []
    topk_lists = []  # per-repeat topk dict
    null_metrics = []

    for rep in range(N_REPEATS):
        print("repeat", rep+1, "of", N_REPEATS)
        rng.shuffle(conds)
        n_train = int(len(conds) * (1 - TEST_FRAC))
        train_cols = conds[:n_train]
        test_cols = conds[n_train:]

        Ytrain = Y[train_cols]
        Ytest = Y[test_cols]
        pert_train = pert.loc[train_cols]
        pert_test = pert.loc[test_cols]

        # gene embedding pca (for linear baseline)
        G_pca, pcaG = gene_embedding_pca(Ytrain, k=K, random_state=SEED+rep)

        # perturbation embedding for linear baseline (pca on training conditions)
        P_train, pcaP = perturbation_embedding_pca(Ytrain, l=L, random_state=SEED+rep)
        P_test = pcaP.transform((Ytest - Ytest.mean(axis=0)).T)

        # fit linear baseline
        W, b = fit_linear_baseline(Ytrain, G_pca, P_train, ridge_lambda=RIDGE_LAMBDA)
        Yhat_linear = predict_linear(G_pca, W, b, P_test)
        rmse_l = rmse_topk(Ytest, Yhat_linear, top_k=min(TOP_K_GENES, len(genes)))
        pear_l = pearson_delta(Ytest.values, Yhat_linear)
        linear_metrics.append((rmse_l, pear_l))
        pd.DataFrame(Yhat_linear, index=genes, columns=test_cols).to_csv(os.path.join(OUTDIR, f"yhat_linear_rep{rep}.csv"))

        # build perturbation feature matrix for rf (prefer scgpt embedding if available, else binary)
        if use_scgpt:
            X_train = build_perturbation_feature_matrix(pert_train, method="scgpt", genes=genes, scgpt_emb=scgpt_emb)
            X_test = build_perturbation_feature_matrix(pert_test, method="scgpt", genes=genes, scgpt_emb=scgpt_emb)
            feature_names = [f"scgpt_dim{d}" for d in range(scgpt_emb.shape[1])]
        else:
            X_train = build_perturbation_feature_matrix(pert_train, method="binary", genes=genes)
            X_test = build_perturbation_feature_matrix(pert_test, method="binary", genes=genes)
            feature_names = genes

        # fit rf per gene in parallel
        t0 = time.time()
        models_rf = fit_genie3_rf_parallel(Ytrain, X_train, rf_params, n_jobs=N_JOBS)
        t1 = time.time()
        print("rf training time (s):", t1 - t0)

        # predict and eval
        Yhat_rf = predict_genie3_rf(models_rf, X_test)
        rmse_r = rmse_topk(Ytest, Yhat_rf, top_k=min(TOP_K_GENES, len(genes)))
        pear_r = pearson_delta(Ytest.values, Yhat_rf)
        rf_metrics.append((rmse_r, pear_r))
        pd.DataFrame(Yhat_rf, index=genes, columns=test_cols).to_csv(os.path.join(OUTDIR, f"yhat_rf_rep{rep}.csv"))

        # extract top-k features per gene from rf importances
        topk = topk_features_from_rf(models_rf, k=TOP_K_FOR_STABILITY, genes=genes, feature_names=feature_names)
        topk_lists.append(topk)
        # save per-repeat rf feature importances (mean abs)
        imps = np.vstack([m.feature_importances_ for m in models_rf])  # genes x features
        pd.DataFrame(imps, index=genes, columns=feature_names).to_csv(os.path.join(OUTDIR, f"rf_importances_rep{rep}.csv"))

        # calibration slopes (aggregate across genes)
        slopes = []
        intercepts = []
        for i,g in enumerate(genes):
            true = Ytest.values[i,:]
            pred = Yhat_rf[i,:]
            s, itc = calibration_slope_intercept(true, pred)
            slopes.append(s); intercepts.append(itc)
        # save calibration stats
        pd.DataFrame({"gene":genes, "slope":slopes, "intercept":intercepts}).to_csv(os.path.join(OUTDIR, f"rf_calibration_rep{rep}.csv"))

        # save rf models for this repeat
        dump(models_rf, os.path.join(OUTDIR, f"rf_models_rep{rep}.joblib"))
        # save linear params
        np.save(os.path.join(OUTDIR, f"linear_W_rep{rep}.npy"), W)
        np.save(os.path.join(OUTDIR, f"linear_b_rep{rep}.npy"), b)

        # permutation null test: shuffle pert labels and retrain a small rf (single iteration to get baseline)
        perm_pert = pert_train.copy().sample(frac=1.0, random_state=SEED+rep).reset_index(drop=False).set_index(pert_train.index)
        X_train_perm = build_perturbation_feature_matrix(perm_pert, method=("scgpt" if use_scgpt else "binary"), genes=genes, scgpt_emb=(scgpt_emb if use_scgpt else None))
        # fit a small rf for speed: train only 20 trees per gene
        small_rf_params = rf_params.copy(); small_rf_params['n_estimators'] = 20
        models_perm = fit_genie3_rf_parallel(Ytrain, X_train_perm, small_rf_params, n_jobs=N_JOBS)
        Yhat_perm = predict_genie3_rf(models_perm, X_test)
        rmse_perm = rmse_topk(Ytest, Yhat_perm, top_k=min(TOP_K_GENES, len(genes)))
        pear_perm = pearson_delta(Ytest.values, Yhat_perm)
        null_metrics.append((rmse_perm, pear_perm))
        print(f"rep {rep}: linear rmse {rmse_l:.4f}, rf rmse {rmse_r:.4f}, perm rmse {rmse_perm:.4f}")

    # aggregate metrics and compute bootstrap cis across repeats
    lin_rmse = [x[0] for x in linear_metrics]
    lin_pear = [x[1] for x in linear_metrics]
    rf_rmse = [x[0] for x in rf_metrics]
    rf_pear = [x[1] for x in rf_metrics]

    lin_mean_rmse, lin_lo, lin_hi = bootstrap_ci(np.array(lin_rmse))
    rf_mean_rmse, rf_lo, rf_hi = bootstrap_ci(np.array(rf_rmse))
    lin_mean_pear, _, _ = bootstrap_ci(np.array(lin_pear))
    rf_mean_pear, _, _ = bootstrap_ci(np.array(rf_pear))

    summary = {
        "linear_rmse_mean": float(lin_mean_rmse), "linear_rmse_ci": [float(lin_lo), float(lin_hi)],
        "rf_rmse_mean": float(rf_mean_rmse), "rf_rmse_ci": [float(rf_lo), float(rf_hi)],
        "linear_pearson_mean": float(lin_mean_pear), "rf_pearson_mean": float(rf_mean_pear),
        "repeats": N_REPEATS,
        "rf_params": rf_params,
        "rf_n_estimators": RF_N_ESTIMATORS,
        "rf_max_depth": RF_MAX_DEPTH
    }
    with open(os.path.join(OUTDIR, "summary_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # compute stability across repeats: per-gene mean jaccard
    stability_scores = stability_topk_across_splits(topk_lists, k=TOP_K_FOR_STABILITY)
    # save stability stats
    pd.DataFrame({"gene":list(stability_scores.keys()), "jaccard":list(stability_scores.values())}).to_csv(os.path.join(OUTDIR, "topk_stability.csv"), index=False)

    # aggregate network edges: for each target gene, take mean importance across repeats and keep top edges
    # load all rf_importances csv and average
    imp_files = [f for f in os.listdir(OUTDIR) if f.startswith("rf_importances_rep")]
    imps_agg = None
    for f in imp_files:
        df = pd.read_csv(os.path.join(OUTDIR, f), index_col=0)
        if imps_agg is None:
            imps_agg = df
        else:
            imps_agg = imps_agg + df
    if imps_agg is not None:
        imps_agg = imps_agg / len(imp_files)
        # build edge list: for each target gene, keep top 20 features with weight
        edges = []
        for target in imps_agg.index:
            row = imps_agg.loc[target]
            top_idx = np.argsort(row.values)[-TOP_K_FOR_STABILITY:][::-1]
            for idx in top_idx:
                src = row.index[idx]
                weight = float(row.values[idx])
                edges.append({"source": src, "target": target, "weight": weight})
        pd.DataFrame(edges).to_csv(os.path.join(OUTDIR, "aggregated_edge_list.csv"), index=False)

    # try shap explanations for rf if available (global mean abs)
    try:
        import shap
        # load one set of models to compute shap on a subset (can be slow) - use last repeat models_rf and X_test
        print("running shap treeexplainer on last repeat (may be slow)")
        all_imp = {}
        for i, rf in enumerate(models_rf):
            expl = shap.TreeExplainer(rf)
            vals = expl.shap_values(X_test)
            imp = np.mean(np.abs(vals), axis=0)
            all_imp[genes[i]] = imp
        df_shap = pd.DataFrame(all_imp).T
        df_shap.columns = feature_names
        df_shap.to_csv(os.path.join(OUTDIR, "rf_shap_importances_lastrep.csv"))
    except Exception as e:
        print("shap not available or failed:", e)

    print("done. summary saved to", OUTDIR)
    print(summary)

if __name__ == "__main__":
    run_experiment()
