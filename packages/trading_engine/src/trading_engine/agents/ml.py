"""Supervised ML agents: Random Forest, XGBoost and a PyTorch MLP.

All three predict the executable next-session direction label
(:mod:`trading_engine.training.labels`) from the genome's feature subset, and
share the :class:`~trading_engine.agents.genomes.PositionPolicy` that turns the
model score into a target weight. Model scores are *not* treated as calibrated
probabilities; they are only compared with the evolved threshold.

Preprocessing that learns statistics (the MLP's standard scaler) is fitted on the
training rows only, inside the agent, and stored with the artifact.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from trading_engine.agents.base import BaseTradingAgent, FitContext, Observations, TrainingSet
from trading_engine.agents.genomes import (
    MLPGenome,
    RandomForestGenome,
    XGBoostGenome,
    _LearnedGenome,
)


def _classification_metrics(y: np.ndarray, p: np.ndarray, prefix: str) -> dict[str, float | None]:
    out: dict[str, float | None] = {
        f"{prefix}_accuracy": float(np.mean((p > 0.5) == (y > 0.5))),
        f"{prefix}_base_rate": float(np.mean(y)),
    }
    try:
        out[f"{prefix}_auc"] = float(roc_auc_score(y, p))
    except ValueError:
        out[f"{prefix}_auc"] = None  # single-class labels
    out[f"{prefix}_logloss"] = float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]))
    return out


class LearnedAgent(BaseTradingAgent):
    is_learned = True
    genome: _LearnedGenome

    def _select(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.loc[:, list(self.genome.features)]

    def _training_rows(self, training: TrainingSet) -> tuple[pd.DataFrame, np.ndarray]:
        X = self._select(training.X)
        y = training.y.to_numpy(dtype="float64")
        if self.genome.train_window is not None and len(X) > self.genome.train_window:
            X, y = X.iloc[-self.genome.train_window :], y[-self.genome.train_window :]
        if len(X) < 100:
            raise ValueError(f"not enough training rows ({len(X)})")
        if np.isnan(X.to_numpy()).any() or np.isnan(y).any():
            raise ValueError("training data contains NaN; drop warm-up rows first")
        return X, y

    def fit(self, training: TrainingSet, context: FitContext) -> dict[str, Any]:
        X, y = self._training_rows(training)
        started = time.perf_counter()
        metrics = self._fit_model(X, y, context)
        p = self._score(X)
        metrics.update(_classification_metrics(y, p, "train"))
        metrics.update(
            {
                "n_samples": len(X),
                "n_features": X.shape[1],
                "train_start": X.index[0].date().isoformat(),
                "train_end": X.index[-1].date().isoformat(),
                "fit_seconds": round(time.perf_counter() - started, 3),
                "fitted": True,
            }
        )
        self.training_metrics = metrics
        self.fitted = True
        return metrics

    def forecast(self, observations: Observations) -> pd.Series:
        X = self._select(observations.features)
        valid = X.notna().all(axis=1).to_numpy()
        out = np.full(len(X), np.nan)
        if valid.any():
            out[valid] = self._score(X.loc[valid])
        return pd.Series(out, index=X.index)

    def position_policy(self, forecast: pd.Series) -> pd.Series:
        policy = self.genome.policy
        if policy.sizing == "binary":
            w = (forecast > policy.threshold).astype(float)
        else:  # proportional: linear ramp over 10 points above the threshold
            w = ((forecast - policy.threshold) / 0.10).clip(0.0, 1.0)
        return (w * policy.max_allocation).where(forecast.notna())

    # model-specific hooks
    def _fit_model(self, X: pd.DataFrame, y: np.ndarray, context: FitContext) -> dict[str, Any]:
        raise NotImplementedError

    def _score(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class RandomForestAgent(LearnedAgent):
    family = "random_forest"
    genome: RandomForestGenome

    def __init__(self, agent_id: str, genome: RandomForestGenome, seed: int) -> None:
        super().__init__(agent_id, genome, seed)
        self.model: Pipeline | None = None

    def _fit_model(self, X: pd.DataFrame, y: np.ndarray, context: FitContext) -> dict[str, Any]:
        g = self.genome
        self.model = Pipeline(
            [
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=g.n_estimators,
                        max_depth=g.max_depth,
                        min_samples_leaf=g.min_samples_leaf,
                        max_features=g.max_features,
                        random_state=self.seed,
                        n_jobs=1,
                    ),
                ),
            ]
        )
        self.model.fit(X.to_numpy(), y.astype(int))
        importances = self.model.named_steps["model"].feature_importances_
        return {
            "feature_importance": dict(
                zip(X.columns, np.round(importances, 5).tolist(), strict=True)
            )
        }

    def _score(self, X: pd.DataFrame) -> np.ndarray:
        assert self.model is not None
        proba = self.model.predict_proba(X.to_numpy())
        classes = list(self.model.classes_)
        return proba[:, classes.index(1)] if 1 in classes else np.zeros(len(X))

    def _save_model(self, directory: Path) -> None:
        joblib.dump(self.model, directory / "model.joblib")

    def _load_model(self, directory: Path) -> None:
        # Only called through ArtifactStore.open_verified (hash-checked, app-written files).
        self.model = joblib.load(directory / "model.joblib")


class XGBoostAgent(LearnedAgent):
    family = "xgboost"
    genome: XGBoostGenome

    def __init__(self, agent_id: str, genome: XGBoostGenome, seed: int) -> None:
        super().__init__(agent_id, genome, seed)
        self.model: XGBClassifier | None = None

    def _make(self) -> XGBClassifier:
        g = self.genome
        return XGBClassifier(
            n_estimators=g.n_estimators,
            max_depth=g.max_depth,
            learning_rate=g.learning_rate,
            subsample=g.subsample,
            colsample_bytree=g.colsample_bytree,
            reg_lambda=g.reg_lambda,
            min_child_weight=g.min_child_weight,
            tree_method="hist",
            n_jobs=1,
            random_state=self.seed,
            eval_metric="logloss",
        )

    def _fit_model(self, X: pd.DataFrame, y: np.ndarray, context: FitContext) -> dict[str, Any]:
        self.model = self._make()
        self.model.fit(X.to_numpy(), y.astype(int))
        gain = self.model.get_booster().get_score(importance_type="gain")
        named = {X.columns[int(k[1:])]: round(float(cast(float, v)), 5) for k, v in gain.items()}
        return {"feature_importance": named}

    def _score(self, X: pd.DataFrame) -> np.ndarray:
        assert self.model is not None
        return self.model.predict_proba(X.to_numpy())[:, 1]

    def _save_model(self, directory: Path) -> None:
        assert self.model is not None
        self.model.save_model(directory / "model.json")  # native JSON, no pickle

    def _load_model(self, directory: Path) -> None:
        self.model = self._make()
        self.model.load_model(directory / "model.json")


class MLPAgent(LearnedAgent):
    """Small feed-forward network trained with AdamW and validation early stopping.

    Early stopping uses the *last* 20% of the training rows (chronologically, after a
    purge gap of ``horizon + 1`` rows), never the walk-forward validation fold.
    """

    family = "mlp"
    genome: MLPGenome

    def __init__(self, agent_id: str, genome: MLPGenome, seed: int) -> None:
        super().__init__(agent_id, genome, seed)
        self.net: Any = None
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def _build(self, n_in: int) -> Any:
        import torch  # noqa: PLC0415 - heavy import kept local
        from torch import nn  # noqa: PLC0415

        torch.manual_seed(self.seed)  # before any parameter is initialised
        layers: list[nn.Module] = []
        width = n_in
        for hidden in self.genome.hidden_layers:
            layers += [nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(self.genome.dropout)]
            width = hidden
        layers.append(nn.Linear(width, 1))
        return nn.Sequential(*layers)

    def _standardise(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.scale_ is not None
        return (X - self.mean_) / self.scale_

    def _fit_model(self, X: pd.DataFrame, y: np.ndarray, context: FitContext) -> dict[str, Any]:
        import torch  # noqa: PLC0415
        from torch import nn  # noqa: PLC0415

        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        g = self.genome
        n = len(X)
        gap = context.horizon + 1
        n_val = max(int(n * 0.2), 50)
        fit_end = n - n_val - gap
        if fit_end < 50:
            raise ValueError("training window too short for MLP early stopping")
        Xa = X.to_numpy(dtype="float64")
        # Scaler statistics from the fitting rows only.
        self.mean_ = Xa[:fit_end].mean(axis=0)
        self.scale_ = Xa[:fit_end].std(axis=0)
        self.scale_[self.scale_ < 1e-12] = 1.0
        Xs = self._standardise(Xa).astype("float32")
        Xt, yt = torch.from_numpy(Xs[:fit_end]), torch.from_numpy(y[:fit_end].astype("float32"))
        Xv, yv = (
            torch.from_numpy(Xs[n - n_val :]),
            torch.from_numpy(y[n - n_val :].astype("float32")),
        )

        self.net = self._build(Xs.shape[1])
        opt = torch.optim.AdamW(
            self.net.parameters(), lr=g.learning_rate, weight_decay=g.weight_decay
        )
        loss_fn = nn.BCEWithLogitsLoss()
        gen = torch.Generator().manual_seed(self.seed)
        best_loss, best_state, best_epoch, stale = math.inf, None, 0, 0
        history = []
        started = time.perf_counter()
        for epoch in range(g.max_epochs):
            self.net.train()
            perm = torch.randperm(len(Xt), generator=gen)
            for start in range(0, len(Xt), g.batch_size):
                batch = perm[start : start + g.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.net(Xt[batch]).squeeze(-1), yt[batch])
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(self.net(Xv).squeeze(-1), yv))
            history.append(round(val_loss, 6))
            if val_loss < best_loss - 1e-5:
                best_loss, best_epoch, stale = val_loss, epoch, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                stale += 1
                if stale >= g.patience:
                    break
            if context.max_seconds and time.perf_counter() - started > context.max_seconds:
                break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()
        return {
            "best_epoch": best_epoch,
            "epochs_run": len(history),
            "early_stop_val_loss": best_loss,
            "val_loss_history": history,
        }

    def _score(self, X: pd.DataFrame) -> np.ndarray:
        import torch  # noqa: PLC0415

        assert self.net is not None
        xs = torch.from_numpy(self._standardise(X.to_numpy(dtype="float64")).astype("float32"))
        with torch.no_grad():
            return torch.sigmoid(self.net(xs).squeeze(-1)).numpy().astype("float64")

    def _save_model(self, directory: Path) -> None:
        import torch  # noqa: PLC0415

        assert self.mean_ is not None and self.scale_ is not None
        torch.save(self.net.state_dict(), directory / "model.pt")
        (directory / "scaler.json").write_text(
            json.dumps(
                {
                    "mean": self.mean_.tolist(),
                    "scale": self.scale_.tolist(),
                    "features": list(self.genome.features),
                }
            )
        )

    def _load_model(self, directory: Path) -> None:
        import torch  # noqa: PLC0415

        scaler = json.loads((directory / "scaler.json").read_text())
        self.mean_ = np.asarray(scaler["mean"])
        self.scale_ = np.asarray(scaler["scale"])
        self.net = self._build(len(self.mean_))
        self.net.load_state_dict(torch.load(directory / "model.pt", weights_only=True))
        self.net.eval()
