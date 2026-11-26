"""Figures for paper and poster. Export PDF vectors and 220-dpi PNGs from the same values."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from data import DATA_DIR, TRAIN_LEN, VAL_LEN, TEST_LEN
import pandas as pd

COLORS = {"transformer":"#124979", "linear":"#df6c21", "mlp":"#397c63", "lstm":"#8357a2", "naive":"#666666"}
LABELS = {"transformer":"Patch Transformer", "linear":"Direct linear", "mlp":"MLP", "lstm":"LSTM", "naive":"Repeat last day"}
MARKERS = {"transformer":"o", "linear":"s", "mlp":"^", "lstm":"D", "naive":"x"}


def save(fig, folder, name):
    for ext in ("pdf", "png"):
        fig.savefig(folder / f"{name}.{ext}", bbox_inches="tight", dpi=220)
    plt.close(fig)


def make_figures(folder, agg, history, sim, forecast, contexts):
    folder.mkdir()
    plt.rcParams.update({"font.size":10, "pdf.fonttype":42, "axes.spines.right":False, "axes.spines.top":False})
    core = agg[agg.variant.eq("main")]
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.3), layout="constrained")
    for row, site in enumerate(("ETTh1", "ETTh2")):
        for col, h in enumerate((24, 96)):
            ax = axes[row, col]
            df = core[(core.dataset==site) & (core.horizon==h)]
            for model, label in LABELS.items():
                d = df[df.model==model].sort_values("percent")
                ax.errorbar(d.percent, d.mae_mean, yerr=d.mae_seed_sd.fillna(0), marker=MARKERS[model],
                            color=COLORS[model], linestyle="--" if model=="naive" else "-", capsize=3, label=label)
            ax.set(title=f"{site}, {h}-hour forecast", xticks=[25,50,100], xlabel="Training history (%)", ylabel="Test MAE (°C)")
            ax.grid(axis="y", alpha=.18)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False)
    save(fig, folder, "history_learning_curves")

    fig, axs = plt.subplots(1, 2, figsize=(9.1,3.5), layout="constrained")
    for ax, site in zip(axs, ("ETTh1","ETTh2")):
        d = core[(core.dataset==site) & (core.horizon==24) & (core.percent==100) & (core.model!="naive")].sort_values("fit_seconds_mean")
        ax.barh([LABELS[m] for m in d.model], d.fit_seconds_mean, color=[COLORS[m] for m in d.model])
        ax.set(title=f"{site}, 24 h, full history", xlabel="Mean time per selected fit (s)")
    save(fig, folder, "training_cost")

    fig, axs = plt.subplots(1, 2, figsize=(9.5,3.6), layout="constrained")
    for ax, site in zip(axs, ("ETTh1","ETTh2")):
        group = forecast[(forecast.dataset==site)&(forecast.horizon==24)&(forecast.percent==100)&
                         (forecast.variant=="main")&(forecast.seed==0)&(forecast.origin_index==0)]
        e = contexts[f"{site}_H24"][0]
        ax.plot(np.arange(-47,1), e["history"][-48:], color="#252525", label="Observed history")
        ax.plot(np.arange(1,25), e["actual"], color="#080808", lw=2.3, label="Observed future")
        for m in ("transformer","linear","naive"):
            g = group[group.model==m].sort_values("lead")
            ax.plot(g.lead, g.forecast, color=COLORS[m], linestyle="--" if m=="naive" else "-", label=LABELS[m])
        ax.axvline(.5, color="#777777", linestyle=":", lw=1)
        ax.set(title=f"{site}, first test origin", xlabel="Hours from last observation", ylabel="Oil temperature (°C)")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    save(fig, folder, "observed_forecasts")

    fig, axs = plt.subplots(1,2,figsize=(8.6,3.6),layout="constrained")
    for ax, (m, other, alt) in zip(axs,[("transformer","depth1","One layer"),("transformer","noIN","Without window normalization")]):
        sub = agg[(agg.dataset=="ETTh1")&(agg.horizon==24)&(agg.model==m)&agg.percent.isin([25,100])]
        for j, (variant, label) in enumerate([("main","Two layers, normalized"),(other,alt)]):
            d=sub[sub.variant==variant].sort_values("percent")
            ax.bar(np.arange(2)+(j-.5)*.34,d.mae_mean,.34,yerr=d.mae_seed_sd,capsize=3,label=label,
                   color=["#124979","#d07b36"][j])
        ax.set(xticks=[0,1],xticklabels=["25%","100%"],xlabel="Training history",ylabel="Test MAE (°C)")
        ax.legend(fontsize=8,frameon=False)
    save(fig,folder,"transformer_ablations")

    fig, ax = plt.subplots(figsize=(7.5,3.4),layout="constrained")
    for j,(v, label) in enumerate([("main","Normalized Transformer"),("noIN","Unnormalized Transformer")]):
        d=sim[(sim.model=="transformer")&(sim.variant==v)].groupby("scenario").mae.mean()
        order=["baseline","shared_shift","future_only_shift"]
        ax.bar(np.arange(3)+(j-.5)*.32,d.reindex(order),.32,label=label,color=["#124979","#d07b36"][j])
    ax.set(xticks=[0,1,2],xticklabels=["Baseline","Past and future shift","Future shifts alone"],ylabel="Simulated MAE (°C)")
    ax.legend(frameon=False)
    save(fig,folder,"paired_shift_simulation")

    fig, axs = plt.subplots(1,2,figsize=(8.8,3.5),layout="constrained")
    for ax,p in zip(axs,(25,100)):
        d=history[(history.dataset=="ETTh1")&(history.horizon==24)&(history.percent==p)&
                  (history.model=="transformer")&(history.variant=="main")&(history.seed==0)]
        ax.plot(d.epoch,d.train_mse,label="Training",color="#124979")
        ax.plot(d.epoch,d.val_mse,label="Validation",color="#d07b36")
        ax.set(title=f"ETTh1, {p}% history",xlabel="Epoch",ylabel="MSE (training-standardized scale)")
        ax.legend(frameon=False)
    save(fig,folder,"optimization_curves")

    fig, axs = plt.subplots(2,1,figsize=(8.5,4.3),sharex=True,layout="constrained")
    context_end = TRAIN_LEN+VAL_LEN+TEST_LEN
    for ax,site in zip(axs,("ETTh1","ETTh2")):
        raw=pd.read_csv(DATA_DIR / f"{site}.csv",usecols=["OT"]).OT.to_numpy()[:context_end]
        days=np.arange(len(raw)//24)
        ax.plot(days,raw.reshape(-1,24).mean(1),color="#124979",lw=1.15)
        ax.axvspan(TRAIN_LEN/24,(TRAIN_LEN+VAL_LEN)/24,color="#f5e7cc",zorder=-4)
        ax.axvspan((TRAIN_LEN+VAL_LEN)/24,len(days),color="#e3efea",zorder=-4)
        ax.set(ylabel=f"{site}\nDaily mean (°C)")
    axs[0].set(title="Training (days 0–359), validation (360–479), test (480–599)")
    axs[1].set(xlabel="Days since 1 July 2016")
    save(fig,folder,"temperature_history")
