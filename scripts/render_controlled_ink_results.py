"""Render the frozen experiment, native CT evidence and explicitly exploratory P800 maps."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from scripts.controlled_ink_benchmark import write_json

NAMES={"residual":"U-Net residual", "residual_physical":"Residual + física", "kintsugi_physical":"Kintsugi + física"}


def bar(ax, shape, spacing, label="5 mm", length=5000):
    height,width=shape
    x,y=width*0.06,height*0.93
    ax.plot([x,x+length/spacing],[y,y],color="cyan",lw=3)
    ax.text(x,y-height*0.025,label,color="cyan",fontsize=8,weight="bold")


def main():
    repo=Path(__file__).resolve().parents[1]
    out=repo/"runs/controlled_ink_20261025"
    data=repo/"data/controlled_ink_20261025"
    report=json.loads((out/"metrics.json").read_text())
    protocol=report["protocol"]
    labels=np.load(out/"test_labels.npz")
    ink,mask,surface=labels["ink"],labels["supervision"],labels["surface"]
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"figure.facecolor":"white","axes.titlesize":10})
    figures=[]
    fig,axes=plt.subplots(2,4,figsize=(18,9),constrained_layout=True)
    truth=np.zeros((*ink.shape,3),np.float32);truth[mask]=[.25,.25,.25];truth[mask&ink]=[.1,.9,.4]
    axes[0,0].imshow(truth);axes[0,0].set_title("Etiquetas de w039\nVerde: tinta revisada; gris: fondo revisado")
    axes[1,0].imshow(surface,cmap="gray");axes[1,0].set_title("CT original, capa central\nZona de prueba ajena al entrenamiento")
    bar(axes[1,0],ink.shape,9.362)
    rows=[]
    for column,name in enumerate(protocol["models"],1):
        runs=[r for r in report["runs"] if r["model"]==name]
        first=next(r for r in runs if r["seed"]==1337)
        probability=np.load(out/first["prediction"])
        axes[0,column].imshow(np.ma.masked_where(~mask,probability),cmap="inferno",vmin=0,vmax=1)
        m=first["test_at_validation_f1_threshold"]
        axes[0,column].set_title(f"{NAMES[name]} · semilla 1337\nAUC {m['roc_auc']:.3f} | AP {m['ap']:.3f}")
        strict=first["test_at_validation_1pct_fpr"]
        predicted=probability>=strict["threshold"]
        rgb=np.zeros((*ink.shape,3),np.float32)
        rgb[mask&predicted&ink]=[.1,.8,.3]
        rgb[mask&predicted&~ink]=[.95,.15,.1]
        rgb[mask&~predicted&ink]=[.1,.4,.8]
        axes[1,column].imshow(rgb)
        axes[1,column].set_title(f"Umbral fijado con 1 % FP en validación\nPrecisión {strict['precision']:.2f} | Recall {strict['recall']:.2f}")
        aps=[r["test_at_validation_f1_threshold"]["ap"] for r in runs]
        aucs=[r["test_at_validation_f1_threshold"]["roc_auc"] for r in runs]
        f1s=[r["test_at_validation_f1_threshold"]["f1"] for r in runs]
        fps=[r["test_at_validation_f1_threshold"]["fp_per_mpix"] for r in runs]
        rows.append({"model":name,"parameters":first["parameters"],"test_ap_mean":float(np.mean(aps)),
                     "test_ap_min":min(aps),"test_ap_max":max(aps),"test_auc_mean":float(np.mean(aucs)),
                     "test_f1_at_frozen_threshold_mean":float(np.mean(f1s)),"test_fp_per_mpix_mean":float(np.mean(fps))})
    for ax in axes.flat:ax.axis("off")
    fig.legend(handles=[Patch(color=[.1,.8,.3],label="Tinta detectada"),Patch(color=[.95,.15,.1],label="Falsa alarma"),Patch(color=[.1,.4,.8],label="Tinta omitida")],loc="lower center",ncol=3,bbox_to_anchor=(.5,-.035))
    fig.suptitle("Comparación controlada en PHerc0139 w039 · sin ajuste de umbral sobre la prueba",fontsize=16)
    f=out/"benchmark_w039.png";fig.savefig(f,dpi=160,bbox_inches="tight");plt.close(fig);figures.append(f.name)

    sweep=np.load(out/"p800_depth_sweep.npz")
    p800=json.loads((out/"p800_summary.json").read_text())
    valid=sweep["valid"]
    own=sweep["z+0"]
    official=[np.load(out/f"official_seed{s}_z+0.npy") for s in (42,43)]
    mean_official=np.mean(official,axis=0)
    own_top=(own>np.quantile(own[valid],.9))&valid
    official_top=(mean_official>np.quantile(mean_official[valid],.9))&valid
    union=own_top|official_top
    overlap=float((own_top&official_top).sum()/max(1,union.sum()))
    stability={}
    for offset in (-2,2):
        other=sweep[f"z{offset:+d}"]
        stability[str(offset)]=float(np.corrcoef(own[valid],other[valid])[0,1])
    fig,axes=plt.subplots(2,4,figsize=(18,9),constrained_layout=True)
    axes[0,0].imshow(sweep["surface"],cmap="gray");axes[0,0].set_title("PHerc0800: CT original\nRecorte nativo de 8,85 × 8,85 mm")
    maps=[(axes[0,1],own,f"{NAMES[p800['selected_by_validation']]}\nMedia de dos semillas, Z central"),
          (axes[0,2],official[0],"Referencia oficial · semilla 42\nZ central, puntuación original"),
          (axes[0,3],official[1],"Referencia oficial · semilla 43\nZ central, puntuación original"),
          (axes[1,0],sweep["z-2"],f"Modelo local, Z −2\nCorrelación con centro {stability['-2']:.2f}"),
          (axes[1,1],sweep["z+2"],f"Modelo local, Z +2\nCorrelación con centro {stability['2']:.2f}"),
          (axes[1,2],mean_official,"Media de referencias oficiales\nZ central")]
    for ax,prob,title in maps:
        ax.imshow(np.ma.masked_where(~valid,prob),cmap="inferno",vmin=0,vmax=1);ax.set_title(title)
    rgb=np.zeros((*valid.shape,3),np.float32)
    rgb[own_top&~official_top]=[.8,.3,.8];rgb[official_top&~own_top]=[.2,.6,.95];rgb[own_top&official_top]=[.1,.9,.4]
    axes[1,3].imshow(rgb);axes[1,3].set_title(f"10 % superior de cada modelo\nCoincidencia Jaccard {overlap:.3f}")
    bar(axes[0,0],valid.shape,8.64,label="2 mm",length=2000)
    for ax in axes.flat:ax.axis("off")
    fig.suptitle("PHerc0800 · exploración sin etiquetas: las puntuaciones y la coincidencia no confirman tinta",fontsize=15)
    fig.text(.5,-.015,"Probabilidades sin calibración común entre modelos. Verde: coinciden; magenta: solo local; azul: solo referencia.",ha="center",fontsize=10)
    f=out/"pherc0800_comparison.png";fig.savefig(f,dpi=160,bbox_inches="tight");plt.close(fig);figures.append(f.name)

    fig,axes=plt.subplots(2,3,figsize=(13,9),constrained_layout=True)
    official_stability={}
    for row,seed in enumerate((42,43)):
        central=np.load(out/f"official_seed{seed}_z+0.npy")
        for column,offset in enumerate((-2,0,2)):
            probability=np.load(out/f"official_seed{seed}_z{offset:+d}.npy")
            correlation=float(np.corrcoef(central[valid],probability[valid])[0,1])
            official_stability[f"seed{seed}_z{offset:+d}"]=correlation
            axes[row,column].imshow(np.ma.masked_where(~valid,probability),cmap="inferno",vmin=0,vmax=1)
            axes[row,column].set_title(f"Referencia oficial · semilla {seed} · Z {offset:+d}\nCorrelación con centro {correlation:.2f}")
            axes[row,column].axis("off")
    fig.suptitle("PHerc0800 · controles de profundidad de ambos modelos oficiales",fontsize=15)
    fig.text(.5,-.015,"Mismo recorte y escala de color 0–1. Desplazamiento de dos capas: 17,28 µm. Sin etiquetas de tinta.",ha="center",fontsize=10)
    f=out/"official_depth_sweep.png";fig.savefig(f,dpi=160,bbox_inches="tight");plt.close(fig);figures.append(f.name)

    volume=np.load(data/"w039_volume.npy",mmap_mode="r")
    positive=ink&mask;negative=~ink&mask
    pos_mean=[];neg_mean=[];pos_q=[];neg_q=[]
    for layer in volume:
        a,b=layer[positive],layer[negative]
        pos_mean.append(float(a.mean()));neg_mean.append(float(b.mean()))
        pos_q.append(np.quantile(a,[.25,.75]));neg_q.append(np.quantile(b,[.25,.75]))
    z=(np.arange(len(volume))-(len(volume)-1)/2)*9.362
    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    for means,quantiles,color,name in [(pos_mean,np.array(pos_q),"#067647","Tinta etiquetada"),(neg_mean,np.array(neg_q),"#175cd3","Fondo revisado")]:
        axes[0].plot(z,means,color=color,label=name,lw=2);axes[0].fill_between(z,quantiles[:,0],quantiles[:,1],color=color,alpha=.12)
    axes[0].set(xlabel="Profundidad relativa (µm)",ylabel="Intensidad CT (0–255)",title="Media y rango intercuartílico por capa")
    axes[0].legend();axes[0].grid(alpha=.2)
    axes[1].plot(z,np.array(pos_mean)-np.array(neg_mean),color="#7f56d9",lw=2);axes[1].axhline(0,color="gray",lw=1)
    axes[1].set(xlabel="Profundidad relativa (µm)",ylabel="Diferencia de intensidad CT",title="Diferencia media: tinta menos fondo");axes[1].grid(alpha=.2)
    fig.suptitle("Perfiles medidos en w039: etiquetas conocidas, sin generación de imagen",fontsize=14)
    f=out/"known_ink_depth_profiles.png";fig.savefig(f,dpi=170,bbox_inches="tight");plt.close(fig);figures.append(f.name)
    summary={"rows":rows,"test_prevalence":report["runs"][0]["test_at_validation_f1_threshold"]["prevalence"],
             "p800_selected_by_validation":p800["selected_by_validation"],"p800_top_decile_jaccard":overlap,
             "p800_depth_correlations":stability,"official_depth_correlations":official_stability,"figures":figures}
    write_json(out/"summary.json",summary)
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
