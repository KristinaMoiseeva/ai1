"""ЛР1, вариант 4: kNN + softmax на MNIST."""
from __future__ import annotations
import csv, json, platform, time
from datetime import datetime, timezone
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from scripts.classifiers import KNearestNeighbor, Softmax
from scripts.classifiers.softmax import softmax_loss_naive, softmax_loss_vectorized
from scripts.data_utils import load_dataset
from scripts.gradient_check import eval_numerical_gradient

ROOT=Path(__file__).resolve().parent
K_CHOICES=(1,3,5,7,9,11,15)
LEARNING_RATES=(0.01,0.05,0.1)
REGULARIZATION=(1e-4,1e-3)

def accuracy(y,p): return float(np.mean(np.asarray(y)==np.asarray(p)))
def write_json(path,v): path.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def write_csv(path,rows):
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def flat(images,scale): return images.reshape(len(images),-1).astype(np.float64)/scale

def distances(knn, q, chunk=200):
    out=np.empty((len(q),len(knn.X_train)))
    for s in range(0,len(q),chunk): out[s:s+chunk]=knn.compute_distances_no_loops(q[s:s+chunk])
    return out

def predict_many(d,y,k_values):
    classes=np.unique(y); k_values=sorted(k_values); out={k:np.empty(len(d),dtype=y.dtype) for k in k_values}
    order=np.argsort(d,axis=1,kind="stable")[:,:max(k_values)]
    for i,row in enumerate(order):
        votes=np.zeros(len(classes),dtype=int)
        for n,idx in enumerate(row,1):
            votes[np.searchsorted(classes,y[idx])]+=1
            if n in out: out[n][i]=classes[np.argmax(votes)]
    return out

def run_experiment(dataset="mnist", output_dir=None, seed=42, num_iters=600):
    if dataset!="mnist": raise ValueError("Variant 4 uses only MNIST")
    out=ROOT/"outputs/mnist" if output_dir is None else Path(output_dir)
    if not out.is_absolute(): out=ROOT/out
    (out/"figures").mkdir(parents=True,exist_ok=True);(out/"models").mkdir(exist_ok=True)
    cfg={"seed":seed,"k_choices":list(K_CHOICES),"cv_folds":5,"learning_rates":list(LEARNING_RATES),"regularization_strengths":list(REGULARIZATION),"num_iters":num_iters,"batch_size":128,"blas_threads":2}
    started=datetime.now(timezone.utc).isoformat();t0=time.perf_counter()
    with threadpool_limits(limits=2):
        b=load_dataset("mnist"); scale=float(b["pixel_max"]); names=list(b["class_names"])
        Xtr,Xv,Xte=(flat(b[f"X_{s}"],scale) for s in ("train","val","test")); ytr,yv,yte=(b[f"y_{s}"] for s in ("train","val","test"))
        summary={"format_version":1,"status":"running","started_at_utc":started,"dataset":{"name":"mnist","variant":4,"image_shape":[28,28],"class_names":names,"pixel_max":scale,"split_counts":{"train":len(Xtr),"val":len(Xv),"test":len(Xte)}},"config":cfg,"environment":{"python":platform.python_version(),"numpy":np.__version__,"platform":platform.platform()}}
        # kNN: choose k by stratified 5-fold CV on train.
        knn=KNearestNeighbor();knn.train(Xtr,ytr);Dtr=distances(knn,Xtr);np.fill_diagonal(Dtr,0)
        rows=[]
        skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)
        for fold,(fit,val) in enumerate(skf.split(Xtr,ytr),1):
            pp=predict_many(Dtr[np.ix_(val,fit)],ytr[fit],K_CHOICES)
            for k,p in pp.items(): rows.append({"k":k,"fold":fold,"accuracy":accuracy(ytr[val],p)})
        scores=[]
        for k in K_CHOICES:
            vals=[r["accuracy"] for r in rows if r["k"]==k];scores.append({"k":k,"mean_accuracy":float(np.mean(vals)),"std_accuracy":float(np.std(vals,ddof=1))})
        best=max(scores,key=lambda x:(x["mean_accuracy"],-x["k"]));best_k=best["k"]
        Dte=distances(knn,Xte);pt=predict_many(Dte,ytr,K_CHOICES); pv=knn.predict(Xv,k=best_k,batch_size=100); ptr=predict_many(Dtr,ytr,[best_k])[best_k]
        summary["knn"]={"best_k":best_k,"cv_mean_accuracy":best["mean_accuracy"],"cv_std_accuracy":best["std_accuracy"],"train_accuracy":accuracy(ytr,ptr),"val_accuracy":accuracy(yv,pv),"test_accuracy":accuracy(yte,pt[best_k]),"cv_scores":scores,"distance_matrix_shape":[len(Xte),len(Xtr)],"checkpoint":"models/knn.npz"}
        write_csv(out/"knn_cv.csv",rows);write_csv(out/"knn_test_curve.csv",[{"k":k,"test_accuracy":accuracy(yte,pt[k])} for k in K_CHOICES]);np.savez_compressed(out/"models/knn.npz",k=np.array(best_k),pixel_max=np.array(scale),class_names=np.array(names))
        # Softmax preprocessing from train statistics only.
        mean=Xtr.mean(axis=0); Ltr=np.column_stack([Xtr-mean,np.ones(len(Xtr))]);Lv=np.column_stack([Xv-mean,np.ones(len(Xv))]);Lte=np.column_stack([Xte-mean,np.ones(len(Xte))])
        rng=np.random.default_rng(seed);Xs=rng.normal(size=(10,20));ys=np.arange(10);Ws=rng.normal(scale=1e-3,size=(20,10));checks=[]
        for reg in (0.,50.):
            loss,ga=softmax_loss_vectorized(Ws,Xs,ys,reg);gn=eval_numerical_gradient(lambda w:softmax_loss_vectorized(w,Xs,ys,reg)[0],Ws.copy(),False,1e-5);rel=np.linalg.norm(ga-gn)/max(1e-12,np.linalg.norm(ga)+np.linalg.norm(gn));checks.append({"reg":reg,"loss":float(loss),"relative_l2_error":float(rel),"checked_parameters":int(Ws.size),"h":1e-5})
        summary["gradient_checks"]={"softmax":{"finite_differences":checks},"softmax_sanity":{"initial_loss":float(softmax_loss_vectorized(np.zeros((Ltr.shape[1],10)),Ltr[:100],ytr[:100],0)[0]),"uniform_loss":float(np.log(10))}}
        baseline=Softmax(random_state=seed);bh=baseline.train(Ltr,ytr,learning_rate=.01,reg=.001,num_iters=num_iters,batch_size=128)
        search=[];best_model=None;best_hist=None;best_acc=-1;best_params=None
        for lr in LEARNING_RATES:
            for reg in REGULARIZATION:
                m=Softmax(random_state=seed);h=m.train(Ltr,ytr,learning_rate=lr,reg=reg,num_iters=num_iters,batch_size=128);va=accuracy(yv,m.predict(Lv));ta=accuracy(ytr,m.predict(Ltr));search.append({"learning_rate":lr,"reg":reg,"train_accuracy":ta,"val_accuracy":va})
                if va>best_acc: best_model,best_hist,best_acc,best_params=m,h,va,{"learning_rate":lr,"reg":reg}
        sp=best_model.predict(Lte)
        summary["softmax"]={"best_params":best_params,"train_accuracy":accuracy(ytr,best_model.predict(Ltr)),"val_accuracy":best_acc,"test_accuracy":accuracy(yte,sp),"baseline":{"learning_rate":.01,"reg":.001,"train_accuracy":accuracy(ytr,baseline.predict(Ltr)),"val_accuracy":accuracy(yv,baseline.predict(Lv)),"test_accuracy":accuracy(yte,baseline.predict(Lte))},"search":search,"checkpoint":"models/softmax.npz"}
        write_csv(out/"softmax_search.csv",search);hist=[]
        for run,h in (("baseline",bh),("best",best_hist)):
            for i,l in enumerate(h,1): hist.append({"classifier":"softmax","run":run,"iteration":i,"loss":float(l)})
        write_csv(out/"loss_history.csv",hist);np.savez_compressed(out/"models/softmax.npz",W=best_model.W,mean_image=mean,scale=np.array(scale),image_shape=np.array([28,28]),class_names=np.array(names),dataset=np.array("mnist"))
        summary["elapsed_seconds"]=time.perf_counter()-t0;summary["completed_at_utc"]=datetime.now(timezone.utc).isoformat();summary["status"]="completed";write_json(out/"summary.json",summary)
    print(f"MNIST variant 4: kNN={summary['knn']['test_accuracy']:.4f}, softmax={summary['softmax']['test_accuracy']:.4f}")
    return summary

if __name__=="__main__": run_experiment()
