"""Reproduce the revised population figures without empirical transport fitting.
Run from any directory: python3 validation/exact_population.py
Dependencies: numpy, scipy, matplotlib. Seed fixes the structural matrix only.
"""
from pathlib import Path
import csv, json
import numpy as np
from scipy.linalg import svd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results'
OUT.mkdir(exist_ok=True)

rows=[]
matrices={}
checks=[]

for d in (2,5):
    S=np.diag(np.arange(1,d+1,dtype=float)); t=np.zeros(d); t[0]=2
    N=np.array([[0.,0.],[1.,0.]]) if d==2 else np.tril(np.random.default_rng(42).normal(size=(d,d)), -1)/np.sqrt(d)
    matrices[str(d)]=N.tolist()
    for c in np.r_[0.,np.geomspace(.01,10.,61)]:
        F=np.eye(d)+c*N
        G=F.T@F
        # Maximize tr(G S R) over orthogonal R; target map in U is S R u+t.
        U,s,Vt=svd(G@S)
        R=Vt.T@U.T
        H=S@R
        A=np.linalg.solve(F.T,(F@H).T).T
        sym=np.linalg.norm(A-A.T)/max(1.,np.linalg.norm(A))
        assert sym<1e-8, (d,c,sym)
        eigA=np.linalg.eigvalsh((A+A.T)/2)
        assert eigA.min()>0
        assert np.allclose(H@H.T,S@S,rtol=1e-9,atol=1e-9)
        DU=np.linalg.norm(S-H,'fro')**2/(d+2)
        DX=np.linalg.norm(F@(S-H),'fro')**2/(d+2)
        CX=np.linalg.norm(F-F@H,'fro')**2/(d+2)+np.linalg.norm(F@t)**2
        CU=np.linalg.norm(np.eye(d)-S,'fro')**2/(d+2)+np.linalg.norm(t)**2
        sv=np.linalg.svd(F,compute_uv=False); kappa=(sv[0]/sv[-1])**2
        KU=1/d; KX=1/eigA.max()
        BX=CX/KX*max(kappa-1,0); BU=CU/KU*max(kappa-1,0)
        GX=4*np.linalg.norm(F@S,'fro')**2/(d+2)
        GU=4*np.linalg.norm(S,'fro')**2/(d+2)
        assert DX<=min(BX,GX)+1e-7 and DU<=min(BU,GU)+1e-7
        if d==2:
            theta=np.arctan(c*(1-2)/(1*(1+c*c)+2))
            du=2.5*(1-np.cos(theta)); dx=(5+c*c)/2*(1-np.cos(theta))
            checks.append(max(abs(DU-du),abs(DX-dx)))
        rows.append(dict(d=d,c=c,kappa=kappa,DX=DX,DU=DU,CX=CX,CU=CU,KX=KX,KU=KU,BX=BX,BU=BU,GX=GX,GU=GU))

with (OUT/'results_population.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)

(OUT/'structural_matrices.json').write_text(json.dumps(matrices,indent=2))

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axs=plt.subplots(1,2,figsize=(10,3.6),constrained_layout=True)

for ax,d in zip(axs,(2,5)):
    rr=[r for r in rows if r['d']==d and r['c']>0]
    c=np.array([r['c'] for r in rr])
    for key,color,label in [('DX','#145b8c',r'$D_X$'),('DU','#bc5a24',r'$D_U$')]:
        ax.loglog(c,[r[key] for r in rr],color=color,lw=2,label=label)
    ax.set(title=f'{d} dimensions',xlabel='Structural parameter c',ylabel='Squared map discrepancy')
    ax.grid(alpha=.2,which='both'); ax.legend()
fig.savefig(OUT/'population_discrepancies.pdf'); plt.close(fig)
fig,axs=plt.subplots(2,2,figsize=(10,6.2),constrained_layout=True)

for col,d in enumerate((2,5)):
    rr=[r for r in rows if r['d']==d and r['c']>0]; c=[r['c'] for r in rr]
    for row,space in enumerate(('X','U')):
        ax=axs[row,col]
        ax.loglog(c,[r['D'+space] for r in rr],color='#145b8c',lw=2,label='Exact discrepancy')
        ax.loglog(c,[r['B'+space] for r in rr],color='#bc5a24',ls='--',label='Stability bound')
        ax.loglog(c,[r['G'+space] for r in rr],color='#53823b',ls=':',label='Covariance bound')
        ax.set(title=f'{d} dimensions, {space} space',xlabel='Structural parameter c',ylabel='Squared discrepancy / bound')
        ax.grid(alpha=.2,which='both'); ax.legend(fontsize=8)

fig.savefig(OUT/'population_bounds.pdf'); plt.close(fig)
