import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pctk_compare as P
from paths import FIGS

d=P.load(); cfg=dict(P.CFG); vEo,vE1=d['vEo'],d['vE1']
Pr,q,QE=d['Pr'],d['q'],d['QE']
VAR={'full':d['diagE'],
     'off':Pr.sum(0)[:,None]*q[0],
     'escape_only':Pr[0][:,None]*q[0]+(Pr[1]+Pr[2])[:,None]*q[1]}
R9={k:v.reshape(170,9,191) for k,v in VAR.items()}
R ={k:v.sum(1) for k,v in R9.items()}
R['center']=R9['full'][:,4,:]
ideal=np.zeros((170,191))
for i,E in enumerate(vE1): ideal[i,int(np.argmin(abs(vEo-E)))]=QE[i]
eth=np.array(cfg['ETH']); idx=np.digitize(vEo,eth)-1
tw=lambda M: np.stack([M[:,idx==l].sum(1) for l in range(4)],1)
spec=lambda b,bo: d['S']*np.exp(-d['mu'][:,0]*b-d['mu'][:,1]*bo)
N0=cfg['N0']; m=(vEo>=10)&(vEo<=140); mE=vE1<=140

# ---- Fig 1: response matrices -------------------------------------------
panels=[('ideal detector',ideal),('full (sharing + K-fluor)',R['full']),
        ('off (sharing only)',R['off']),('escape_only',R['escape_only'])]
fig,ax=plt.subplots(2,2,figsize=(11,9))
for a,(t,M) in zip(ax.ravel(),panels):
    Z=np.maximum(M[np.ix_(mE,m)],1e-6)
    im=a.imshow(Z,origin='lower',aspect='auto',norm=LogNorm(1e-5,3e-1),cmap='magma',
                extent=[vEo[m][0],vEo[m][-1],vE1[mE][0],vE1[mE][-1]])
    a.plot([10,140],[10,140],'w--',lw=.6,alpha=.5)
    a.plot([10,115],[35,140],'c--',lw=.8,alpha=.7)
    a.set_title(t,fontsize=10); a.set_xlabel('deposited energy $E_{out}$ (keV)'); a.set_ylabel('incident $E_1$ (keV)')
fig.colorbar(im,ax=ax,label='counts / incident photon / keV',shrink=.8)
fig.suptitle('PcTK 3.2 spectral response  (white = photopeak, cyan = K-escape ridge, $E_1-25$ keV)')
fig.savefig(FIGS+'/physics_response_matrix.png',dpi=130,bbox_inches='tight'); plt.close(fig)

# ---- Fig 2: slices -------------------------------------------------------
fig,ax=plt.subplots(1,3,figsize=(14,4))
for a,E in zip(ax,[60,100,120]):
    i=int(np.where(vE1==E)[0][0])
    for k,c in [('full','C0'),('off','C1'),('escape_only','C2'),('center','C3')]:
        a.semilogy(vEo[m],np.maximum(R[k][i][m],1e-6),c,lw=1.3,label=k)
    for t in eth: a.axvline(t,color='.7',lw=.7,ls=':')
    a.axvline(E,color='k',lw=.8,ls='--'); a.axvline(E-24.5,color='c',lw=.8,ls='--')
    a.set_title('$E_1$ = %d keV'%E); a.set_xlabel('$E_{out}$ (keV)'); a.set_ylim(1e-5,.3)
ax[0].set_ylabel('counts / incident photon / keV'); ax[0].legend(fontsize=8)
fig.suptitle('Recorded spectrum for monoenergetic input (black = photopeak, cyan = $E_1-25$ keV K-escape)')
fig.tight_layout(); fig.savefig(FIGS+'/physics_response_slices.png',dpi=130,bbox_inches='tight'); plt.close(fig)

# ---- Fig 3: 3x3 spatial map ---------------------------------------------
s=spec(20.,2.); W9=N0*np.einsum('e,epw->pw',s,np.stack([tw(R9['full'][:,p,:]) for p in range(9)],1))
fig,ax=plt.subplots(1,4,figsize=(13,3.6))
for l,a in enumerate(ax):
    G=W9[:,l].reshape(3,3)
    im=a.imshow(G,cmap='viridis'); a.set_xticks([]); a.set_yticks([])
    for i in range(3):
        for j in range(3):
            a.text(j,i,'%.2f'%G[i,j],ha='center',va='center',
                   color='w' if G[i,j]<G.max()*.6 else 'k',fontsize=9)
    a.set_title('window %d  [%g-%s) keV\ncentre %.0f%% of %.2f counts'
                %(l+1,eth[l],eth[l+1] if l<3 else 'inf',100*G[1,1]/G.sum(),G.sum()),fontsize=9)
fig.suptitle('Where the counts land: 3x3 neighbourhood, 20 cm brain + 2 cm bone  ("center" toggle keeps only the middle cell)')
fig.tight_layout(); fig.savefig(FIGS+'/physics_crosstalk_3x3.png',dpi=130,bbox_inches='tight'); plt.close(fig)

# ---- Fig 4: window counts ------------------------------------------------
paths=cfg['PATHS']; keys=['full','off','escape_only','center']
fig,ax=plt.subplots(1,3,figsize=(14,4))
for a,(nm,b,bo) in zip(ax,paths):
    s=spec(b,bo); wi=N0*(tw(ideal).T@s); x=np.arange(4); w=.19
    for i,k in enumerate(keys):
        a.bar(x+(i-1.5)*w,N0*(tw(R[k]).T@s),w,label=k)
    for j,v in enumerate(wi): a.hlines(v,j-.42,j+.42,color='k',lw=2)
    a.set_yscale('log'); a.set_title(nm,fontsize=10); a.set_xticks(x)
    a.set_xticklabels(['[%g-%s)'%(eth[l],eth[l+1] if l<3 else 'inf') for l in range(4)],fontsize=8)
    a.set_xlabel('window (keV)')
ax[0].set_ylabel('counts / pixel / view'); ax[0].legend(fontsize=8,ncol=2)
fig.suptitle('Window counts by variant;  black bars = ideal detector (ground truth)')
fig.tight_layout(); fig.savefig(FIGS+'/physics_window_ablation.png',dpi=130,bbox_inches='tight'); plt.close(fig)
print('wrote fig1_response.png fig2_slices.png fig3_spatial.png fig4_windows.png')
