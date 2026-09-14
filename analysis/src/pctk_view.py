"""Visualise the PcTK count sinograms in 2_outputdata/."""
import numpy as np, matplotlib; matplotlib.use('Agg')
from paths import PCTK, FIGS
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

O=PCTK+'/2_outputdata/'; B='_dpix_225_dz_1600_r0_24_esig_2.0_Eth_20.0_50.0_65.0_80.0_keV'
SH=(4,1854,7,2000); ETH=['[20-50)','[50-65)','[65-80)','[80+)']
ld=lambda t: np.memmap(O+'m4_sino_pcd_%s%s.flt'%(t,B),dtype='<f4',mode='r').reshape(SH,order='F')
M,N=ld('mean'),ld('noisy')
ROW=3
m=np.array(M[:,:,ROW,:],dtype=np.float64)   # (4, 1854, 2000)
n=np.array(N[:,:,ROW,:],dtype=np.float64)
air=m[:,1,0]                                 # unattenuated reference per window
print('air per window:',np.round(air,1))

# --- Fig A: raw count sinograms, mean vs noisy ---------------------------
fig,ax=plt.subplots(2,4,figsize=(16,7))
for l in range(4):
    for r,(D,t) in enumerate([(m,'mean'),(n,'noisy')]):
        im=ax[r,l].imshow(D[l].T,aspect='auto',cmap='gray',norm=LogNorm(2e3,1.1e6),
                          extent=[0,1854,2000,0])
        ax[r,l].set_title('%s  %s keV'%(t,ETH[l]),fontsize=9)
        ax[r,l].set_xlabel('channel'); ax[r,l].set_ylabel('view')
fig.colorbar(im,ax=ax,label='counts',shrink=.7)
fig.suptitle('PcTK count sinogram, detector row 4 of 7  (log scale)')
fig.savefig(FIGS+'/data_sinogram_counts_4bin.png',dpi=120,bbox_inches='tight'); plt.close(fig)

# --- Fig B: line integrals, what reconstruction actually eats -------------
g=-np.log(np.maximum(m,1e-3)/air[:,None,None])
fig,ax=plt.subplots(1,4,figsize=(16,4))
for l in range(4):
    im=ax[l].imshow(g[l].T,aspect='auto',cmap='magma',vmin=0,vmax=g.max(),extent=[0,1854,2000,0])
    ax[l].set_title('line integral, %s keV'%ETH[l],fontsize=9); ax[l].set_xlabel('channel')
ax[0].set_ylabel('view'); fig.colorbar(im,ax=ax,label=r'$-\ln(N/N_0)$',shrink=.8)
fig.suptitle('Line integrals (noise-free) -- the input to reconstruction')
fig.savefig(FIGS+'/data_line_integrals.png',dpi=120,bbox_inches='tight'); plt.close(fig)

# --- Fig C: one view, profiles + noise ------------------------------------
V=0
fig,ax=plt.subplots(1,3,figsize=(16,4))
for l in range(4):
    ax[0].semilogy(m[l,:,V],lw=1,label=ETH[l])
ax[0].set_title('view 1: counts vs channel'); ax[0].set_xlabel('channel'); ax[0].legend(fontsize=8)
ax[0].set_ylabel('counts')
sl=slice(900,960)
ax[1].plot(np.arange(*sl.indices(1854)),m[0,sl,V],'k-',lw=2,label='mean')
ax[1].plot(np.arange(*sl.indices(1854)),n[0,sl,V],'r-',lw=1,label='noisy')
ax[1].set_title('window 1, channels 900-960: noise'); ax[1].set_xlabel('channel'); ax[1].legend(fontsize=8)
z=((n-m)/np.sqrt(np.maximum(m,1e-9))).ravel()
ax[2].hist(z,bins=120,range=(-5,5),density=True,color='C0',alpha=.75)
x=np.linspace(-5,5,200); ax[2].plot(x,np.exp(-x*x/2)/np.sqrt(2*np.pi),'k--',lw=1.5,label='N(0,1)')
ax[2].set_title('QA: (noisy-mean)/sqrt(mean)   mean=%.3f  sd=%.3f'%(z.mean(),z.std()),fontsize=9)
ax[2].legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIGS+'/data_noise_qa.png',dpi=120,bbox_inches='tight'); plt.close(fig)
print('z-score: mean %.4f  sd %.4f'%(z.mean(),z.std()))

# --- Fig D: the authors' reference reconstruction -------------------------
fig,ax=plt.subplots(1,4,figsize=(15,4))
for l in range(4):
    I=np.fromfile(PCTK+'/5_refdata/m2_image_z112_ft_win%d_512x512.flt'%(l+1),dtype='<f4').reshape(512,512)
    ax[l].imshow(I,cmap='gray'); ax[l].set_title('refdata FBP, window %d'%(l+1),fontsize=9); ax[l].axis('off')
fig.suptitle("Authors' reference reconstruction (5_refdata, 32-row config) -- what the phantom looks like")
fig.tight_layout(); fig.savefig(FIGS+'/recon_pctk_authors_reference.png',dpi=120,bbox_inches='tight'); plt.close(fig)
print('wrote viewA..viewD')
