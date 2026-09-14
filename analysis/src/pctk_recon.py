"""Fan-to-parallel rebinning + parallel-beam FBP for the PcTK count sinograms.

Geometry from 1_inputdata/v_N0.mat:  R=600 mm source-to-isocentre, Rd=1080 mm
source-to-detector, 1854 channels at 225 um pitch (equiangular), 2000 views / 360 deg.
"""
import numpy as np, matplotlib; matplotlib.use('Agg')
from paths import PCTK, FIGS, OUT
import matplotlib.pyplot as plt

O=PCTK+'/2_outputdata/'; B='_dpix_225_dz_1600_r0_24_esig_2.0_Eth_20.0_50.0_65.0_80.0_keV'
SH=(4,1854,7,2000); ETH=['[20-50)','[50-65)','[65-80)','[80+)']
Nch,Nview,ROW = 1854,2000,3
R, Rd, dpix = 600.0, 1080.0, 0.225                  # mm
dgam = dpix/Rd
gam  = (np.arange(Nch)-(Nch-1)/2)*dgam              # fan angle per channel
beta = np.arange(Nview)*2*np.pi/Nview               # source angle

ld=lambda t: np.memmap(O+'m4_sino_pcd_%s%s.flt'%(t,B),dtype='<f4',mode='r').reshape(SH,order='F')

def line_integral(D,l):
    s=np.array(D[l,:,ROW,:],dtype=np.float64)       # (Nch, Nview)
    return -np.log(np.maximum(s,1e-3)/s[1,0])

def rebin(g):
    """fan (gamma, beta) -> parallel (t, theta).  t = R sin(gamma), theta = beta + gamma."""
    out=np.empty_like(g)
    for i in range(Nch):                            # interpolate along view, periodic
        b=(beta-gam[i])%(2*np.pi)
        out[i]=np.interp(b,beta,g[i],period=2*np.pi)
    t=R*np.sin(gam)
    tu=np.linspace(t[0],t[-1],Nch)                  # resample to uniform t
    return np.stack([np.interp(tu,t,out[:,k]) for k in range(Nview)],1), tu

def fbp(p,tu,Npix=512,fov=250.0):
    dt=tu[1]-tu[0]; Npad=1<<int(np.ceil(np.log2(2*len(tu))))
    filt=2*np.abs(np.fft.fftfreq(Npad))/dt
    P=np.fft.fft(p,Npad,axis=0)*filt[:,None]
    pf=np.real(np.fft.ifft(P,axis=0))[:len(tu)]
    ax=(np.arange(Npix)-(Npix-1)/2)*(fov/Npix)
    X,Y=np.meshgrid(ax,ax); img=np.zeros((Npix,Npix))
    th=beta                                          # theta grid == beta grid
    for k in range(Nview):
        tp=X*np.cos(th[k])+Y*np.sin(th[k])
        img+=np.interp(tp,tu,pf[:,k],left=0,right=0)
    return img*(np.pi/Nview)                         # 360 deg -> half weight

M,N=ld('mean'),ld('noisy')
res={}
for l in range(4):
    for tag,D in (('mean',M),('noisy',N)):
        p,tu=rebin(line_integral(D,l)); res[(tag,l)]=fbp(p,tu)
        print('window %d %-5s  mu range %.4f .. %.4f /mm'%(l+1,tag,res[(tag,l)].min(),res[(tag,l)].max()),flush=True)

fig,ax=plt.subplots(2,4,figsize=(16,8.4))
for l in range(4):
    for r,tag in enumerate(('mean','noisy')):
        I=res[(tag,l)]*10.0                          # 1/mm -> 1/cm
        ax[r,l].imshow(I,cmap='gray',vmin=0.15,vmax=0.35)
        ax[r,l].set_title('%s  %s keV'%(tag,ETH[l]),fontsize=9); ax[r,l].axis('off')
fig.suptitle('FBP reconstruction of the PcTK sinogram, row 4  (display 0.15-0.35 cm$^{-1}$)')
fig.tight_layout(); fig.savefig(FIGS+'/recon_fbp_4bin.png',dpi=120,bbox_inches='tight')
np.save(OUT+'/recon_mu_per_cm.npy',np.stack([[res[(t,l)]*10 for l in range(4)] for t in ('mean','noisy')]))
print('wrote viewE_recon.png, recon_mu_per_cm.npy')
