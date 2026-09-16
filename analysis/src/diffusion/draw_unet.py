"""Simple U-shaped block diagram of the diffusion UNet (unet.py): one block per
level with its size and channels, down/up arrows, skip connections. Nothing else.

    python3 draw_unet.py --out ../../figures/unet_architecture.png

Drawn BY HAND from unet.UNet's defaults as trained (base=96, mults=(1, 2, 2));
it does not read the network (no torch on the development machine), so update it
if unet.py or the training flags change.

Deliberately simplified: each level actually sends THREE skip tensors across (one
per decoder ResBlock), drawn here as one arrow; the blocks hide the ResBlocks and
self-attention inside them. See unet.py for the detail.
"""
import argparse, os

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default=os.path.join(HERE, '..', '..', 'figures', 'unet_architecture.png'))
    a = ap.parse_args()

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10); ax.set_ylim(-0.4, 6.4); ax.axis('off')
    W, H = 1.5, 0.9

    def block(x, y, text, colour):
        ax.add_patch(FancyBboxPatch((x - W / 2, y - H / 2), W, H,
                                    boxstyle='round,pad=0.02,rounding_size=0.1', fc=colour, ec='0.3', lw=1.3))
        ax.text(x, y, text, ha='center', va='center', fontsize=12)

    def arrow(x0, y0, x1, y1, **kw):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='-|>', color=kw.get('c', '0.2'), lw=kw.get('lw', 1.8),
                                    ls=kw.get('ls', '-'), shrinkA=0, shrinkB=0))

    ys = [5.0, 3.4, 1.8]                     # levels 0, 1, 2
    xe = [1.3, 2.4, 3.5]; xd = [8.7, 7.6, 6.5]; xb, yb = 5.0, 0.4
    sizes = [('16×16', 96), ('8×8', 192), ('4×4', 192)]
    enc, dec, mid = '#dbe9f6', '#fbe3cf', '#e8def3'

    for k, (s, c) in enumerate(sizes):
        block(xe[k], ys[k], '%s\n%d ch' % (s, c), enc)
        block(xd[k], ys[k], '%s\n%d ch' % (s, c), dec)
        arrow(xe[k] + W / 2, ys[k], xd[k] - W / 2, ys[k], c='0.55', ls='--', lw=1.6)      # skip
    block(xb, yb, '4×4\n192 ch', mid)

    for k in range(2):                                                            # down / up
        arrow(xe[k], ys[k] - H / 2, xe[k + 1] - 0.3, ys[k + 1] + H / 2)
        arrow(xd[k + 1] + 0.3, ys[k + 1] + H / 2, xd[k], ys[k] - H / 2)
    arrow(xe[2], ys[2] - H / 2, xb - W / 2, yb)
    arrow(xb + W / 2, yb, xd[2], ys[2] - H / 2)

    ax.text(xe[0], ys[0] + H / 2 + 0.55, '18 ch in', ha='center', fontsize=11)
    arrow(xe[0], ys[0] + H / 2 + 0.45, xe[0], ys[0] + H / 2)
    ax.text(xd[0], ys[0] + H / 2 + 0.55, '9 ch out', ha='center', fontsize=11)
    arrow(xd[0], ys[0] + H / 2, xd[0], ys[0] + H / 2 + 0.45)

    ax.text(xe[1] - 1.5, ys[1] - 0.1, 'encoder', ha='center', fontsize=11, color='0.4', rotation=90)
    ax.text(xd[1] + 1.5, ys[1] - 0.1, 'decoder', ha='center', fontsize=11, color='0.4', rotation=90)
    ax.text(xb, yb - H / 2 - 0.25, 'bottleneck', ha='center', fontsize=11, color='0.4')
    ax.text(5.0, ys[0] + 0.25, 'skip connections', ha='center', fontsize=10, color='0.5')

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=150, bbox_inches='tight')
    print('wrote', a.out)


if __name__ == '__main__':
    main()
