"""Small, reversible denoising experiment; no training data is overwritten.

Classical spectral attenuation, protected vocal candidates and soft background
attenuation. This is not a laughter recognizer or a certified clean/noisy label.
"""
import numpy as np
from scipy import signal, ndimage

PROFILES = {
    'mild': dict(noise_strength=1.0, spectral_floor_db=-9., protected_floor_db=-1.5,
                 background_db=-6.),
    'strong': dict(noise_strength=2.0, spectral_floor_db=-15., protected_floor_db=-3.,
                   background_db=-12.),
}
SR=48000
NFFT=2048
HOP=480


def intervals_mask(times, intervals, before=0., after=0.):
    mask=np.zeros(len(times),dtype=bool)
    for a,b in intervals:
        if b>a:
            mask |= (times>=a-before)&(times<=b+after)
    return mask


def analyze(raw, vad, words):
    """Use annotations to protect candidates, not to hard-mute unannotated time."""
    _,times,z=signal.stft(raw,fs=SR,nperseg=NFFT,noverlap=NFFT-HOP,
                          boundary='zeros',padded=True)
    power=np.abs(z)**2
    energy=power.sum(axis=0)
    annotated=intervals_mask(times,vad+words,before=.20,after=.30)
    quiet=~intervals_mask(times,vad+words,before=.40,after=.50)
    # Estimate only from the lower-energy half of unannotated time. These remain
    # noise-reference candidates, never ground-truth silence.
    noise_frames=quiet & (energy<=np.quantile(energy[quiet],.5)) if quiet.any() else quiet
    enough=noise_frames.sum()*HOP/SR>=1.
    noise=np.median(power[:,noise_frames],axis=1) if enough else np.zeros(z.shape[0])
    # Additional transient protection for sounds missed by supplied VAD/text.
    floor=float(np.median(energy[noise_frames])) if enough else float(np.quantile(energy,.15))
    novel=(energy>max(floor*10.,1e-12)) & ~annotated
    # Runs shorter than 50 ms are not reliable onset evidence; their attenuation
    # remains bounded by the background/spectral floors, never a hard mute.
    labels,count=ndimage.label(novel)
    sizes=np.bincount(labels)
    novel=(sizes[labels]>=5)&(labels>0)
    protected=annotated | ndimage.maximum_filter1d(novel,size=41)
    protection=ndimage.gaussian_filter1d(protected.astype(float),sigma=3)
    protection[protected]=1.
    anchor=intervals_mask(times,words)
    # Do not loudness-match based on noise, unknown reactions, or each short clip.
    anchor &= energy>max(float(np.quantile(energy[anchor],.2)) if anchor.any() else 0.,1e-12)
    return dict(z=z,times=times,power=power,noise=noise,protection=protection,
                anchor=anchor,protected=protected,noise_profile_available=bool(enough),
                noise_reference_seconds=float(noise_frames.sum()*HOP/SR),length=len(raw))


def reduce(prepared,profile):
    p=PROFILES[profile];power=prepared['power'];eps=1e-20
    gain=np.sqrt(np.maximum(1-p['noise_strength']*prepared['noise'][:,None]/(power+eps),
                            10**(p['spectral_floor_db']/10)))
    gain=ndimage.gaussian_filter(gain,sigma=(1.,2.))
    protect=prepared['protection']
    gain=np.maximum(gain,protect[None,:]*10**(p['protected_floor_db']/20))
    # Soft attenuation only, with waveform-derived unknown events protected too.
    background=10**(p['background_db']/20)
    gain*=background+(1-background)*protect[None,:]
    _,wave=signal.istft(prepared['z']*gain,fs=SR,nperseg=NFFT,noverlap=NFFT-HOP,
                        input_onesided=True,boundary=True)
    wave=wave[:prepared['length']]
    return wave.astype(np.float32)


def anchor_rms(wave,words):
    # Words, not enclosing transcript segments (which can include long silences).
    selected=np.zeros(len(wave),dtype=bool)
    for a,b in words:
        selected[max(0,round(a*SR)):min(len(wave),round(b*SR))]=True
    if selected.sum()<SR:
        return None
    return float(np.sqrt(np.mean(wave[selected].astype(np.float64)**2)))


def db(value):
    return float(20*np.log10(max(float(value),1e-15)))


def metrics(before,after):
    return dict(before_rms_dbfs=db(np.sqrt(np.mean(before.astype(float)**2))),
                after_rms_dbfs=db(np.sqrt(np.mean(after.astype(float)**2))),
                peak_dbfs=db(np.max(np.abs(after))),
                rms_change_db=db(np.linalg.norm(after)/max(np.linalg.norm(before),1e-15)))
