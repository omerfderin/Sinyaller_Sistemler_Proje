import soundfile as sf
import numpy as np
import sounddevice as sd
import time

def fft(x):
    N = len(x)
    if N <= 1:
        return x
    even = fft(x[::2])
    odd = fft(x[1::2])
    T = []
    for k in range(N // 2):
        T.append(np.exp(-2j * np.pi * k / N) * odd[k])
    
    result = []
    for k in range(N // 2):
        result.append(even[k] + T[k])
    for k in range(N // 2):
        result.append(even[k] - T[k])
    return result

def ifft(X):
    N = len(X)
    X_conj = []
    for x in X:
        X_conj.append(np.conj(x))
    
    x = fft(X_conj)
    result = []
    for val in x:
        result.append(np.conj(val) / N)
    return result

def record_noise():
    rate = 16000
    frame_len = 1024
    adaptation_time = 3.0
    
    window = np.zeros(frame_len)
    for i in range(frame_len):
        window[i] = 0.5 * (1 - np.cos(2 * np.pi * i / (frame_len - 1)))
    
    print(f"Gürültü profili kaydediliyor {int(adaptation_time)} saniye sessiz kalın.")
    noise_data = []

    def callback(data, frames, time_info, status):
        noise_data.append(data[:, 0].copy())

    with sd.InputStream(samplerate=rate, channels=1, 
                       callback=callback, blocksize=frame_len):
        time.sleep(adaptation_time)

    all_frames = np.vstack(noise_data)
    noise_spectrums = []
    
    for frame in all_frames:
        windowed = frame * window
        spectrum = fft(windowed.tolist())
        mag = np.abs(spectrum[:frame_len // 2 + 1])
        noise_spectrums.append(mag)
    
    noise_profile = np.median(np.array(noise_spectrums), axis=0)
    return noise_profile * 1.5

noise_profile = record_noise()
rate = 16000
frame_len = 1024
hop_len = 512
out_buf = np.zeros(frame_len * 2, dtype=np.float32)
weight_buf = np.zeros_like(out_buf)
in_buf = np.zeros(frame_len, dtype=np.float32)
buf_pos = 0
old_gains = np.ones(frame_len // 2 + 1) 
saved_audio = []

def process_audio(data, outdata, frames, time_info, status):
    global out_buf, weight_buf, in_buf, buf_pos, old_gains
    
    floor_level = 0.05
    amp_gain = 2.0
    smooth_factor = 0.5
    low_band = (0, 300)
    mid_band = (300, 3400)
    high_band = (3400, 8000)
    bands = [low_band, mid_band, high_band]
    reduce_factors = [5.0, 2.0, 5.0]
    
    window = np.zeros(frame_len)
    for i in range(frame_len):
        window[i] = 0.5 * (1 - np.cos(2 * np.pi * i / (frame_len - 1)))
    
    band_ranges = []
    for low_f, high_f in bands:
        low_index = int(low_f * frame_len / rate)
        high_index = int(high_f * frame_len / rate)
        band_ranges.append((low_index, high_index))

    audio_in = data[:, 0].copy()
    copy_len = min(len(audio_in), frame_len - buf_pos)
    in_buf[buf_pos:buf_pos + copy_len] = audio_in[:copy_len]
    buf_pos += copy_len
    
    if buf_pos >= frame_len:
        windowed = in_buf * window
        spectrum = fft(windowed.tolist())
        mag = np.abs(spectrum)
        phase = np.angle(spectrum)
        half_len = frame_len // 2 + 1
        clean_mag = np.zeros(half_len)
        
        for band_i, (low_index, high_index) in enumerate(band_ranges):
            reduce_by = reduce_factors[band_i]
            
            for i in range(low_index, min(high_index, half_len)):
                reduced = mag[i] - reduce_by * noise_profile[i]
                reduced = max(reduced, floor_level * mag[i])
                gain = reduced / (mag[i] + 1e-10)
                gain = smooth_factor * old_gains[i] + (1 - smooth_factor) * gain
                old_gains[i] = gain
                clean_mag[i] = gain * mag[i]
        
        full_mag = np.concatenate([clean_mag, clean_mag[-2:0:-1]])
        clean_spectrum = full_mag * np.exp(1j * phase)
        clean_audio = np.real(ifft(clean_spectrum.tolist()))
        clean_audio *= window
        
        out_buf[:frame_len] += clean_audio
        weight_buf[:frame_len] += window ** 2
        
        in_buf[:frame_len-hop_len] = in_buf[hop_len:frame_len]
        in_buf[frame_len-hop_len:] = 0
        buf_pos = frame_len - hop_len
        
    output = np.zeros(frames, dtype=np.float32)
    valid_mask = weight_buf[:frames] > 1e-6
    output[valid_mask] = out_buf[:frames][valid_mask] / weight_buf[:frames][valid_mask]
    output *= amp_gain
    
    quiet_level = 0.02
    quiet_mask = np.abs(output) < quiet_level
    output[quiet_mask] *= 0.08
    
    outdata[:, 0] = np.clip(output, -1.0, 1.0)
    saved_audio.append(outdata[:, 0].copy())
    
    out_buf[:-hop_len] = out_buf[hop_len:]
    out_buf[-hop_len:] = 0
    weight_buf[:-hop_len] = weight_buf[hop_len:]
    weight_buf[-hop_len:] = 0

try:
    with sd.Stream(callback=process_audio, samplerate=rate,
                   blocksize=hop_len, channels=1, dtype='float32'):
        while True:
            time.sleep(0.1)
except KeyboardInterrupt:
    if saved_audio:
        filename = input("Sesin kaydedileceği dosya adını giriniz: ")
        audio_data = np.concatenate(saved_audio)
        sf.write(filename, audio_data, rate)
