import math
import soundfile as sf

def fft(x):
    N = len(x)
    if N <= 1:
        return x
    even = fft([x[i] for i in range(0, N, 2)])
    odd = fft([x[i] for i in range(1, N, 2)])
    X = [0] * N
    for k in range(N//2):
        theta = -2 * math.pi * k / N
        e = complex(math.cos(theta), math.sin(theta))
        t =    e * odd[k]
        X[k] = even[k] + t
        X[k + N//2] = even[k] - t
    return X

def ifft(X):
    N = len(X)
    X_conj = [x.conjugate() for x in X]
    x_conj = fft(X_conj)
    x = [value.conjugate() / N for value in x_conj]
    return x

def hann_window(size):
    return [0.5 * (1 - math.cos(2 * math.pi * n / (size - 1))) for n in range(size)]

def power_of_2(n):
    power = 1
    while power < n:
        power *= 2
    return power

def to_mono(audio):
    if len(audio.shape) == 1:
        return audio
    return [sum(audio[i]) / len(audio[i]) for i in range(len(audio))]

def noise_filter(input_file, output_file_file, noise_start=0, noise_end=1, frame_size=1024, hop_size=256):
    fft_size = power_of_2(frame_size) 
    audio, sample_rate = sf.read(input_file)
    
    if len(audio.shape) > 1:
        audio = to_mono(audio)
    
    max_val = max(abs(min(audio)), abs(max(audio)))
    if max_val > 0:
        audio = [sample / max_val for sample in audio]
    
    window = hann_window(fft_size)
    noise_start_sample = int(noise_start * sample_rate)
    noise_end_sample = min(int(noise_end * sample_rate), len(audio))
    noise_profile = [0.0] * (fft_size // 2 + 1)
    frame_count = 0
    
    for i in range(noise_start_sample, noise_end_sample - fft_size + 1, hop_size):
        frame = audio[i:i + fft_size]
        if len(frame) < fft_size:
            continue
        windowed_frame = [frame[j] * window[j] for j in range(fft_size)]
        fft_frame = fft(windowed_frame)
        for j in range(fft_size // 2 + 1):
            noise_profile[j] += abs(fft_frame[j]) ** 2
        frame_count += 1
    
    noise_profile = [math.sqrt(val / frame_count) for val in noise_profile]
    output_file_len = len(audio) + fft_size - hop_size
    output_file_audio = [0.0] * output_file_len
    weights = [0.0] * output_file_len
    alpha = 3
    beta = 0.0000000001
    
    for i in range(0, len(audio) - fft_size + 1, hop_size):
        frame = audio[i:i + fft_size]
        windowed_frame = [frame[j] * window[j] for j in range(fft_size)]
        fft_frame = fft(windowed_frame)
        for j in range(len(fft_frame)):
            magnitude = abs(fft_frame[j])
            signal_power = magnitude ** 2
            phase = math.atan2(fft_frame[j].imag, fft_frame[j].real)
            noise_index = j if j < len(noise_profile) else len(noise_profile) - 2 - (j - len(noise_profile))
            noise_index = max(0, min(noise_index, len(noise_profile) - 1))
            noise_power = (alpha * noise_profile[noise_index]) ** 2
            gain = max(signal_power / (signal_power + noise_power), beta)
            new_magnitude = magnitude * gain
            fft_frame[j] = complex(new_magnitude * math.cos(phase), new_magnitude * math.sin(phase))
        clean_frame = ifft(fft_frame)
        for j in range(len(clean_frame)):
            clean_sample = clean_frame[j].real * window[j]
            if i + j < len(output_file_audio):
                output_file_audio[i + j] += clean_sample
                weights[i + j] += window[j] ** 2
    
    for i in range(len(output_file_audio)):
        if weights[i] > 1e-6:
            output_file_audio[i] /= weights[i]
    
    clean_audio = output_file_audio[:len(audio)]
    
    if max_val > 0:
        clean_audio = [max(-1.0, min(1.0, sample * max_val)) for sample in clean_audio]
    
    sf.write(output_file_file, clean_audio, sample_rate)

if __name__ == "__main__":
    input_file = input("Ses dosyasının adını giriniz (örn: 15.wav): ")
    output_file = input("Kaydedilecek dosya adını giriniz (örn: ses.wav): ")
    noise_filter(input_file, output_file, noise_start=0.5, noise_end=1.5)
    print(f"Temizlenmiş ses '{output_file}' adlı dosyaya kaydedildi.")