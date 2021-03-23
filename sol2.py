import numpy as np
from imageio import imread
from skimage import color as sk
import scipy.io.wavfile
from scipy.signal import convolve2d
import math

def read_image(filename, representation):
    """
    Reads image
    :param filename: name of file
    :param representation: 1 for bw, 2 for color
    :return: image, bw or color as wanted, in 0-1 range
    """
    image = imread(filename)
    image = image.astype('float64')
    image = image/255
    if(representation == 1 and len(image.shape) == 3):
        if(image.shape[2] == 3):
            image = sk.rgb2gray(image)
        else:
            # I downloaded some RGBA images for testing so I added this:
            image = sk.rgb2gray(sk.rgba2rgb(image))
    return image

# ============================= [ PART 1 ] ==========================
def get_DFT_array(N):
    # u*x is a multiplication table of N*N:
    u, x = np.meshgrid(np.arange(N), np.arange(N))
    DFT_array = np.exp(-2 * np.pi * 1j * (u * x) / N)
    return DFT_array


def DFT(signal):
    """
    :param signal: a signal, float 64
    :return: an array of dtype complex128 with same shape.
    """
    N = signal.shape[0]
    DFT_array = get_DFT_array(N)
    return np.dot(DFT_array, signal).astype('complex128')


def IDFT(fourier_signal):
    """
    :param fourier_signal: a fourier signal, complex128.
    :return: an array of dtype complex 128 with same shape.
    """
    N = fourier_signal.shape[0]
    IDFT_array = np.linalg.inv(get_DFT_array(N))
    return np.dot(IDFT_array, fourier_signal).astype('complex128')


def DFT2(image):
    """
    :param image: float 64 bw image
    :return: complex128 signal of same shape.
    """
    flag = False
    if(len(image.shape) == 3):
        flag = True
        shape = image.shape
        image = image[:,:,0]
    temp = DFT(image).T
    temp = DFT(temp).T
    temp = temp.reshape(image.shape)
    if (flag): temp = temp.reshape(shape)
    return temp.astype('complex128')

def IDFT2(fourier_image):
    """
    :param fourier_image: complex128 2d signal
    :return: complex 128 image (???)
    """
    flag = False
    if(len(fourier_image.shape) == 3):
        flag = True
        shape = fourier_image.shape
        fourier_image = fourier_image[:,:,0]
    temp = IDFT(fourier_image).T
    temp = IDFT(temp).T
    if(flag): temp = temp.reshape(shape)
    return temp

# ============================= [ PART 2 ] ==========================

def change_rate(filename, ratio):
    rate, data = scipy.io.wavfile.read(filename)
    new_rate = int(round(rate * ratio))
    scipy.io.wavfile.write("change_rate.wav", new_rate, data)

def resize(data, ratio):
    """
    :param data: signal
    :param ratio: ratio
    :return: resize signal with same dype as signal
    """
    if(len(data) == 0): return data
    fourier_signal = DFT(data)
    resized_signal = resize_helper(fourier_signal, ratio)
    if (len(resized_signal) == 0): return resized_signal
    x = IDFT(resized_signal)
    if(data.dtype == 'complex128'):
        return x.astype('complex128')
    return x.real.astype('float64')


def resize_helper(fourier_signal, ratio):
    fourier_signal = np.fft.fftshift(fourier_signal)
    if(ratio == 1):
        return np.fft.ifftshift(fourier_signal)
    elif(ratio > 1):
        midpoint = int(round(fourier_signal.shape[0] / 2))
        pointer = (1/ratio) * fourier_signal.shape[0] * 0.5
        if(pointer - math.floor(pointer) >= 0.5):
            pointer = math.floor(pointer)
            fourier_signal = fourier_signal[midpoint - pointer:midpoint + pointer + 1]
        else:
            pointer = math.floor(pointer)
            fourier_signal = fourier_signal[midpoint - pointer:midpoint + pointer]

    else:
        pad_size = math.floor(fourier_signal.shape[0] / ratio) - fourier_signal.shape[0]
        if(pad_size % 2 == 1): fourier_signal = np.concatenate((fourier_signal, np.zeros(1)))
        pad_size = math.floor(pad_size*0.5)
        fourier_signal = np.concatenate((np.zeros(pad_size), fourier_signal, np.zeros(pad_size)))
    fourier_signal = np.fft.ifftshift(fourier_signal)
    return fourier_signal



def change_samples(filename, ratio):
    """
    :param filename: filename
    :param ratio: ratio
    :return: float 64 array
    """
    rate, signal = scipy.io.wavfile.read(filename)
    resized_signal = resize(signal, ratio)
    scipy.io.wavfile.write("change_samples.wav", rate, resized_signal)
    return resized_signal

def resize_spectrogram(data, ratio):
    """
    :param data: float64 array
    :param ratio: positive float64
    :return: float64 resized data
    """
    spectogram = stft(data)
    resized_sepctogram = [resize(spectogram[i], ratio) for i in range(spectogram.shape[0])]
    resized_sepctogram = np.array(resized_sepctogram)
    return istft(resized_sepctogram).real.astype('float64')

def resize_vocoder(data, ratio):
    """
    :param data: float64 array
    :param ratio: positive float64
    :return: float64 resized data
    """
    stft_array = stft(data)
    resized_stft_array = phase_vocoder(stft_array, ratio)
    resized_data = istft(resized_stft_array).real.astype('float64')
    return resized_data

# ============================= [ PART 3 ] ==========================

def conv_der(im):
    """
    :param im: float64 image
    :return: float64 magnitude image
    """
    x_conv = np.array([0.5, 0, -0.5])
    x_conv = np.atleast_2d(x_conv)
    y_conv = x_conv.T

    dx = convolve2d(im, x_conv, mode="same")
    dy = convolve2d(im, y_conv, mode="same")

    magnitude = np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2)
    return magnitude.astype('float64')

def fourier_der(im):
    """
    :param im: float64 image
    :return: float64 magnitude image
    """

    fourier_im = DFT2(im)
    fourier_im = np.fft.fftshift(fourier_im)

    N = im.shape[0]
    M = im.shape[1]

    u = np.arange(-N//2, N//2)
    v = np.arange(-M//2, M//2)

    dx = fourier_im * (2*np.pi*1j*(1/N))
    dx = ((dx.T)*u).T
    dx = np.fft.ifft2(dx)

    dy = fourier_im * (2*np.pi*1j*(1/M))
    dy = dy * v
    dy = np.fft.ifft2(dy)

    magnitude = np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2)
    return magnitude.real.astype('float64')


# ============================= [ Your Provided Code ] ==========================

from scipy import signal
from scipy.ndimage.interpolation import map_coordinates


def stft(y, win_length=640, hop_length=160):
    fft_window = signal.windows.hann(win_length, False)

    # Window the time series.
    n_frames = 1 + (len(y) - win_length) // hop_length
    frames = [y[s:s + win_length] for s in np.arange(n_frames) * hop_length]

    stft_matrix = np.fft.fft(fft_window * frames, axis=1)
    return stft_matrix.T


def istft(stft_matrix, win_length=640, hop_length=160):
    n_frames = stft_matrix.shape[1]
    y_rec = np.zeros(win_length + hop_length * (n_frames - 1), dtype=np.float)
    ifft_window_sum = np.zeros_like(y_rec)

    ifft_window = signal.windows.hann(win_length, False)[:, np.newaxis]
    win_sq = ifft_window.squeeze() ** 2

    # invert the block and apply the window function
    ytmp = ifft_window * np.fft.ifft(stft_matrix, axis=0).real

    for frame in range(n_frames):
        frame_start = frame * hop_length
        frame_end = frame_start + win_length
        y_rec[frame_start: frame_end] += ytmp[:, frame]
        ifft_window_sum[frame_start: frame_end] += win_sq

    # Normalize by sum of squared window
    y_rec[ifft_window_sum > 0] /= ifft_window_sum[ifft_window_sum > 0]
    return y_rec


def phase_vocoder(spec, ratio):
    time_steps = np.arange(spec.shape[1]) * ratio
    time_steps = time_steps[time_steps < spec.shape[1]]

    # interpolate magnitude
    yy = np.meshgrid(np.arange(time_steps.size), np.arange(spec.shape[0]))[1]
    xx = np.zeros_like(yy)
    coordiantes = [yy, time_steps + xx]
    warped_spec = map_coordinates(np.abs(spec), coordiantes, mode='reflect', order=1).astype(np.complex)

    # phase vocoder
    # Phase accumulator; initialize to the first sample
    spec_angle = np.pad(np.angle(spec), [(0, 0), (0, 1)], mode='constant')
    phase_acc = spec_angle[:, 0]

    for (t, step) in enumerate(np.floor(time_steps).astype(np.int)):
        # Store to output array
        warped_spec[:, t] *= np.exp(1j * phase_acc)

        # Compute phase advance
        dphase = (spec_angle[:, step + 1] - spec_angle[:, step])

        # Wrap to -pi:pi range
        dphase = np.mod(dphase - np.pi, 2 * np.pi) - np.pi

        # Accumulate phase
        phase_acc += dphase

    return warped_spec