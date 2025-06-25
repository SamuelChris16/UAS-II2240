import cv2
import numpy as np
from scipy.fftpack import dct, idct
from collections import Counter
import heapq
import matplotlib.pyplot as plt  # Tambahkan ini

# Fungsi konversi RGB ke YCbCr (standar JPEG)
def rgb_to_ycbcr(img):
    img = img.astype(np.float32)
    Y  =  0.299*img[:,:,2] + 0.587*img[:,:,1] + 0.114*img[:,:,0]
    Cb = -0.168736*img[:,:,2] - 0.331264*img[:,:,1] + 0.5*img[:,:,0] + 128
    Cr =  0.5*img[:,:,2] - 0.418688*img[:,:,1] - 0.081312*img[:,:,0] + 128
    return np.stack([Y, Cb, Cr], axis=2)

# Subsampling 4:2:0
def subsample_420(channel):
    return channel[::2, ::2]

# Zigzag order
def zigzag(input):  
    h, w = input.shape
    result = []
    for s in range(h + w - 1):
        if s % 2 == 0:
            for y in range(s, -1, -1):
                x = s - y
                if x < w and y < h:
                    result.append(input[y, x])
        else:
            for x in range(s, -1, -1):
                y = s - x
                if x < w and y < h:
                    result.append(input[y, x])
    return np.array(result)

# Huffman coding untuk DC
class Node:
    def __init__(self, symbol, freq):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(data):
    freq = Counter(data)
    heap = [Node(symbol, freq) for symbol, freq in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)
        merged = Node(None, node1.freq + node2.freq)
        merged.left = node1
        merged.right = node2
        heapq.heappush(heap, merged)
    return heap[0]

def build_codes(node, prefix="", codebook=None):
    if codebook is None:
        codebook = {}
    if node.symbol is not None:
        codebook[node.symbol] = prefix
    else:
        build_codes(node.left, prefix + "0", codebook)
        build_codes(node.right, prefix + "1", codebook)
    return codebook

def huffman_encode(data, codebook):
    return ''.join([codebook[val] for val in data])

# Load image
img = cv2.imread('sample.jpg')
if img is None:
    raise FileNotFoundError("Gambar tidak ditemukan.")

# Konversi ke YCbCr
img_ycbcr = rgb_to_ycbcr(img)

# Subsampling 4:2:0 pada Cb dan Cr
Y  = img_ycbcr[:,:,0]
Cb = subsample_420(img_ycbcr[:,:,1])
Cr = subsample_420(img_ycbcr[:,:,2])

# Shift ke [-128,127]
Y_shift  = Y - 128
Cb_shift = Cb - 128
Cr_shift = Cr - 128

# Ambil blok 8x8 dari Y
block = Y_shift[1150:1158, 302:310]

# DCT
dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')

# Quantization matrix
Q = np.array([[16,11,10,16,24,40,51,61],
              [12,12,14,19,26,58,60,55],
              [14,13,16,24,40,57,69,56],
              [14,17,22,29,51,87,80,62],
              [18,22,37,56,68,109,103,77],
              [24,35,55,64,81,104,113,92],
              [49,64,78,87,103,121,120,101],
              [72,92,95,98,112,100,103,99]])

quantized = np.round(dct_block / Q).astype(np.int16)

# Zigzag ordering
zigzagged = zigzag(quantized)

# Ekstrak DC dan AC
DC = zigzagged[0]
AC = zigzagged[1:]

# Huffman encoding untuk DC (jika hanya satu blok, gunakan biner 9 bit)
DC_binary = format((DC+256)%512, '09b')  # 9 bit untuk -255..255

# Bitstream encoding untuk AC (9 bit per koefisien)
AC_binary = ''.join([format((x+256)%512, '09b') for x in AC])  # 9 bit untuk -255..255

# Inverse Quantization dan Inverse DCT untuk visualisasi blok hasil rekonstruksi
dequantized = quantized * Q
reconstructed = idct(idct(dequantized.T, norm='ortho').T, norm='ortho')
reconstructed = np.clip(reconstructed, -128, 127)

# Kembalikan ke rentang asli (tanpa shift)
original_block = block + 128
reconstructed_block = reconstructed + 128
difference = np.abs(original_block - reconstructed_block)

# UI: Tampilkan blok asli, blok hasil rekonstruksi, dan perbedaannya
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.title("Blok Asli (Y)")
plt.imshow(original_block, cmap='gray', vmin=0, vmax=255)
plt.colorbar()
plt.subplot(1,3,2)
plt.title("Blok Rekonstruksi")
plt.imshow(reconstructed_block, cmap='gray', vmin=0, vmax=255)
plt.colorbar()
plt.subplot(1,3,3)
plt.title("Perbedaan")
plt.imshow(difference, cmap='hot')
plt.colorbar()
plt.tight_layout()
plt.show()

# Output
print("DC binary:", DC_binary)
print("AC binary:", AC_binary)