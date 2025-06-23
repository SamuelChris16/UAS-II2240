import cv2
import numpy as np
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
from collections import Counter
import heapq

# Fungsi Run-Length Encoding (RLE)
def rle_encode(arr):
    flat = arr.flatten()
    result = []
    prev = flat[0]
    count = 1
    for val in flat[1:]:
        if val == prev:
            count += 1
        else:
            result.append((prev, count))
            prev = val
            count = 1
    result.append((prev, count))
    return result

def rle_decode(rle, shape):
    flat = []
    for val, count in rle:
        flat.extend([val] * count)
    return np.array(flat).reshape(shape)

# Fungsi Huffman Coding
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

def huffman_decode(encoded, tree, size):
    decoded = []
    node = tree
    for bit in encoded:
        node = node.left if bit == '0' else node.right
        if node.symbol is not None:
            decoded.append(node.symbol)
            node = tree
        if len(decoded) == size:
            break
    return decoded

# Load image and convert to YCrCb
img = cv2.imread('sample.jpg')
if img is None:
    raise FileNotFoundError("Gambar tidak ditemukan.")
img = cv2.resize(img, (128, 128))
img_ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

# Ambil channel luminance (Y)
Y = img_ycc[:, :, 0].astype(np.float32) - 128

# Ambil 1 blok 8x8
block = Y[0:8, 0:8]

# DCT
dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')

# Matriks kuantisasi standar
Q = np.array([[16,11,10,16,24,40,51,61],
              [12,12,14,19,26,58,60,55],
              [14,13,16,24,40,57,69,56],
              [14,17,22,29,51,87,80,62],
              [18,22,37,56,68,109,103,77],
              [24,35,55,64,81,104,113,92],
              [49,64,78,87,103,121,120,101],
              [72,92,95,98,112,100,103,99]])

# Kuantisasi
quantized = np.round(dct_block / Q).astype(np.int16)

# RLE Encoding
rle = rle_encode(quantized)

# Huffman Encoding
flat_quantized = quantized.flatten()
tree = build_huffman_tree(flat_quantized)
codebook = build_codes(tree)
encoded = huffman_encode(flat_quantized, codebook)

# Huffman Decoding
decoded_flat = huffman_decode(encoded, tree, flat_quantized.size)
decoded_quantized = np.array(decoded_flat).reshape(quantized.shape)

# Dekuantisasi
dequantized = decoded_quantized * Q

# IDCT
idct_block = idct(idct(dequantized.T, norm='ortho').T, norm='ortho') + 128

# Blok asli untuk perbandingan
original_block = block + 128
difference = np.abs(original_block - idct_block)

# Visualisasi hasil
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].imshow(original_block, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
axs[0].set_title("Original 8x8 Block")
axs[1].imshow(idct_block, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
axs[1].set_title("Decoded 8x8 Block")
axs[2].imshow(difference, cmap='hot', interpolation='nearest')
axs[2].set_title("Difference")
for ax in axs:
    ax.axis('off')
plt.tight_layout()
plt.subplots_adjust(wspace=0.05, hspace=0)
plt.show()