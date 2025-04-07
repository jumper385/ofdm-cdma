import numpy as np
import matplotlib.pyplot as plt

from common.qam_helpers import seq_to_qam16, qam16_to_seq
from common.orthoganality_helpers import walsh_hadamard
from common.tx_path_helpers import add_noise

# Generate random QAM symbols for each user (e.g., 8 symbols per user)
num_symbols = 8
user1_seq = np.random.randint(0, 16, num_symbols)
user2_seq = np.random.randint(0, 16, num_symbols)
user3_seq = np.random.randint(0, 16, num_symbols)

user1_sig = seq_to_qam16(user1_seq)
user2_sig = seq_to_qam16(user2_seq)
user3_sig = seq_to_qam16(user3_seq)

code_length = 16 # must be power of 2
codes = walsh_hadamard(code_length)
user1_code = codes[0]
user2_code = codes[1]
user3_code = codes[2]

print("user1_code:", user1_code)
print("user2_code:", user2_code)
print("user3_code:", user3_code)

# for each symbol, multiply with code
user1_chips = np.array([symbol * user1_code for symbol in user1_sig])
user2_chips = np.array([symbol * user2_code for symbol in user2_sig])
user3_chips = np.array([symbol * user3_code for symbol in user3_sig])

# generate time series signal
user1_bb = np.fft.ifft(user1_chips, axis=1)
user2_bb = np.fft.ifft(user2_chips, axis=1)
user3_bb = np.fft.ifft(user3_chips, axis=1)

# sum the signals to create the transmitted signal as worst case; then make noisy
bb_sig_time = user1_bb + user2_bb + user3_bb
bb_sig_time = add_noise(bb_sig_time, sig_snr=20)

# receiver side
received_signal = np.fft.fft(bb_sig_time, axis=1)

user1_recovered = np.array([
    np.sum(symbol_chips * np.conjugate(user1_code)) / code_length 
    for symbol_chips in received_signal
])
user2_recovered = np.array([
    np.sum(symbol_chips * np.conjugate(user2_code)) / code_length 
    for symbol_chips in received_signal
])
user3_recovered = np.array([
    np.sum(symbol_chips * np.conjugate(user3_code)) / code_length 
    for symbol_chips in received_signal
])

# plot signal on output
user1_recovered_seq = qam16_to_seq(user1_recovered)
user2_recovered_seq = qam16_to_seq(user2_recovered)
user3_recovered_seq = qam16_to_seq(user3_recovered)

print("--- USER 1 ---")
print("user1_recovered_seq: ", user1_recovered_seq)
print("user1_original_seq: ", user1_seq)
print("user1 match: ", np.all(user1_recovered_seq == user1_seq))

print("--- USER 2 ---")
print("user2_recovered_seq: ", user2_recovered_seq)
print("user2_original_seq: ", user2_seq)
print("user2 match: ", np.all(user2_recovered_seq == user2_seq))

print("--- USER 3 ---")
print("user3_recovered_seq: ", user3_recovered_seq)
print("user3_original_seq: ", user3_seq)
print("user3 match: ", np.all(user3_recovered_seq == user3_seq))
