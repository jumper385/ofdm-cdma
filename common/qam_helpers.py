import numpy as np

# Convert an integer (0 to 15) to the corresponding 16-QAM symbol
def seq_to_qam16(seq):
    # Define the 16-QAM constellation points
    qam16 = np.array([
        -3-3j, -3-1j, -3+3j, -3+1j,
        -1-3j, -1-1j, -1+3j, -1+1j,
         3-3j,  3-1j,  3+3j,  3+1j,
         1-3j,  1-1j,  1+3j,  1+1j
    ])
    return qam16[seq]

def qam16_to_seq(qam_symbols):
    # Define the 16-QAM constellation points
    qam16 = np.array([
        -3-3j, -3-1j, -3+3j, -3+1j,
        -1-3j, -1-1j, -1+3j, -1+1j,
         3-3j,  3-1j,  3+3j,  3+1j,
         1-3j,  1-1j,  1+3j,  1+1j
    ])
    return np.array([np.argmin(np.abs(qam_symbols[i] - qam16)) for i in range(len(qam_symbols))])