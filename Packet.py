#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import signal
from zcsequence import zcsequence_f, zcsequence_t
from helpers import corr, fshift, tfft, itfft, with_sample_offset, NFFT, MAXNCARRIERS, NCARRIERS, MAXNCARRIERS_c2, NCARRIERS_c2, CP_LENGTHS_legacy, ZC_SYMBOL_IDX_legacy, CP_LENGTHS, CP_LENGTHS_C2, ZC_SYMBOL_IDX, ZC_SYMBOL_IDX_c2


class Packet:
    """Demodulate frames from raw samples to QPSK data 15.36e6"""
    def __init__(self, raw_samples, Fs=15.36e6, enable_zc_detection=True, debug=True, legacy = True, packet_type = "droneid"):
        self.debug = debug
        self.NCARRIERS = NCARRIERS
        self.MAXNCARRIERS = MAXNCARRIERS

        if legacy and packet_type == "droneid":
            self.CP_LENGTHS = CP_LENGTHS_legacy
            self.ZC_SYMBOL_IDX = ZC_SYMBOL_IDX_legacy
        elif packet_type == "droneid":
            self.CP_LENGTHS = CP_LENGTHS
            self.ZC_SYMBOL_IDX = ZC_SYMBOL_IDX
        elif packet_type == "c2":
            self.CP_LENGTHS = CP_LENGTHS_C2
            self.ZC_SYMBOL_IDX = ZC_SYMBOL_IDX_c2
            self.NCARRIERS = NCARRIERS_c2
            self.MAXNCARRIERS = MAXNCARRIERS_c2          

        # sample rate
        self.Fs = Fs

        # carrier frequency offset
        self.detected_ffo = 0

        # first sample start
        self.start = 0

        # normalize amplitudes
        self.raw_samples = raw_samples
        self.raw_samples /= np.max(np.abs(raw_samples))

        # keep the original data in case we have to correct some operations
        self.raw_samples_orig = self.raw_samples

        self.start, self.detected_ffo = self.find_fine_start(self.raw_samples)

        if self.debug:
            print("First Symbol at Sample %i, FFO %f" % (self.start, self.detected_ffo))

        # coarsely extract symbols
        self.symbols_time_domain, self.symbols_freq_domain = self.raw_data_to_symbols(self.raw_samples_orig, self.start, ffo=self.detected_ffo)

        if enable_zc_detection:
            print(f"CHECK - ZC -- {self.ZC_SYMBOL_IDX[0]} -- {self.ZC_SYMBOL_IDX[1]}")
            # make sure that we actually found the ZC sequence
            zc_seq_1 = self.find_zc_seq(self.symbols_freq_domain[self.ZC_SYMBOL_IDX[0]])
            zc_seq_2 = self.find_zc_seq(self.symbols_freq_domain[self.ZC_SYMBOL_IDX[1]])
        else:
            print("NO CHECK - ZC")
            zc_seq_1 = 600
            zc_seq_2 = 147
    
        # first ZC is variable (coarse sync) so not predictable
        # second ZC for fine sync, must be 147
        if not (zc_seq_2 == 147) and packet_type == "droneid":
            #print("ZC Sequence not found. Expected: 600 and 147, Found: %i and %i" % (zc_seq_1, zc_seq_2))
            raise ValueError("ZC Sequence not found. Expected: 600 and 147, Found: %i and %i" % (zc_seq_1, zc_seq_2))

        print("Found ZC sequences:",zc_seq_1, zc_seq_2)
        self.channel = self.estimate_channel(self.ZC_SYMBOL_IDX[0], zc_seq_1)
        self.channel += self.estimate_channel(self.ZC_SYMBOL_IDX[1], zc_seq_2)
        self.channel *= 0.5

        # We also try with this, but we do not find any results
        #zc_cyc = self.find_zc_shift(self.symbol_equalized(ZC_SYMBOL_IDX[0], self.channel), 600)
        #print("ZC Cyclic Shift: %i" % zc_cyc)

        zc_cyc = 0

        # why do we this just for the first ZC?
        self.sampling_offset = self.find_zc_offset(self.ZC_SYMBOL_IDX[0], 600, zc_cyc)
        print("ZC Offset: %f" % self.sampling_offset)

        self.symbols_time_domain, self.symbols_freq_domain = self.raw_data_to_symbols(self.raw_samples, self.start, ffo=self.detected_ffo, sampling_offset=self.sampling_offset)

        angle = self.find_zc_angle(self.symbols_freq_domain[self.ZC_SYMBOL_IDX[0]], 600)
        self.symbols_time_domain, self.symbols_freq_domain = self.raw_data_to_symbols(self.raw_samples, self.start, ffo=self.detected_ffo, sampling_offset=self.sampling_offset, angle=angle)

        # equalized time domain data without CPs
        yfake = np.zeros(len(self.CP_LENGTHS)*NFFT, dtype=np.complex64)
        for i, symbol_f in enumerate(self.symbols_freq_domain):
            print(f"Processing symbol {i} -- {len(self.CP_LENGTHS)-1}")
            yfake[i*NFFT:(i+1)*NFFT] = itfft(self.symbol_equalized(symbol_f, self.channel))

        if self.debug:
            plt.title("Channel-Equalized Packet")
            plt.specgram(yfake, Fs=Fs)
            plt.savefig("paper/channel-eq.png")
            plt.show()

        self.plot_cp_correlation(self.raw_samples, self.start)

        fine_s1 = self.symbols_time_domain[0][-100:]
        inizio_s2 = self.symbols_time_domain[1][:100]
        self.print_cp(self.raw_samples)
        transizione = np.concatenate([fine_s1, inizio_s2])

        plt.figure(figsize=(12, 4))
        plt.plot(np.arange(-100, 100), np.real(transizione), color='green')
        plt.axvline(x=0, color='red', linestyle='--', label='Junction Point')
        plt.title("Time Transition between Symbol 1 and Symbol 2")
        plt.xlabel("Samples related to the junction point")
        plt.ylabel("Amplitued")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_cp_correlation(self, raw_samples, start_idx, n_fft=NFFT, cp_len=72):
        """
        It detects OFDM symbol boundaries by calculating the sliding cross-correlation between the cyclic prefix (CP) and the symbol tail within a signal segment.
It then generates a two-panel plot comparing the raw signal amplitude to these correlation peaks, adding vertical lines to visually verify that the time synchronization and FFT start positions are correctly aligned."""
        segment_len = (n_fft + cp_len) * 8
        data = raw_samples[start_idx : start_idx + segment_len]
        

        corrs = []
        for i in range(len(data) - n_fft - cp_len):
            win1 = data[i : i + cp_len]
            win2 = data[i + n_fft : i + n_fft + cp_len]
            corrs.append(np.abs(np.sum(win1 * np.conj(win2))))
        
        corrs = np.array(corrs)
        
        plt.figure(figsize=(15, 6))
        
        plt.subplot(2, 1, 1)
        plt.plot(np.abs(data[:len(corrs)]), color='gray', alpha=0.5, label='Signal Magnitude')
        plt.title(f"Timing analysis starting from {start_idx}")
        plt.ylabel("Amplitude")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(corrs, color='blue', label='CP Correlation')
        
        for i in range(8):
            if(i == 7):
                expected_pos = i * (n_fft + 80)
            else:
                expected_pos = i * (n_fft + cp_len)
            plt.axvline(x=expected_pos, color='red', linestyle='--', alpha=0.7)
            if i == 0:
                plt.text(expected_pos, np.max(corrs), ' FFT Start (expected)', color='red')

        plt.title("Correlation peaks")
        plt.xlabel("Samples")
        plt.ylabel("Correlation value")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def check_FFO_symbols(self,index, cp_len):
        """
            This function evaluates Fractional Frequency Offset (FFO) correction success by comparing an OFDM symbol's cyclic prefix (start) with its original tail (end).
            It calculates the residual error between these theoretically identical segments and generates a plot to visually verify their alignment.
        """
        simbolo = self.symbols_time_domain[index] 
        cp_s = simbolo[:cp_len]
        cp_e = simbolo[-cp_len:]

        differenza = np.mean(np.abs(cp_s - cp_e))
        print(f"Residual error post-correction: {differenza}")

        plt.plot(np.real(cp_s), label="Start CP")
        plt.plot(np.real(cp_e), '--', label="End Signal")
        plt.title("Analysis CP after FFO correction")
        plt.legend()
        plt.savefig(f"paper/CP_symbol_{index}_correction.png")
        plt.show()

    def raw_data_to_symbols(self, samples, first_symbol_offset, ffo = None, sampling_offset = None, angle = None, linear_rotation=None):
        """Convert raw samples into OFDM symbols"""

        samples = samples[first_symbol_offset:]

        if ffo != None:
            samples = fshift(samples, -ffo, self.Fs)

        symbols_time_domain = []
        symbols_freq_domain = []

        if sampling_offset != None:
            samples = with_sample_offset(samples, sampling_offset)

        if angle != None:
            samples *= np.exp(-1j * angle)
        
        sample_offset = 0
        for i, cp_len in enumerate(self.CP_LENGTHS):
            symbols_time_domain.append(samples[sample_offset:sample_offset+NFFT+cp_len])
            sample_offset = sample_offset + NFFT + cp_len

        for i, _ in enumerate(symbols_time_domain):
            # skip CP for FFT
            sym = symbols_time_domain[i][self.CP_LENGTHS[i]:]
            symbols_freq_domain.append(tfft(sym))

        if linear_rotation != None:
            for i, symbol in enumerate(symbols_freq_domain):
                x = np.linspace(-.5 * linear_rotation * len(symbol), .5 *
                            linear_rotation * len(symbol), len(symbol))
                symbols_freq_domain[i] *= np.exp(x * 2j * np.pi)

        return symbols_time_domain, symbols_freq_domain

    def estimate_channel(self, sym_index, zc_seq):
        if sym_index not in self.ZC_SYMBOL_IDX:
            raise ValueError("Bad ZC Symbol Index")

        # if sym_index == ZC_SYMBOL_IDX[0]:
        #     zc_seq = 600
        # if sym_index == ZC_SYMBOL_IDX[1]:
        #     zc_seq = 147

        expected_signal = zcsequence_f(zc_seq, NCARRIERS)
        received_signal = self.symbols_freq_domain[sym_index]

        expected_signal[NCARRIERS//2] = 1
        channel = np.divide(received_signal, expected_signal)

        if self.debug:
            plt.title("Channel Estimation")
            plt.plot(np.abs(channel))
            plt.savefig(f"paper/ch_est_{sym_index}.png")
            plt.show()

        return channel

    def symbol_equalized(self, symbol_f, channel):
        return np.divide(symbol_f, channel)

    def find_zc_angle(self, symbol_f, zc_seq):
        a = zcsequence_t(zc_seq, NCARRIERS)

        if (symbol_f == 0).any():
            symbol_f += 1

        adiff = np.angle(a / symbol_f)
        adiff[NCARRIERS//2] = adiff[NCARRIERS//2+1]
        adiff = np.unwrap(adiff)

        slope = np.max(adiff) - np.min(adiff)
        slope = (adiff - np.mean(adiff))
        slope = np.sqrt(np.mean(slope**2))

        if self.debug:
            print("slope", slope)
            print("phase 0", np.angle(symbol_f[NCARRIERS//2]))
            plt.plot(adiff)
            plt.title("Phase diff of ZC Seq")
            plt.savefig("paper/phasedif.png")
            plt.show()

        return np.angle(symbol_f[NCARRIERS//2])

    def find_fine_start(self, samples):
        """Fine-tune symbol start using cyclic prefixes (first symbol only"""
        res = []
        cpl = self.CP_LENGTHS[0]

        for n in range(NFFT, len(samples) - cpl):
            ac = np.sum(samples[n:n+cpl] * np.conj(samples[n-NFFT:n-NFFT+cpl]))
            #ac = np.max(corr(samples[n:n+cpl], samples[n-NFFT:n-NFFT+cpl]))
            res.append(ac)

        res_abs = np.abs(res)
        # distance is roughly number of samples of a symbol at Fs
        peaks, _ = signal.find_peaks(res_abs, distance = 1000)
        peak_prominences, _, _ = signal.peak_prominences(res_abs, peaks)
        # discard small peaks
        peak_index = np.where(peak_prominences > 1.0)[0][0]
        
        for p in peaks:
            print(f"PEAK FOUND AT {p}")

        if self.debug:
            x = np.linspace(0, len(samples) / (NFFT + cpl), len(res_abs))
            plt.plot(x, np.array(res_abs)*300)
            # distance is at least symbol length, so at least NFFT
            plt.scatter(x[peaks], abs(res_abs[peaks])*300, marker='x')
            plt.scatter(x[peaks-cpl//2], abs(res_abs[peaks])*300, marker='x')
            plt.specgram(samples, Fs=NFFT + cpl, NFFT=NFFT//64,
                         window=matplotlib.mlab.window_none, noverlap=0)
            plt.title("Raw Spectrum + Rough Packet Peak Estimation")
            plt.savefig("paper/peaks_est.png")
            plt.show()
            
            x = np.arange(790, 800+20)
            plt.plot(x, np.real(samples[790:800+20]), label='Real', color='blue', alpha=0.8)
            plt.title(f"Start of the signal debugging")
            plt.savefig("paper/start_check.png")
            plt.show()

        start = peaks[peak_index]

        ffo = self.Fs / (2 * np.pi * NFFT) * np.angle(res[start])
        print("FFO: %f" % ffo)
        return start, ffo

    def print_cp(self, samples):
        start_index = 806
        cpl = self.CP_LENGTHS[0] 
        
        plot_end = 800 + NFFT + cpl + 20 
        print("-"*30)
        print("The following 2 plots represent the CP at the beginning of the symbol 1 and the end of that symbols.")
        print("Visually, the samples appear different because the Cyclic Prefix absorbs Inter-Symbol Interference (ISI) from the previous symbol and undergoes phase rotation due to Carrier Frequency Offset (CFO). However, they produce a high correlation peak because the auto-correlation mathematically cancels the phase shift and averages out the noise, allowing the matching underlying signal to constructively add up.")
        self.plot_symbol(samples[start_index-10:start_index+cpl+10], start_index-10, start_index+cpl+10)
        self.plot_symbol(samples[plot_end-cpl-10:plot_end+10], plot_end-cpl-10, plot_end+10)


    def plot_symbol(self, symbol, start, end):
        plt.figure(figsize=(12, 5))
        x = np.arange(start, end)
        plt.plot(x, np.real(symbol), label='Real', color='blue', alpha=0.8)

        plt.title("Symbols")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def find_zc_seq(self, symbol_f):
        res_details = []
        
        # Cerchiamo la root migliore
        for r in range(1, NCARRIERS + 1): 
            a = zcsequence_t(r, NCARRIERS)
            # Calcolo correlazione complessa (una sola volta!)
            corr_vec = corr(symbol_f, a)
            abs_corr = np.abs(corr_vec)
            max_idx = np.argmax(abs_corr)
            peak_val = corr_vec[max_idx] 

            res_details.append({
                'root': r,
                'magnitude': np.abs(peak_val),
                'phase': np.angle(peak_val),
                'offset': max_idx,
                'complex_peak': peak_val
            })

        # CORREZIONE ERRORE: Estraiamo le magnitudo per trovare l'indice del migliore
        magnitudes = [d['magnitude'] for d in res_details]
        best_idx = np.argmax(magnitudes)
        best_match = res_details[best_idx]

        best_r = best_match['root']
        peak_complex = best_match['complex_peak']
        peak_idx = best_match['offset']
        
        if self.debug:
            # Calcolo Confidence Ratio (Punto 1)
            sorted_mags = sorted(magnitudes, reverse=True)
            ratio = sorted_mags[0] / sorted_mags[1] if len(sorted_mags) > 1 else 0
            print(f"--- ZC Detection ---")
            print(f"Root: {best_r} | Offset: {peak_idx} | Phase: {np.angle(peak_complex):.4f} rad")
            print(f"Confidence Ratio: {ratio:.2f}")

        return best_r

    def find_zc_offset(self, symbol_idx, seq, cyc):
        a = zcsequence_t(seq, NCARRIERS)

        resx = []
        resy = []

        # fine-tune sample alignment by seaching for peak in ZC correlation
        samples = self.raw_samples_orig[self.start:]
        samples = fshift(samples, -self.detected_ffo, self.Fs)

        for i in np.linspace(-15, 15, 1000):
            _, symbols_f = self.raw_data_to_symbols(samples, 0, ffo=None, sampling_offset=i)

            zc_sym_f = symbols_f[symbol_idx]

            # prevent division by zero
            if (zc_sym_f == 0).any():
                zc_sym_f += 1

            adiff = np.angle(a / zc_sym_f)
            # remove DC carrier
            adiff[NCARRIERS//2] = adiff[NCARRIERS//2+1]
            adiff = np.unwrap(adiff)

            slope = np.max(adiff) - np.min(adiff)
            slope = (adiff - np.mean(adiff))
            slope = np.sqrt(np.mean(slope**2))

            # x = np.linspace(0, len(adiff)-1, len(adiff))
            #npslope, npoffset = np.polyfit(x, adiff, 1)

            resx.append(i)
            resy.append(slope)

        if self.debug:
            plt.title("RMS for ZC sequence")
            plt.xlabel("Sample Offset Correction")
            plt.ylabel("")
            plt.plot(resx, resy)
            plt.plot(resx[np.argmin(resy)], np.min(resy), marker='X')
            plt.savefig("paper/rmsZc.png")
            plt.show()
    
        return resx[np.argmin(resy)]

    def find_zc_shift(self, symbol_f, seq: int, cyc=0):
        """Find ZC cyclic shift"""
        a = np.zeros(NFFT, dtype=np.complex64)

        a = zcsequence_f(seq, MAXNCARRIERS)
        rx_symbol_f = self.symbol_equalized(symbol_f, self.channel)
    
        am = np.argmax(np.abs(corr(rx_symbol_f, a)))
        return (cyc - am) % (NCARRIERS)
    
    def get_symbol_data(self, linear_rotation=0, _sampling_offset=0, tune=0, skip_zc=False):
        sampling_offset = self.sampling_offset+_sampling_offset
        ffo = self.detected_ffo+tune

        # all symbols
        _, all_symbols_f = self.raw_data_to_symbols(self.raw_samples_orig, self.start, ffo = ffo, sampling_offset=sampling_offset, linear_rotation=linear_rotation)
        
        symbols_f = []
        for i, symbol in enumerate(all_symbols_f):
            if skip_zc and i in self.ZC_SYMBOL_IDX:
                continue
            symbols_f.append(symbol)
        return symbols_f
