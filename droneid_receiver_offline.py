#!/usr/bin/env python3

import argparse
import numpy as np

from SpectrumCapture import SpectrumCapture
from Packet import Packet
from qpsk import Decoder
from droneid_packet import DroneIDPacket
from gui import interactive
import matplotlib.pyplot as plt

def correct_residual_cfo_ofdm(symbols):
    symbols = np.array(symbols)

    # symbols shape: (n_ofdm_symbols, n_subcarriers)
    n_sym, n_sc = symbols.shape

    # usa solo subcarrier centrali (più stabili)
    sc_center = slice(n_sc//4, 3*n_sc//4)

    phases = []
    for i in range(n_sym):
        phase = np.angle(np.mean(symbols[i, sc_center]))
        phases.append(phase)

    phases = np.unwrap(phases)
    idx = np.arange(len(phases))

    # stima rotazione di fase per simbolo
    p = np.polyfit(idx, phases, 1)
    phase_slope = p[0]

    # applica correzione per simbolo OFDM
    for i in range(n_sym):
        symbols[i, :] *= np.exp(-1j * phase_slope * i)

    return symbols

def decision_directed_phase_correction(symbols, mu=0.01):
    """
    symbols: array 1D di simboli complessi (QPSK)
    mu: passo del loop (0.005–0.02)
    """
    symbols = np.array(symbols, dtype=np.complex128)

    phase_est = 0.0
    out = np.zeros_like(symbols)

    for i, s in enumerate(symbols):
        s_rot = s * np.exp(-1j * phase_est)

        # decisione QPSK
        d = np.sign(np.real(s_rot)) + 1j * np.sign(np.imag(s_rot))
        d /= np.sqrt(2)

        # errore di fase
        err = np.angle(s_rot * np.conj(d))

        # aggiorna stima
        phase_est += mu * err

        out[i] = s_rot

    return out

import numpy as np

# def plot_qpsk(symbols, title="QPSK Constellation"):
#     symbols = np.array(symbols)

#     plt.figure(figsize=(5, 5))
#     plt.scatter(
#         np.real(symbols),
#         np.imag(symbols),
#         s=6,
#         alpha=0.4
#     )
#     plt.axhline(0, color='gray')
#     plt.axvline(0, color='gray')
#     plt.grid(True)
#     plt.title(title)
#     plt.xlabel("In-Phase")
#     plt.ylabel("Quadrature")
#     plt.axis("equal")
#     plt.show()


def main(_args):
    """Decode capture file"""
    raw = np.memmap(_args.input_file, mode='r', dtype="<f").astype(np.float32).view(np.complex64)

    packets_decoded = 0
    crc_error = 0

    drone_coords = []
    app_coords = []

    chunk_samples = int(500e-3 * _args.sample_rate) # in seconds
    chunks = len(raw) // chunk_samples +1

    for i in range(chunks):
        print("Drone-ID Frame Detection")

        capture = SpectrumCapture(raw[i*chunk_samples:(i+1)*chunk_samples], skip_detection = args.skip_detection, Fs=_args.sample_rate, debug=args.debug, legacy=args.legacy)
        print(f"Found {len(capture.packets)} Drone-ID RF frames in spectrum capture.")

        for packet_num, _ in enumerate(capture.packets):
            payload = None

            print(f"################## Decoding Frame {packet_num+1}/{len(capture.packets)} ##################")

            # get a Drone ID frame, resampled and with coarse center frequency correction.
            packet_data = capture.get_packet_samples(pktnum=packet_num)

            try:
                packet = Packet(packet_data, debug=args.debug, enable_zc_detection=not args.disable_zc_detection, legacy=args.legacy)
            except Exception as error:
                print(f"Demodulation FAILED (Frame {packet_num+1}): {error}")
                continue
            
            # GUI for manual RF inspection
            if _args.gui:
                interactive(packet)

            symbols = packet.get_symbol_data(skip_zc=True)
            # symbols = np.array(symbols)

            # symbols = correct_residual_cfo_ofdm(symbols)

            # i = symbols.shape[0] // 2
            # sym = symbols[i]


            # sym_pll = decision_directed_phase_correction(sym, mu=0.01)
            # if(capture.droneid_found):
            #     plot_qpsk(sym, "Before DD PLL")
            #     plot_qpsk(sym_pll, "After DD PLL")
            #     print(f"type {type(capture.packets)}")


            decoder = Decoder(symbols)
            
            #brute force QPSK alignment
            for phase_corr in range(4):
                decoder.raw_data_to_symbol_bits(phase_corr)
                droneid_duml = decoder.magic()

                try:
                    payload = DroneIDPacket(droneid_duml)
                except:
                    continue

                print(f"## Drone-ID Payload ##")
                print(payload)

                if not payload.check_crc():
                    print("CRC error!")

                    # CRC check failed
                    crc_error += 1
                    break

                drone_lat, drone_lon, app_lat, app_lon, height = DroneIDPacket(droneid_duml).get_coords()

                # congrats, you received a valid Drone-ID packet
                packets_decoded += 1

                if drone_lat != 0.0 and drone_lon != 0.0:
                    drone_coords.append((drone_lat, drone_lon, height))

                if app_lat != 0.0 and app_lon != 0.0:
                    app_coords.append((app_lat,app_lon))
    
                # we're done for this packet
                break

            if not payload:
                print(f"Frame {packet_num}/{len(capture.packets)}: Decoding failed.")

    print("\n\n")
    print(f"Frame detection: {len(capture.packets)} candidates")
    print(f"Decoder: {packets_decoded+crc_error} total, CRC OK: {packets_decoded} ({crc_error} CRC errors)")
    
    print("Drone Coordinates:")
    for coords in drone_coords:
        print(coords)

    print("App Coordinates:")
    for coords in app_coords:
        print(coords)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gui', default=False, action="store_true", help="Show interactive")
    parser.add_argument('-i', '--input-file', default="../samples/mini2_sm", help="Binary Sample Input")
    parser.add_argument('-s', '--sample-rate', default="50e6", type=float, help="Sample Rate")
    parser.add_argument('-l', '--legacy', default=False, action="store_true", help="Support of legacy drones (Mavic Pro, Mavic 2)")
    parser.add_argument('-d', '--debug', default=True, action="store_true", help="Enable debug output")
    parser.add_argument('-z', '--disable-zc-detection', default=True, action="store_false", help="Disable per-symbol ZC sequence detection (faster)")
    parser.add_argument('-f', '--skip-detection', default=False, action="store_true", help="Skip packet detection and enforce decoding of input file")
    args = parser.parse_args()

    main(args)
