from distutils.log import debug
import numpy as np
import matplotlib.pyplot as plt
from packetizer import find_packet_candidate_time
from helpers import estimate_offset, fshift, resample

class SpectrumCapture:
    """Class for storing raw captures and providing coarsely packetized Drone ID frames"""
    raw_data: np.array
    sampling_rate: float
    packets: list
    debug: bool

    def __init__(self, raw_data=None, skip_detection=False, Fs=50e6, debug=False, p_type = "droneid", legacy = False):
        """Read capture from file"""
        self.legacy = legacy
        self.raw_data = raw_data
        self.debug = debug
        self.sampling_rate = Fs
        self.packet_type = p_type
        self.droneid_found = False
        if skip_detection:
            print("-"*10+"THIS IS 0"+"-"*10)
            self.packets = [self.raw_data, ]
        else:
            print("-"*10+"THIS IS 1"+"-"*10)
            self._packetize_coarse()

        if debug:
            print(f"SpectrumCapture: found {len(self.packets)} packets")

    def _packetize_coarse(self):
        """Packetize input data"""
        self.droneid_found = False

        self.packets, cfo = find_packet_candidate_time(self.raw_data, self.sampling_rate, debug = self.debug, packet_type=self.packet_type, legacy = self.legacy)

        if self.debug:
            # show all packets found
            for p in self.packets:
                print("-"*10+"THIS IS packet coarse"+"-"*10)
                plt.specgram(p,Fs=self.sampling_rate)
                plt.savefig("paper/spect.png")
                plt.show()

        if len(self.packets) > 0:
            self.droneid_found = True
            print(f"droneid_found, -> length {len(self.packets)}\n\n")
            print()
            

        if not self.droneid_found:
            if self.debug:
                print("Could not verify DroneID packet!")

        #self.packets = droneid_pkt

    def get_packet_samples(self, pktnum=0, debug=False, save=False):
        """Return a Drone ID frame with center frequency corrected and resampled to 15.36 MHz."""
        if pktnum >= len(self.packets):
            raise ValueError("Only %i packets available but you requested packet %i" % (len(self.packets), pktnum))

        packet_data = self.packets[pktnum].copy()

        # correct frequency offset
        print(f"get_packet_samples pkt={pktnum}")
        offset, success = estimate_offset(packet_data, self.sampling_rate, packet_type=self.packet_type)
        if success:
            packet_data = fshift(packet_data, -1.0*offset, self.sampling_rate)
        else:
            return ValueError("Cannot estimate carrier offset for packet %i" % (pktnum))

        if self.packet_type == "droneid" or self.packet_type == "beacon":
            resample_rate = 15.36e6
        elif self.packet_type == "c2":
            resample_rate = 1.92e6

        # resample to LTE freq
        if self.sampling_rate > resample_rate + .1e6:
            if debug:
                print("Resampling from %i MHz to %f MHz" % ((self.sampling_rate / 1e6),resample_rate))
            packet_data = resample(packet_data, self.sampling_rate, resample_rate)
            if save:
                self.save_resampled_packet(packet_data)
        elif self.sampling_rate < resample_rate - .1e6:
            raise ValueError("Your sampling rate is too low")
        else:
            if debug:
                print("Sampling rate matches, not resampling.")
        
        return packet_data
    
    def save_resampled_packet(self, packet_data):
        complex_data = packet_data.astype(np.complex64)

        complex_data.tofile('resampled_file_1951.fc32')