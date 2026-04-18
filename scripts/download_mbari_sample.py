"""
Download a small sample from MBARI Pacific Sound (16kHz) using HTTP range request.
Full files are ~4GB (24h recording) — we grab first 60 seconds only.

16kHz WAV, mono = 16000 samples/s * 2 bytes * 60s = 1,920,000 bytes + 44 byte header
"""
import requests
import struct
import os

BUCKET = "pacific-sound-16khz"
FILE = "2024/01/MARS-20240101T000000Z-16kHz.wav"
URL = f"https://{BUCKET}.s3.amazonaws.com/{FILE}"
OUTPUT = "data/mbari/sample_60s.wav"

# 60 seconds of 16kHz mono 16-bit = 1,920,000 bytes of audio data
SECONDS = 60
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
CHANNELS = 1
AUDIO_BYTES = SECONDS * SAMPLE_RATE * BYTES_PER_SAMPLE * CHANNELS
HEADER_SIZE = 44  # standard WAV header
TOTAL_BYTES = HEADER_SIZE + AUDIO_BYTES

print(f"Downloading first {SECONDS}s from MBARI ({TOTAL_BYTES / 1024:.0f} KB)...")
print(f"URL: {URL}")

response = requests.get(URL, headers={"Range": f"bytes=0-{TOTAL_BYTES - 1}"})
print(f"Status: {response.status_code} ({len(response.content)} bytes)")

if response.status_code in (200, 206):
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

    data = response.content

    # Read original WAV header to get actual format
    if data[:4] == b'RIFF':
        # Fix the RIFF chunk size to match our truncated data
        fixed_data = bytearray(data)
        # RIFF chunk size = file size - 8
        struct.pack_into('<I', fixed_data, 4, len(fixed_data) - 8)
        # data chunk size (starts at byte 40 in standard WAV)
        struct.pack_into('<I', fixed_data, 40, len(fixed_data) - HEADER_SIZE)

        with open(OUTPUT, 'wb') as f:
            f.write(fixed_data)
        print(f"Saved: {OUTPUT}")
    else:
        print("WARNING: Not a WAV file header!")
        with open(OUTPUT, 'wb') as f:
            f.write(data)
else:
    print(f"Failed: {response.text[:200]}")
