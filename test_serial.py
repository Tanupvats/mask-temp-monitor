import serial
import time

PORT = 'COM5' 
BAUD = 115200

print(f"[*] Attempting to connect to {PORT}...")
try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
    print("[*] Connected! Listening for raw data (Press Ctrl+C to stop)...\n")
    
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            print(f"Raw data received: {line}")
            
except serial.SerialException as e:
    print(f"[!] Access Denied or Port Not Found: {e}")
    print("[!] Is the Arduino Serial Monitor still open?")
except KeyboardInterrupt:
    print("\n[*] Exiting.")