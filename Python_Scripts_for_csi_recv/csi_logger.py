import serial
import time
import os
import sys

# --- ΡΥΘΜΙΣΕΙΣ ---
PORT = 'COM6' 
BAUD = 2000000 

def main():
    if not os.path.exists("datasets"):
        os.makedirs("datasets")

    print("\n" + "="*45)
    print("📡 ESP32-C6 RADAR - HIGH SPEED LOGGER")
    print("="*45)
    
    label = input("📝 Δώσε όνομα κίνησης (π.χ. walk_1, fall_3, empty): ")
    if not label.strip():
        print("❌ Δεν έδωσες όνομα. Έξοδος.")
        sys.exit(1)

    filename = f"datasets/{label}_{int(time.time())}.txt"
    bytes_written = 0

    try:
        ser = serial.Serial(PORT, BAUD, timeout=0.1)
        
        if os.name == 'nt':
            ser.set_buffer_size(rx_size=2000000)
        
        print(f"\n🚀 Η καταγραφή ξεκίνησε!")
        print(f"📁 Αρχείο: {filename}")
        print("⏳ Κάνε την κίνηση σου... (Πατάς Ctrl+C για τερματισμό)\n")

        with open(filename, "wb") as f:
            while True:
                waiting = ser.in_waiting
                if waiting > 0:
                    chunk = ser.read(waiting)
                    f.write(chunk)
                    f.flush()  # ✅ FIX: Ασφάλεια σε περίπτωση Blue Screen / Crash
                    bytes_written += len(chunk)
                    
                    # ✅ FIX: Ζωντανή μέτρηση χωρίς να μπλοκάρει (overwrite στην ίδια γραμμή)
                    print(f"\r💾 Καταγράφηκαν: {bytes_written / 1024:.1f} KB", end="", flush=True)
                else:
                    # ✅ FIX: Αποφυγή 100% CPU Usage (Busy-Wait)
                    time.sleep(0.001) 
                    
    except serial.SerialException as e:
        print(f"\n\n❌ Σφάλμα Σειριακής: Μήπως η {PORT} χρησιμοποιείται από το idf.py monitor;")
    except KeyboardInterrupt:
        print(f"\n\n✅ ΤΕΛΟΣ ΚΑΤΑΓΡΑΦΗΣ!")
        print(f"📊 Αποθηκεύτηκαν συνολικά {bytes_written / 1024:.1f} KB στο αρχείο: {filename}")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()

if __name__ == "__main__":
    main()