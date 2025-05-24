print("[DEBUG] Importing hivemind.DHT...")
try:
    from hivemind import DHT
    print("[DEBUG] Successfully imported DHT.")
except Exception as e:
    print(f"[ERROR] Failed to import DHT: {e}")
    raise
import time

if __name__ == "__main__":
    print("[TEST] Creating DHT...")
    try:
        dht = DHT(start=True)
        print("[TEST] DHT started:", dht.is_alive())
    except Exception as e:
        print(f"[ERROR] Exception during DHT creation/start: {e}")
        raise
    time.sleep(2)
    print("[TEST] Shutting down DHT...")
    try:
        dht.shutdown()
        print("[TEST] Shutdown successful.")
    except Exception as e:
        print(f"[ERROR] Exception during DHT shutdown: {e}")
        raise
    print("[TEST] Done.")
