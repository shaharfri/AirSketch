import traceback
import cv2, numpy as np, sounddevice as sd
import openvino_genai as ov_genai

print("loading whisper..."); 
pipe = ov_genai.WhisperPipeline("models/whisper-base-ov", "CPU")
print("whisper loaded OK")

cap = cv2.VideoCapture(0)
print("camera opened:", cap.isOpened())
print(">>> Window open. Click it, press V to record 2s, Q to quit.")
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("camera read FAILED"); break
        cv2.imshow("voice test", frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord('q'), 27):
            break
        elif k in (ord('v'), ord('V')):
            print("V pressed -> recording 2s (speak now)")
            buf = sd.rec(int(2*16000), samplerate=16000, channels=1, dtype='float32')
            sd.wait()
            print("recorded peak:", float(np.abs(buf).max()), "-> transcribing...")
            out = pipe.generate(buf.flatten())
            print("HEARD:", out.texts[0] if out.texts else "(none)")
except Exception:
    traceback.print_exc()
finally:
    cap.release(); cv2.destroyAllWindows()
    print(">>> exited cleanly")