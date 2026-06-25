"""AirSketch launcher — a small GUI front-end that turns the app's CLI flags into
a friendly feature-flags screen, then starts the app the way you chose.

Three modes (one file, so the SAME PyInstaller exe is both launcher and runner):

    launcher.exe                  -> show the GUI (default)
    launcher.exe --launch-app ... -> run the AirSketch app with these flags
    launcher.exe --check          -> import-only smoke test (used to verify a build)

The GUI builds a flag list (see ``build_args``) and spawns the app as a SUBPROCESS
(``sys.executable --launch-app <flags>``) so it works identically in dev and when
frozen, and so the app's own ``os._exit(0)`` cleanup never takes the launcher down.
"""
from __future__ import annotations

import os
import subprocess
import sys


# ---------------------------------------------------------------------------
# Pure flag builder (unit-tested; no GUI needed)
# ---------------------------------------------------------------------------
# Defaults that match airsketch.config.AppConfig — used to decide when a chosen
# value is non-default and therefore worth emitting as an explicit flag.
_DEFAULT_HAND_DEVICE = "AUTO"
_DEFAULT_CNN_DEVICE = "AUTO"
_DEFAULT_VLM_DEVICE = "AUTO"
_DEFAULT_OCR_DEVICE = "CPU"
_DEFAULT_LLM_DEVICE = "CPU"
_DEFAULT_VOICE_DEVICE = "CPU"
_DEFAULT_SPEAKER_DEVICE = "CPU"


def build_args(opts: dict) -> list[str]:
    """Translate GUI option values into a list of `airsketch.main` CLI flags.

    ``opts`` keys (all optional; sensible defaults applied):
        mode           'classroom' | 'notebook'      (default 'classroom')
        theme          'geometry' | 'objects' | 'mixed'
        camera         int
        mirror         bool
        rotate180      bool   (camera 180-rotation; True = default-on for VDI)
        hand_backend   'openvino' | 'mediapipe'
        hand_device    'AUTO' | 'CPU' | 'GPU' | 'NPU'
        hand_debug     bool
        no_cnn         bool
        cnn_device     'AUTO' | 'CPU' | 'GPU' | 'NPU'
        voice          bool
        teacher_voice  bool
        board          bool
        understand     bool
        ocr_device / llm_device   device strings
        vlm            bool   (notebook only)
        vlm_device     device string
        snap           bool
    """
    g = opts.get
    mode = g("mode", "classroom")
    args: list[str] = []

    if mode == "classroom":
        args.append("--classroom")
        args += ["--theme", g("theme", "geometry")]

    # Language (en default). Affects Whisper voice, command keywords, board LLM.
    lang = g("language", "en")
    if lang and lang != "en":
        args += ["--lang", lang]

    # Camera / orientation
    camera = int(g("camera", 0) or 0)
    if camera != 0:
        args += ["--camera", str(camera)]
    if g("mirror", False):
        args.append("--mirror")
    if not g("rotate180", True):
        args.append("--no-rotate")

    # Hand tracking
    hand_backend = g("hand_backend", "openvino")
    args += ["--hand-backend", hand_backend]
    if hand_backend == "openvino":
        hd = g("hand_device", _DEFAULT_HAND_DEVICE)
        if hd and hd != _DEFAULT_HAND_DEVICE:
            args += ["--hand-device", hd]
        if g("hand_debug", False):
            args.append("--hand-debug")

    # CNN
    if g("no_cnn", False):
        args.append("--no-cnn")
    else:
        cd = g("cnn_device", _DEFAULT_CNN_DEVICE)
        if cd and cd != _DEFAULT_CNN_DEVICE:
            args += ["--cnn-device", cd]

    # Voice / teacher voice
    if g("voice", False):
        args.append("--voice")
    if g("teacher_voice", False):
        args.append("--teacher-voice")
    if (g("voice", False) or g("teacher_voice", False)):
        mt = g("mic_timeout", 6)
        if mt and float(mt) != 6.0:
            args += ["--mic-timeout", str(mt)]
        wm = g("whisper_model", "base")
        if wm and wm != "base":
            args += ["--whisper-model", wm]
        vcd = g("voice_device", _DEFAULT_VOICE_DEVICE)
        if vcd and vcd != _DEFAULT_VOICE_DEVICE:
            args += ["--voice-device", vcd]
    if g("teacher_voice", False):
        spd = g("speaker_device", _DEFAULT_SPEAKER_DEVICE)
        if spd and spd != _DEFAULT_SPEAKER_DEVICE:
            args += ["--speaker-device", spd]

    # Board capture + understanding LLM
    if g("board", False):
        args.append("--board")
    if g("understand", False):
        args.append("--understand")
    if g("board", False) or g("understand", False):
        od = g("ocr_device", _DEFAULT_OCR_DEVICE)
        if od and od != _DEFAULT_OCR_DEVICE:
            args += ["--ocr-device", od]
    if g("understand", False):
        ld = g("llm_device", _DEFAULT_LLM_DEVICE)
        if ld and ld != _DEFAULT_LLM_DEVICE:
            args += ["--llm-device", ld]

    # Notebook-only: VLM titling
    if mode == "notebook" and g("vlm", False):
        args.append("--vlm")
        vd = g("vlm_device", _DEFAULT_VLM_DEVICE)
        if vd and vd != _DEFAULT_VLM_DEVICE:
            args += ["--vlm-device", vd]

    # Snap-to-shape
    if g("snap", False):
        args.append("--snap")

    return args


def preview_command(opts: dict) -> str:
    """Human-readable preview of how the app will be launched."""
    return "airsketch " + " ".join(build_args(opts))


# ---------------------------------------------------------------------------
# Run-mode helpers
# ---------------------------------------------------------------------------
def base_dir() -> str:
    """The exe's / script's own folder."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def resolve_root() -> str:
    """The directory the app should run from — the nearest ancestor (incl. self)
    that contains a ``models/`` folder. Lets the exe work whether it sits in the
    project root OR in ``dist/`` (PyInstaller's output dir), without moving it.
    Falls back to the exe's own folder if no ``models/`` is found.
    """
    d = base_dir()
    for _ in range(5):
        if os.path.isdir(os.path.join(d, "models")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return base_dir()


def _ensure_importable(base: str) -> None:
    src = os.path.join(base, "src")
    if os.path.isdir(src) and src not in sys.path:
        sys.path.insert(0, src)


def _ensure_std_streams(root: str) -> None:
    """A PyInstaller --windowed (no-console) exe has sys.stdout/stderr == None.
    The app prints freely and even calls sys.stdout.flush() on shutdown, which
    raises AttributeError on None. Redirect to a line-buffered log file (fallback
    os.devnull) so prints are captured and flush() works.
    """
    if sys.stdout is not None and sys.stderr is not None:
        return
    stream = None
    try:
        stream = open(os.path.join(root, "airsketch_run.log"), "w",
                      encoding="utf-8", buffering=1)
    except Exception:
        try:
            stream = open(os.devnull, "w")
        except Exception:
            stream = None
    if stream is not None:
        if sys.stdout is None:
            sys.stdout = stream
        if sys.stderr is None:
            sys.stderr = stream


def run_app(flags: list[str]) -> None:
    """In-process dispatch into airsketch.main with the given flags."""
    root = resolve_root()
    _ensure_importable(root)
    _ensure_importable(base_dir())
    _ensure_std_streams(root)
    try:
        os.chdir(root)   # so relative models/ and outputs/ paths resolve
    except OSError:
        pass
    sys.argv = ["airsketch"] + list(flags)
    from airsketch.main import main as app_main
    app_main()


def spawn_app(flags: list[str]) -> subprocess.Popen:
    """Start the app as an independent subprocess (re-invoking this exe/script)."""
    root = resolve_root()
    if getattr(sys, "frozen", False):
        cmd = [sys.executable, "--launch-app", *flags]
    else:
        cmd = [sys.executable, os.path.abspath(__file__), "--launch-app", *flags]
    return subprocess.Popen(cmd, cwd=root)


def do_check() -> int:
    """Import-only smoke test — verifies a (frozen) build bundled everything."""
    ok = True
    for mod in ("airsketch.main", "airsketch.hand_tracker_ov", "cv2", "numpy", "openvino"):
        try:
            _ensure_importable(base_dir())
            __import__(mod)
            print(f"  import {mod}: OK")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"  import {mod}: FAILED ({type(e).__name__}: {e})")
    print("CHECK OK" if ok else "CHECK FAILED")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
def launch_gui() -> None:
    import tkinter as tk
    from tkinter import ttk

    DEVICES = ("AUTO", "CPU", "GPU", "NPU")

    root = tk.Tk()
    root.title("AirSketch Launcher")
    root.minsize(560, 0)
    try:
        style = ttk.Style()
        style.theme_use("clam")
    except tk.TclError:
        pass

    # ---- state vars ----
    v_mode = tk.StringVar(value="classroom")
    v_theme = tk.StringVar(value="geometry")
    v_language = tk.StringVar(value="en")
    v_camera = tk.IntVar(value=0)
    v_mirror = tk.BooleanVar(value=False)
    v_rotate = tk.BooleanVar(value=True)   # VDI camera is upside-down by default
    v_hand_backend = tk.StringVar(value="openvino")
    v_hand_device = tk.StringVar(value="AUTO")
    v_hand_debug = tk.BooleanVar(value=False)
    v_no_cnn = tk.BooleanVar(value=False)
    v_cnn_device = tk.StringVar(value="AUTO")
    v_voice = tk.BooleanVar(value=False)
    v_teacher = tk.BooleanVar(value=False)
    v_board = tk.BooleanVar(value=False)
    v_understand = tk.BooleanVar(value=False)
    v_vlm = tk.BooleanVar(value=False)
    v_vlm_device = tk.StringVar(value="AUTO")
    v_ocr_device = tk.StringVar(value="CPU")
    v_llm_device = tk.StringVar(value="CPU")
    v_voice_device = tk.StringVar(value="CPU")
    v_speaker_device = tk.StringVar(value="CPU")
    v_snap = tk.BooleanVar(value=False)
    v_mic_timeout = tk.IntVar(value=6)
    v_whisper = tk.StringVar(value="base")

    def collect() -> dict:
        return dict(
            mode=v_mode.get(), theme=v_theme.get(), language=v_language.get(),
            camera=v_camera.get(),
            mirror=v_mirror.get(), rotate180=v_rotate.get(),
            hand_backend=v_hand_backend.get(), hand_device=v_hand_device.get(),
            hand_debug=v_hand_debug.get(), no_cnn=v_no_cnn.get(),
            cnn_device=v_cnn_device.get(), voice=v_voice.get(),
            teacher_voice=v_teacher.get(), board=v_board.get(),
            understand=v_understand.get(), vlm=v_vlm.get(),
            vlm_device=v_vlm_device.get(), ocr_device=v_ocr_device.get(),
            llm_device=v_llm_device.get(), voice_device=v_voice_device.get(),
            speaker_device=v_speaker_device.get(), snap=v_snap.get(),
            mic_timeout=v_mic_timeout.get(), whisper_model=v_whisper.get(),
        )

    pad = dict(padx=8, pady=4)

    # ---- header ----
    hdr = ttk.Frame(root, padding=(12, 10))
    hdr.grid(row=0, column=0, columnspan=2, sticky="ew")
    ttk.Label(hdr, text="AirSketch", font=("Segoe UI", 18, "bold")).pack(anchor="w")
    ttk.Label(hdr, text="Choose how to launch the app, then click Launch.",
              foreground="#555").pack(anchor="w")

    body = ttk.Frame(root, padding=(12, 4))
    body.grid(row=1, column=0, columnspan=2, sticky="nsew")

    # ---- Mode ----
    f_mode = ttk.LabelFrame(body, text="Mode", padding=8)
    f_mode.grid(row=0, column=0, sticky="nsew", **pad)
    ttk.Radiobutton(f_mode, text="Classroom challenge game", value="classroom",
                    variable=v_mode).grid(row=0, column=0, sticky="w", columnspan=2)
    ttk.Radiobutton(f_mode, text="Notebook (freehand drawing)", value="notebook",
                    variable=v_mode).grid(row=1, column=0, sticky="w", columnspan=2)
    ttk.Label(f_mode, text="Theme:").grid(row=2, column=0, sticky="w", pady=(6, 0))
    cmb_theme = ttk.Combobox(f_mode, textvariable=v_theme, width=12, state="readonly",
                             values=("geometry", "objects", "mixed"))
    cmb_theme.grid(row=2, column=1, sticky="w", pady=(6, 0))
    ttk.Label(f_mode, text="Language:").grid(row=3, column=0, sticky="w", pady=(6, 0))
    ttk.Combobox(f_mode, textvariable=v_language, width=12, state="readonly",
                 values=("en", "he")).grid(row=3, column=1, sticky="w", pady=(6, 0))

    # ---- Camera ----
    f_cam = ttk.LabelFrame(body, text="Camera", padding=8)
    f_cam.grid(row=0, column=1, sticky="nsew", **pad)
    ttk.Label(f_cam, text="Index:").grid(row=0, column=0, sticky="w")
    ttk.Spinbox(f_cam, from_=0, to=8, width=5, textvariable=v_camera).grid(
        row=0, column=1, sticky="w")
    ttk.Checkbutton(f_cam, text="Mirror (selfie view)", variable=v_mirror).grid(
        row=1, column=0, columnspan=2, sticky="w")
    ttk.Checkbutton(f_cam, text="Rotate 180° (VDI camera is upside-down)",
                    variable=v_rotate).grid(row=2, column=0, columnspan=2, sticky="w")

    # ---- AI backends / devices ----
    f_dev = ttk.LabelFrame(body, text="AI backends & devices (OpenVINO)", padding=8)
    f_dev.grid(row=1, column=0, sticky="nsew", **pad)
    ttk.Label(f_dev, text="Hand tracking:").grid(row=0, column=0, sticky="w")
    ttk.Combobox(f_dev, textvariable=v_hand_backend, width=11, state="readonly",
                 values=("openvino", "mediapipe")).grid(row=0, column=1, sticky="w")
    ttk.Label(f_dev, text="Hand device:").grid(row=1, column=0, sticky="w")
    cmb_hd = ttk.Combobox(f_dev, textvariable=v_hand_device, width=11, state="readonly",
                          values=DEVICES)
    cmb_hd.grid(row=1, column=1, sticky="w")
    ttk.Checkbutton(f_dev, text="Hand-tracking debug overlay",
                    variable=v_hand_debug).grid(row=2, column=0, columnspan=2, sticky="w")
    ttk.Label(f_dev, text="CNN device:").grid(row=3, column=0, sticky="w")
    ttk.Combobox(f_dev, textvariable=v_cnn_device, width=11, state="readonly",
                 values=DEVICES).grid(row=3, column=1, sticky="w")
    # remaining per-model devices (apply when the matching feature/mode is on)
    for _r, (_lbl, _var) in enumerate([
        ("Voice device:", v_voice_device),
        ("Speaker device:", v_speaker_device),
        ("OCR device:", v_ocr_device),
        ("LLM device:", v_llm_device),
        ("VLM device:", v_vlm_device),
    ], start=4):
        ttk.Label(f_dev, text=_lbl).grid(row=_r, column=0, sticky="w")
        ttk.Combobox(f_dev, textvariable=_var, width=11, state="readonly",
                     values=DEVICES).grid(row=_r, column=1, sticky="w")
    ttk.Label(f_dev, text="(device applies only if that feature is enabled)",
              foreground="#777").grid(row=9, column=0, columnspan=2, sticky="w", pady=(4, 0))

    # ---- Features ----
    f_feat = ttk.LabelFrame(body, text="Features", padding=8)
    f_feat.grid(row=1, column=1, sticky="nsew", **pad)
    ttk.Checkbutton(f_feat, text="Voice commands (Whisper)",
                    variable=v_voice).grid(row=0, column=0, sticky="w")
    ttk.Checkbutton(f_feat, text="Teacher voice / dictation",
                    variable=v_teacher).grid(row=1, column=0, sticky="w")
    ttk.Checkbutton(f_feat, text="Whiteboard capture (PP-OCR)",
                    variable=v_board).grid(row=2, column=0, sticky="w")
    ttk.Checkbutton(f_feat, text="Understand board (LLM summary, ~1.8 GB)",
                    variable=v_understand).grid(row=3, column=0, sticky="w")
    cb_vlm = ttk.Checkbutton(f_feat, text="Qwen2-VL titling (notebook, ~1.7 GB)",
                             variable=v_vlm)
    cb_vlm.grid(row=4, column=0, sticky="w")
    ttk.Checkbutton(f_feat, text="Snap-to-shape on pen-up",
                    variable=v_snap).grid(row=5, column=0, sticky="w")
    ttk.Checkbutton(f_feat, text="Disable CNN classifier",
                    variable=v_no_cnn).grid(row=6, column=0, sticky="w")
    mt_row = ttk.Frame(f_feat)
    mt_row.grid(row=7, column=0, sticky="w", pady=(4, 0))
    ttk.Label(mt_row, text="Mic open timeout (s):").pack(side="left")
    ttk.Spinbox(mt_row, from_=3, to=30, width=4, textvariable=v_mic_timeout).pack(side="left")
    wm_row = ttk.Frame(f_feat)
    wm_row.grid(row=8, column=0, sticky="w", pady=(2, 0))
    ttk.Label(wm_row, text="Whisper model:").pack(side="left")
    ttk.Combobox(wm_row, textvariable=v_whisper, width=7, state="readonly",
                 values=("base", "small")).pack(side="left")

    # ---- command preview ----
    f_prev = ttk.LabelFrame(body, text="Launch command", padding=8)
    f_prev.grid(row=2, column=0, columnspan=2, sticky="ew", **pad)
    v_prev = tk.StringVar()
    ent = ttk.Entry(f_prev, textvariable=v_prev, font=("Consolas", 9))
    ent.pack(fill="x")

    def refresh(*_):
        opts = collect()
        v_prev.set(preview_command(opts))
        # context-sensitive enable/disable
        is_class = v_mode.get() == "classroom"
        cmb_theme.configure(state="readonly" if is_class else "disabled")
        cb_vlm.configure(state="disabled" if is_class else "normal")
        cmb_hd.configure(state="readonly" if v_hand_backend.get() == "openvino"
                         else "disabled")

    for var in (v_mode, v_theme, v_language, v_camera, v_mirror, v_rotate, v_hand_backend,
                v_hand_device, v_hand_debug, v_no_cnn, v_cnn_device, v_voice,
                v_teacher, v_board, v_understand, v_vlm, v_vlm_device, v_ocr_device,
                v_llm_device, v_voice_device, v_speaker_device, v_snap,
                v_mic_timeout, v_whisper):
        var.trace_add("write", refresh)
    refresh()

    # ---- buttons ----
    f_btn = ttk.Frame(root, padding=(12, 10))
    f_btn.grid(row=2, column=0, columnspan=2, sticky="ew")
    f_btn.columnconfigure(0, weight=1)

    status = ttk.Label(f_btn, text="", foreground="#176")
    status.grid(row=0, column=0, sticky="w")

    def on_launch():
        flags = build_args(collect())
        try:
            spawn_app(flags)
        except Exception as e:  # noqa: BLE001
            status.configure(text=f"Launch failed: {type(e).__name__}: {e}",
                             foreground="#a00")
            return
        root.destroy()   # close on launch (per chosen behavior)

    ttk.Button(f_btn, text="Quit", command=root.destroy).grid(row=0, column=1, padx=6)
    launch = ttk.Button(f_btn, text="Launch ▶", command=on_launch)
    launch.grid(row=0, column=2)
    launch.focus_set()

    root.mainloop()


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "--launch-app":
        run_app(argv[1:])
        return 0
    if argv and argv[0] == "--check":
        return do_check()
    launch_gui()
    return 0


if __name__ == "__main__":
    sys.exit(main())
