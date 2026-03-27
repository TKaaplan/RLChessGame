"""
Interaktif training launcher.

Calistir:
    python run_train.py

Yapacaklari:
  1. Mevcut checkpoint'leri listeler
  2. Hangisinden devam etmek istedigini sorar
  3. Hangi modu kullanmak istedigini sorar
  4. Egitimi baslatir
"""

import os
import sys
import glob
import subprocess
import webbrowser
import time

# train.py ile ayni klasorde olmali
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

CHECKPOINT_DIR = os.path.join(BASE, "checkpoints")
ROOT_ZIPS      = glob.glob(os.path.join(BASE, "*.zip"))
TB_LOG_DIR     = os.path.join(BASE, "tb_logs")
TB_PORT        = 6006


# ── TensorBoard ───────────────────────────────────────────────────────────────

def _find_free_port(start=6006, end=6020):
    """start-end arasinda bos bir port bulur."""
    import socket
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", port)) != 0:
                return port   # baglanamadi → bos
    return start  # hepsi doluysa varsayilana don


def _start_tensorboard():
    """TensorBoard'u arka planda baslatir; zaten calısıyorsa yeniden baslatmaz."""
    import socket

    # Once 6006-6020 araliginda calisan bir TB var mi?
    for port in range(6006, 6021):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", port)) == 0:
                print(f"\n  TensorBoard zaten calisiyor → http://localhost:{port}")
                webbrowser.open(f"http://localhost:{port}")
                return None   # yeni proses baslatma

    port = _find_free_port()
    print(f"\n  TensorBoard baslatiliyor → http://localhost:{port}")
    proc = subprocess.Popen(
        [sys.executable, "-m", "tensorboard.main",
         "--logdir", TB_LOG_DIR, "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(3)  # sunucunun ayaga kalkmasini bekle
    webbrowser.open(f"http://localhost:{port}")
    print(f"  TensorBoard acildi (PID {proc.pid}). Tarayici otomatik acilmadiysa:")
    print(f"  http://localhost:{port}\n")
    return proc


# ── Yardimcilar ───────────────────────────────────────────────────────────────

def _list_checkpoints():
    """checkpoints/ klasoru + kok dizindeki tum .zip dosyalari."""
    ckpt_zips = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "*.zip")))
    all_zips  = ckpt_zips + sorted(ROOT_ZIPS)
    # Tekrar edenleri kaldir, sirali tut
    seen, result = set(), []
    for p in all_zips:
        key = os.path.abspath(p)
        if key not in seen:
            seen.add(key)
            result.append(p)
    return result


def _pick(prompt, options, labels=None):
    """
    Numarali secim menusu.  Secilen index'i dondurur.
    labels verilmezse options'in string gosterimi kullanilir.
    """
    if labels is None:
        labels = [str(o) for o in options]
    print()
    for i, lbl in enumerate(labels):
        print(f"  [{i+1}] {lbl}")
    print()
    while True:
        raw = input(f"{prompt} (1-{len(options)}): ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1
        print(f"  Gecersiz giris. 1 ile {len(options)} arasinda bir sayi gir.")


def _sep(title=""):
    w = 60
    if title:
        pad  = (w - len(title) - 2) // 2
        print("\n" + "=" * pad + f" {title} " + "=" * (w - pad - len(title) - 2))
    else:
        print("\n" + "=" * w)


# ── Mod tanimlari ─────────────────────────────────────────────────────────────

MODES = [
    {
        "key":   "stable",
        "label": "Stable Fine-Tune  (onerilen) — dusuk LR, depth 2→3→5, 6M adim",
        "desc":  "Policy collapse'i onler. improved_sfd3 sonrasi icin ideal.",
    },
    {
        "key":   "improve",
        "label": "Improve           — depth 2→3, 4M adim (eski pipeline)",
        "desc":  "Eski improve() fonksiyonu. Yuksek LR, hizli ama dengesiz olabilir.",
    },
    {
        "key":   "resume_custom",
        "label": "Custom Resume     — tek asama, parametreleri kendin gir",
        "desc":  "Tek bir depth/lr/adim kombinasyonu denemek istiyorsan.",
    },
]


def _run_stable(ckpt):
    from train import stable_finetune
    stable_finetune(ckpt)


def _run_improve(ckpt):
    from train import improve
    improve(ckpt)


def _run_custom(ckpt):
    from train import resume

    _sep("Custom Resume Parametreleri")

    def _ask_int(prompt, default):
        raw = input(f"  {prompt} [{default}]: ").strip()
        return int(raw) if raw.isdigit() else default

    def _ask_float(prompt, default):
        try:
            raw = input(f"  {prompt} [{default}]: ").strip()
            return float(raw) if raw else default
        except ValueError:
            return default

    depth  = _ask_int  ("Stockfish depth (1/2/3/5)",              2)
    steps  = _ask_int  ("Ek adim sayisi (ornek: 1000000)",  1_000_000)
    lr     = _ask_float("Learning rate  (ornek: 0.00002)",       2e-5)
    ent    = _ask_float("Entropy coef   (ornek: 0.008)",         0.008)
    clip   = _ask_float("Clip range     (ornek: 0.10)",          0.10)
    prefix = input(     "  Kayit prefix  [custom_run]: ").strip() or "custom_run"

    resume(
        checkpoint_path   = ckpt,
        extra_timesteps   = steps,
        vs_stockfish      = True,
        sf_opponent_depth = depth,
        lr                = lr,
        ent_coef          = ent,
        clip_range        = clip,
        save_prefix       = prefix,
    )


# ── Ana akis ──────────────────────────────────────────────────────────────────

def main():
    _sep("Chess RL Training Launcher")

    # 1. Checkpoint sec
    checkpoints = _list_checkpoints()
    if not checkpoints:
        print("HATA: Hic checkpoint bulunamadi!")
        print(f"  Aranan klasor: {CHECKPOINT_DIR}")
        sys.exit(1)

    labels = [os.path.relpath(p, BASE) for p in checkpoints]
    print("\nHangi checkpoint'ten devam etmek istiyorsun?")
    idx_ckpt = _pick("Checkpoint sec", checkpoints, labels)
    chosen_ckpt = checkpoints[idx_ckpt]

    # 2. Parametreler
    _run_custom(chosen_ckpt)


if __name__ == "__main__":
    main()
