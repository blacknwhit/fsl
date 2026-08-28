import os
import urllib.request

import torch

BASE = "/data/xiangyuyue/ULLM-zf/fsl-20260209/pretrained/ijepa"
FINAL = os.path.join(BASE, "IN1K-vit.h.16-448px-300e.pth.tar")
TMP = FINAL + ".redownload"
URL = "https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.16-448px-300e.pth.tar"
EXPECTED = 10367908521
CHUNK = 8 * 1024 * 1024


def main() -> int:
    os.makedirs(BASE, exist_ok=True)
    for path in (FINAL, TMP):
        if os.path.exists(path):
            os.remove(path)
    print("[start] removed old files")

    last_err = None
    for attempt in range(1, 4):
        try:
            if os.path.exists(TMP):
                os.remove(TMP)
            print(f"[attempt {attempt}] downloading {URL}")
            with urllib.request.urlopen(URL, timeout=60) as resp, open(TMP, "wb") as out:
                total = 0
                while True:
                    data = resp.read(CHUNK)
                    if not data:
                        break
                    out.write(data)
                    total += len(data)
                    if total % (512 * 1024 * 1024) < CHUNK:
                        pct = total * 100.0 / EXPECTED
                        print(f"[attempt {attempt}] progress: {total} bytes ({pct:.2f}%)")

            size = os.path.getsize(TMP)
            print(f"[attempt {attempt}] downloaded size={size}")
            if size != EXPECTED:
                raise RuntimeError(f"size mismatch: expected {EXPECTED}, got {size}")

            print(f"[attempt {attempt}] validating torch.load")
            obj = torch.load(TMP, map_location="cpu")
            print(f"[attempt {attempt}] torch.load ok: {type(obj).__name__}")
            if isinstance(obj, dict):
                print(f"[attempt {attempt}] dict keys={len(obj)}")

            os.replace(TMP, FINAL)
            print(f"[success] ready: {FINAL}")
            return 0
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"[attempt {attempt}] failed: {exc}")

    print(f"[error] failed after retries: {last_err}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
