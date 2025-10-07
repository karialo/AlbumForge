#!/usr/bin/env python3
# AlbumForge — Monolith v0.6.1
# Single-file tool for album workflows:
# - create/move/art/video/status/lyrics/preview + doctor (deps check/install)
# - Inline modules: term_images, render, artmatch

from __future__ import annotations
import sys as _sys
_sys.modules.setdefault("AlbumForge", _sys.modules[__name__])
import argparse, json, os, random, re, shutil, subprocess, sys, shlex, tempfile, base64, time
from pathlib import Path
from datetime import datetime, timezone
from difflib import SequenceMatcher

APP = "AlbumForge"
APP_TITLE = "AlbumForge"
VERSION = "0.6.1"


# =========================
# Embedded modules (sources)
# =========================

_EMBED_TERM_IMAGES = r"""
# term_images (embedded)
from __future__ import annotations
import os, shutil, subprocess, base64, sys, os.path as op

def have(cmd:str)->bool: 
    return shutil.which(cmd) is not None

def is_wezterm()->bool:
    return os.environ.get("TERM_PROGRAM") == "WezTerm" or bool(
        os.environ.get("WEZTERM_EXECUTABLE") or os.environ.get("WT_SESSION")
    )

def _term_wh() -> tuple[int, int]:
    try:
        sz = shutil.get_terminal_size((80, 24))
        return sz.columns, sz.lines
    except Exception:
        return 80, 24

def preview(path: str) -> bool:
    # Try (in order): WezTerm imgcat, Kitty icat (auto-size), iTerm2/WezTerm OSC 1337, SIXEL, chafa, viu.
    path = op.abspath(path)

    # 0) WezTerm imgcat (most reliable on WezTerm)
    if is_wezterm() and have("wezterm"):
        try:
            subprocess.run(["wezterm", "imgcat", path], check=True, timeout=6)
            return True
        except Exception:
            pass

    # 1) Kitty icat — only if actually inside Kitty, auto-size to terminal
    if have("kitty") and (os.environ.get("TERM", "").startswith("xterm-kitty")
                          or os.environ.get("KITTY_INSTALLATION")):
        try:
            cols, rows = _term_wh()
            subprocess.run(["kitty", "+kitten", "icat", "--place", f"{cols}x{rows}@0x0", path],
                           check=True, timeout=6)
            return True
        except Exception:
            pass

    # 2) OSC 1337 (iTerm2 / WezTerm)
    if is_wezterm() or os.environ.get("TERM_PROGRAM") in ("iTerm.app", "iTerm2"):
        try:
            with open(path, "rb") as f:
                raw = f.read()
            b64 = base64.b64encode(raw).decode()
            # size must be original byte length, not base64 length
            sys.stdout.write(f"\033]1337;File=name={op.basename(path)};inline=1;size={len(raw)}:{b64}\a\n")
            sys.stdout.flush()
            return True
        except Exception:
            pass

    # 3) SIXEL
    if have("img2sixel"):
        try:
            subprocess.run(["img2sixel", path], check=True, timeout=6)
            return True
        except Exception:
            pass

    # 4) Fallbacks — text previews work everywhere
    if have("chafa"):
        subprocess.run(["chafa", path])
        return True
    if have("viu"):
        subprocess.run(["viu", path])
        return True

    return False

"""

_EMBED_RENDER = r"""
# render (embedded) — static video renderer (art + audio → mp4 in Video/)
from __future__ import annotations
import shutil, subprocess, re, json, os, sys, time
from pathlib import Path

AUDIO_EXTS = {".mp3", ".m4a", ".flac", ".wav", ".ogg", ".opus"}
IMG_EXTS   = {".png", ".jpg", ".jpeg", ".webp"}

def have(cmd:str) -> bool:
    return shutil.which(cmd) is not None

def _stem(p:Path) -> str:
    return re.sub(r"\.[^.]+$","",p.name)

def _ffprobe_duration(audio:Path) -> float:
    cmd = ["ffprobe","-v","error","-show_entries","format=duration","-of","json",str(audio)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0: return 0.0
    try:
        return float(json.loads(p.stdout)["format"]["duration"])
    except Exception:
        return 0.0

def _fmt_time(secs:float) -> str:
    m, s = divmod(int(max(0, secs)), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

def _term_cols(default:int=80) -> int:
    try:
        return shutil.get_terminal_size((default, 20)).columns
    except Exception:
        return default

def _shorten(text:str, maxlen:int) -> str:
    if len(text) <= maxlen: return text
    if maxlen <= 1: return text[:maxlen]
    head = max(1, (maxlen - 1) // 2)
    tail = max(0, maxlen - head - 1)
    return text[:head] + "…" + (text[-tail:] if tail else "")

def _progress_line(title:str, audio:Path, out:Path, img:Path, ffmpeg_args:list[str], say=print) -> int:
    # Run ffmpeg with -progress pipe:1, render single-line adaptive bar (TTY only).
    duration = _ffprobe_duration(audio)
    proc = subprocess.Popen(
        ["ffmpeg","-hide_banner","-loglevel","quiet","-y", *ffmpeg_args, "-progress","pipe:1", str(out)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1
    )
    last: dict[str, str] = {}
    is_tty = os.isatty(1)

    def render():
        cols = _term_cols()
        try:
            tcur = float(last.get("out_time_ms","0"))/1_000_000.0
        except ValueError:
            tcur = 0.0
        pct  = (tcur/duration*100.0) if duration > 0 else 0.0
        if   pct < 0:   pct = 0.0
        elif pct > 100: pct = 100.0
        speed = last.get("speed","?")

        base_title = f"🎬 {title} "
        suffix = f" {pct:6.2f}% {_fmt_time(tcur)}/{_fmt_time(duration)} speed {speed}"
        min_bar = 8
        max_title = max(10, cols - (len(suffix) + min_bar + 6))
        title_sh = _shorten(base_title, max_title)
        barroom = cols - (len(title_sh) + len(suffix) + 3)
        barlen = max(min_bar, min(60, barroom))
        filled = int(round((pct/100.0)*barlen)) if barlen > 0 else 0
        bar = "[" + ("█"*filled + "░"*(barlen - filled)) + "]"
        line = f"{title_sh}{bar}{suffix}"

        if is_tty:
            sys.stdout.write("\r" + line[:cols-1] + "\x1b[K")
            sys.stdout.flush()
        else:
            print(line[:cols-1], flush=True)

    try:
        for raw in proc.stdout:
            line = raw.strip()
            if not line: 
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                last[k] = v
            if line.startswith("progress="):
                render()
        proc.wait()
    finally:
        if is_tty:
            print()  # finish the line
    return proc.returncode

def run(album_root:Path, *, force:bool=False, say=print) -> int:
    # tiny settle in case artwork just landed (FS race guards)
    time.sleep(0.2)

    if not have("ffmpeg"):
        raise SystemExit("❌ ffmpeg not found in PATH.")
    audio_dir = album_root/"Audio"
    art_dir   = album_root/"Artwork"
    out_dir   = album_root/"Video"
    out_dir.mkdir(parents=True, exist_ok=True)

    audios = [p for p in audio_dir.glob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
    arts   = [p for p in art_dir.glob("*")   if p.is_file() and p.suffix.lower() in IMG_EXTS]
    if not audios:
        say("❌ No audio files found."); return 1

    A = { _stem(a): a for a in audios }
    R = { _stem(i): i for i in arts }

    common = sorted(set(A) & set(R))
    missing_art = sorted(set(A) - set(R))
    for s in missing_art:
        say(f"⚠️  No artwork for {A[s].name}, skipping.")

    failures = 0
    for stem in common:
        a = A[stem]; img = R[stem]
        out = out_dir / f"{stem}.mp4"
        if out.exists() and not force:
            say(f"⏭️  Exists, skipping: {out.name}"); continue
        ffmpeg_args = [
            "-loop","1","-i",str(img),
            "-i",str(a),
            "-c:v","libx264","-tune","stillimage","-r","30",
            "-vf","scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
            "-c:a","aac","-b:a","192k",
            "-movflags","+faststart","-shortest"
        ]
        code = _progress_line(stem, a, out, img, ffmpeg_args, say)
        if code == 0:
            say(f"✅ Done: {out.relative_to(album_root)}")
        else:
            failures += 1
            say(f"❌ ffmpeg failed ({code}) for {stem}")
    if failures:
        say(f"⚠️  Render finished with {failures} failure(s).")
        return 2
    say("✨ All done.")
    return 0
"""

_EMBED_ARTMATCH = r'''
# artmatch (embedded) — deterministic, optimal image→track mapping
# Truth order: (A) leading NN lock, (B) global optimal filename match (Hungarian),
# (C) OCR fallback (opt-in). If strict=True, anything low-confidence goes to _UNMATCHED/.
from __future__ import annotations
import re, shutil, subprocess, math
from pathlib import Path
from difflib import SequenceMatcher

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
NUM_RX   = re.compile(r'^\s*(\d{1,2})\s*[-._ ]\s*', re.U)

def have(cmd:str)->bool: 
    return shutil.which(cmd) is not None

# ---------- normalization ----------
def _slug(s:str)->str:
    s = s.lower()
    s = re.sub(r'\(feat[^)]*\)', '', s)
    s = s.replace('&', ' and ')
    s = re.sub(r'[_\.\-]+', ' ', s)
    s = re.sub(r'[^a-z0-9 ]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    # common canon
    s = s.replace(' over clock ', ' overclock ')
    s = s.replace(' 404 ', ' four oh four ')
    s = s.replace(' consume exe', ' consume exe')
    return s

def _num(name:str) -> int | None:
    m = NUM_RX.match(name)      # only accept at START, 1–2 digits
    if not m: return None
    n = int(m.group(1))
    return n if 1 <= n <= 99 else None

def _title_from_filename(p:Path) -> str:
    return NUM_RX.sub('', p.stem)

def _sim(a:str, b:str) -> float:
    return SequenceMatcher(None, _slug(a), _slug(b)).ratio()

# ---------- OCR (optional fallback) ----------
def _ocr_text(img:Path, lang:str, psm:str|None, timeout:int=8) -> str:
    if not have("tesseract"): return ""
    cmd = ["tesseract", str(img), "stdout", "-l", lang]
    if psm: cmd += ["--psm", psm]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=timeout)
        out = re.sub(r'[^A-Za-z0-9 ._\-]', ' ', out)
        out = re.sub(r'\s+', ' ', out).strip()
        return out
    except Exception:
        return ""

# ---------- Hungarian (maximize) ----------
def _hungarian_max(cost: list[list[float]]) -> list[tuple[int,int]]:
    """Return list of (i,j) maximizing total cost; rectangular OK (pads with 0)."""
    n = max(len(cost), max((len(r) for r in cost), default=0))
    M = [[0.0]*n for _ in range(n)]
    mx = 0.0
    for i,row in enumerate(cost):
        for j,v in enumerate(row):
            M[i][j] = v
            if v > mx: mx = v
    W = [[mx - M[i][j] for j in range(n)] for i in range(n)]

    u = [0.0]*(n+1); v = [0.0]*(n+1); p = [0]*(n+1); way = [0]*(n+1)
    for i in range(1, n+1):
        p[0] = i; j0 = 0
        minv = [math.inf]*(n+1); used = [False]*(n+1)
        while True:
            used[j0] = True; i0 = p[j0]; delta = math.inf; j1 = 0
            for j in range(1, n+1):
                if used[j]: continue
                cur = W[i0-1][j-1] - u[i0] - v[j]
                if cur < minv[j]: minv[j] = cur; way[j] = j0
                if minv[j] < delta: delta = minv[j]; j1 = j
            for j in range(0, n+1):
                if used[j]: u[p[j]] += delta; v[j] -= delta
                else:       minv[j] -= delta
            j0 = j1
            if p[j0] == 0: break
        while True:
            j1 = way[j0]; p[j0] = p[j1]; j0 = j1
            if j0 == 0: break
    match = []
    for j in range(1, n+1):
        i = p[j]
        if i and i-1 < len(cost) and j-1 < len(cost[i-1]):
            match.append((i-1, j-1))
    return match

# ---------- Planner ----------
def plan_matches(images:list[Path], tracks:list[str], *, lang:str="eng", psm:str|None="6", use_ocr:bool=False,
                 filename_thresh:float=0.82):
    """
    Return:
      pairs: list of (img_idx, track_idx, score, method)  — accepted matches only
      leftover_images: indices not confidently matched
    """
    # A) number lock
    locked = []
    used_i, used_j = set(), set()
    track_nums = []
    NUM_RX = re.compile(r'^\s*(\d{1,2})\s*[-._ ]\s*', re.U)
    def _num(name:str) -> int | None:
        m = NUM_RX.match(name)
        if not m: return None
        n = int(m.group(1))
        return n if 1 <= n <= 99 else None

    for t in tracks:
        m = NUM_RX.match(t); track_nums.append(int(m.group(1)) if m else None)

    for i, img in enumerate(images):
        ni = _num(img.stem)
        if ni is None: continue
        for j, nj in enumerate(track_nums):
            if j in used_j: continue
            if nj == ni:
                locked.append((i,j,1.0,"number"))
                used_i.add(i); used_j.add(j)
                break

    # B) global optimal filename matching
    rem_i = [i for i in range(len(images)) if i not in used_i]
    rem_j = [j for j in range(len(tracks)) if j not in used_j]
    def _title_from_filename(p:Path) -> str:
        return NUM_RX.sub('', p.stem)

    def _slug(s:str)->str:
        s = s.lower()
        s = re.sub(r'\(feat[^)]*\)', '', s)
        s = s.replace('&', ' and ')
        s = re.sub(r'[_\.\-]+', ' ', s)
        s = re.sub(r'[^a-z0-9 ]+', ' ', s)
        s = re.sub(r'\s+', ' ', s).strip()
        s = s.replace(' over clock ', ' overclock ')
        s = s.replace(' 404 ', ' four oh four ')
        s = s.replace(' consume exe', ' consume exe')
        return s

    from difflib import SequenceMatcher
    def _sim(a:str, b:str) -> float:
        return SequenceMatcher(None, _slug(a), _slug(b)).ratio()

    if rem_i and rem_j:
        cost = []
        for i in rem_i:
            ti = _title_from_filename(images[i])
            row = []
            for j in rem_j:
                tj = NUM_RX.sub('', tracks[j])
                row.append(_sim(ti, tj))
            cost.append(row)
        assign = _hungarian_max(cost)
        for a_i, a_j in assign:
            i = rem_i[a_i]; j = rem_j[a_j]; sc = cost[a_i][a_j]
            if sc >= filename_thresh:
                locked.append((i,j,sc,"filename"))
                used_i.add(i); used_j.add(j)

    # C) optional OCR fallback (greedy but only on leftovers)
    if use_ocr:
        rem_i = [i for i in range(len(images)) if i not in used_i]
        rem_j = [j for j in range(len(tracks)) if j not in used_j]
        if rem_i and rem_j:
            for i in rem_i:
                txt = _ocr_text(images[i], lang=lang, psm=psm) or _title_from_filename(images[i])
                best_j, best_sc = None, -1.0
                for j in rem_j:
                    sc = _sim(txt, NUM_RX.sub('', tracks[j]))
                    if sc > best_sc: best_j, best_sc = j, sc
                if best_j is not None:
                    locked.append((i,best_j,best_sc,"ocr"))
                    used_i.add(i); used_j.add(best_j)

    locked.sort(key=lambda x: x[1])
    leftover_images = [i for i in range(len(images)) if i not in used_i]
    return locked, leftover_images

# ---------- Runner ----------
def run(album_root:Path, src_dir:Path, tracks:list[tuple[str,str]], *,
        out_ext:str="png", lang:str="eng", psm:str|None="6",
        dry_run:bool=False, force:bool=False, say=print, ocr:bool=False, strict:bool=True,
        filename_thresh:float=0.82):
    out_dir = album_root/"Artwork"
    out_dir.mkdir(parents=True, exist_ok=True)
    images = sorted([p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])
    if not images:
        say("❌ No images found in source directory."); return 1

    labels = [f"{nn} - {title}" for nn, title in tracks]
    pairs, leftovers = plan_matches(images, labels, lang=lang, psm=psm, use_ocr=ocr, filename_thresh=filename_thresh)

    say("Artwork pairing plan:")
    for i, j, sc, method in pairs:
        nn, title = tracks[j]
        dst = out_dir / f"{nn} - {title}.{out_ext}"
        say(f"  {images[i].name}  →  {dst.relative_to(album_root)}  [{method}  score={sc:.2f}]")

    if leftovers:
        say("\nUnmatched (kept original name):")
        for i in leftovers:
            say(f"  {images[i].name}  →  Artwork/_UNMATCHED/{images[i].name}")

    if dry_run:
        say("DRY-RUN: no files moved."); return 0

    # Apply accepted matches
    used_dst = set()
    for i, j, sc, method in pairs:
        nn, title = tracks[j]
        src = images[i]
        dst = out_dir / f"{nn} - {title}.{out_ext}"
        if dst in used_dst:
            say(f"⏭️  Skipping duplicate target: {dst.name}"); 
            continue
        used_dst.add(dst)
        if dst.exists() and not force:
            say(f"⏭️  Exists: {dst.name}"); 
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        try: shutil.move(str(src), str(dst))
        except Exception: shutil.copy2(str(src), str(dst))
        say(f"✓ {src.name} → {dst.relative_to(album_root)}")

    # Park leftovers safely (strict mode)
    if leftovers:
        park = out_dir / "_UNMATCHED"
        park.mkdir(parents=True, exist_ok=True)
        for i in leftovers:
            src = images[i]
            dst = park / src.name
            if dst.exists() and not force:
                say(f"⏭️  Exists: {dst.name}"); 
                continue
            try: shutil.move(str(src), str(dst))
            except Exception: shutil.copy2(str(src), str(dst))
            say(f"• parked → {dst.relative_to(album_root)}")
    return 0
'''

_EMBED_YOUTUBE = r"""
# youtube (embedded) — OAuth + upload + playlist
from __future__ import annotations
from pathlib import Path
import json, webbrowser, time
from typing import Optional, Iterable

try:
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
except Exception:
    # Doctor will install these; we purposely import lazily in top-level script.
    InstalledAppFlow = Credentials = Request = build = MediaFileUpload = None

# Scopes:
# - youtube            : full manage (needed for playlists create/insert)
# - youtube.upload     : upload/manage videos (explicit upload permission)
# - youtube.readonly   : read channel & playlist metadata (nice to have)
SCOPES = [
    "https://www.googleapis.com/auth/youtube",
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
]
# (If you prefer the single broad scope only, you can reduce to just 'youtube'.)

def _profiles_base() -> Path:
    return Path.home() / ".config" / "AlbumForge" / "youtube" / "profiles"

def profile_dir(name:str="default") -> Path:
    p = _profiles_base() / name
    p.mkdir(parents=True, exist_ok=True)
    return p

def load_creds(profile:str="default"):
    p = profile_dir(profile)
    tok = p/"token.json"; sec = p/"client_secret.json"
    if Credentials is None:
        raise RuntimeError("YouTube client not available. Run: AlbumForge doctor")
    creds = None
    if tok.exists():
        creds = Credentials.from_authorized_user_file(str(tok), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            # Refresh and persist the refreshed token so future runs don't re-auth
            creds.refresh(Request())
            try:
                tok.write_text(creds.to_json())
            except Exception:
                pass
        else:
            if not sec.exists():
                raise FileNotFoundError(
                    f"Missing {sec}. Download OAuth client (Desktop) JSON and place it there."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(sec), SCOPES)
            try:
                # Desktop-friendly local server flow (opens browser)
                creds = flow.run_local_server(port=0, open_browser=True)
            except Exception:
                # Fallback: console flow
                creds = flow.run_console()
            tok.write_text(creds.to_json())
    return creds

def client(profile:str="default"):
    creds = load_creds(profile)
    # static_discovery=False avoids fetching the big discovery doc
    return build("youtube", "v3", credentials=creds, static_discovery=False)

def whoami(y, default="?"):
    ch = y.channels().list(part="snippet,contentDetails", mine=True, maxResults=1).execute()
    items = ch.get("items") or []
    if not items: return default
    it = items[0]
    return f"{it['snippet']['title']} ({it['id']})"

def ensure_playlist(y, title:str, privacy:str="public", desc:str="") -> str:
    # Try to find existing playlist by exact title
    req = y.playlists().list(part="snippet", mine=True, maxResults=50)
    while True:
        resp = req.execute()
        for pl in resp.get("items", []):
            if pl["snippet"]["title"] == title:
                return pl["id"]
        tok = resp.get("nextPageToken")
        if not tok:
            break
        req = y.playlists().list(part="snippet", mine=True, maxResults=50, pageToken=tok)
    # Create if not found
    resp = y.playlists().insert(
        part="snippet,status",
        body={
            "snippet": {"title": title, "description": desc},
            "status": {"privacyStatus": privacy},
        },
    ).execute()
    return resp["id"]

def upload_mp4(y, path:Path, *, title:str, desc:str="", tags:Optional[list[str]]=None, privacy:str="unlisted") -> str:
    media = MediaFileUpload(str(path), chunksize=-1, resumable=True, mimetype="video/mp4")
    body = {
        "snippet": {"title": title, "description": desc, "tags": tags or []},
        "status": {"privacyStatus": privacy},
    }
    req = y.videos().insert(part="snippet,status", body=body, media_body=media)

    # Resumable upload with simple exponential backoff
    response = None
    backoff = 1.0
    while response is None:
        try:
            status, response = req.next_chunk()
        except Exception:
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 30.0)
    return response["id"]

def add_to_playlist(y, playlist_id:str, video_id:str, position:int|None=None):
    body = {
        "snippet": {
            "playlistId": playlist_id,
            "resourceId": {"kind": "youtube#video", "videoId": video_id},
        }
    }
    if position is not None:
        body["snippet"]["position"] = position
    y.playlistItems().insert(part="snippet", body=body).execute()
"""

# ================
# Module bootstrap
# ================
import types as _types

def _build_module(name: str, src: str):
    mod = _types.ModuleType(name)
    ns = mod.__dict__
    exec(src, ns, ns)
    return mod

# expose embedded modules
ti = _build_module("term_images", _EMBED_TERM_IMAGES)
rv = _build_module("render",       _EMBED_RENDER)
am = _build_module("artmatch",     _EMBED_ARTMATCH)
yt = _build_module("youtube", _EMBED_YOUTUBE)


# =====================
# Config & dependencies
# =====================
DEFAULTS = {
    "music_root": str(Path.home() / "Music"),
    "downloads_dir": str(Path.home() / "Downloads"),
    "ocr_lang": "eng",
    "ocr_psm": "6",
    "image_ext": "png",
    "freshness_minutes": 0,     # 0 = off
    "editor": os.environ.get("EDITOR", "nano"),
}

CFG_DIR = Path.home() / ".config" / "album_forge"
CFG_FILE = CFG_DIR / "config.toml"
STATE_FILE_NAME = ".album_forge.json"

AUDIO_EXTS = {".mp3", ".m4a", ".flac", ".wav", ".ogg", ".opus"}
IMG_EXTS   = {".png", ".jpg", ".jpeg", ".webp"}

# tomllib for 3.11+, fallback to tomli if present
def _load_toml_bytes(b: bytes) -> dict:
    try:
        import tomllib
        return tomllib.loads(b.decode())
    except Exception:
        try:
            import tomli
            return tomli.loads(b.decode())
        except Exception as e:
            raise SystemExit("❌ Need Python 3.11+ (tomllib) or install 'tomli' (pip install tomli).")

def load_config() -> dict:
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    if not CFG_FILE.exists():
        with CFG_FILE.open("wb") as f:
            lines = ["# album_forge config\n"]
            for k, v in DEFAULTS.items():
                lines.append(f'{k} = "{v}"\n' if isinstance(v, str) else f"{k} = {v}\n")
            f.write("".join(lines).encode())
    with CFG_FILE.open("rb") as f:
        cfg = _load_toml_bytes(f.read())
    for k, v in DEFAULTS.items():
        cfg.setdefault(k, v)
    return cfg


# -----------
# UI helpers
# -----------
def say(x): print(x)
def ok(x): print(f"\033[32m{x}\033[0m")
def warn(x): print(f"\033[33m{x}\033[0m")
def err(x): print(f"\033[31m{x}\033[0m", file=sys.stderr)

# --- YouTube: upload whole album as a playlist --------------------------------
def _album_render_dir(album_name: str) -> Path:
    return MUSIC_ROOT / album_name / "_video"

def _album_meta_file(album_name: str) -> Path:
    return MUSIC_ROOT / album_name / ".albumforge.youtube.json"

def youtube_upload_album(album_name: str, *, privacy: str = "unlisted", profile: str = "default") -> None:
    """Upload all rendered MP4s for an album as a YouTube playlist.
       Skips already-uploaded tracks using a small local map file."""
    from pathlib import Path
    import json

    # lazy import from the embedded client
    y = client(profile)
    playlist_title = album_name
    playlist_desc  = f"{album_name} — uploaded by AlbumForge"
    playlist_id = ensure_playlist(y, playlist_title, privacy=privacy, desc=playlist_desc)

    # state map (filename -> videoId)
    meta_path = _album_meta_file(album_name)
    state = {}
    if meta_path.exists():
        try:
            state = json.loads(meta_path.read_text())
        except Exception:
            state = {}

    # discover rendered videos
    vdir = _album_render_dir(album_name)
    if not vdir.exists():
        say(f"No rendered videos found at: {vdir}")
        say("Run: AlbumForge video --album \"{album_name}\" first.")
        return

    # keep stable order: 01.., 02.. etc.
    files = sorted([p for p in vdir.glob("*.mp4") if p.is_file()], key=lambda p: p.name.lower())
    if not files:
        say(f"No .mp4 files in {vdir}")
        return

    ok(f"Uploading to: {whoami(y)}")
    ok(f"PlayList ready: {playlist_title} (id: {playlist_id})")

    uploaded = 0
    added    = 0

    for idx, path in enumerate(files, start=1):
        title = path.stem
        if path.name in state:
            # Skip upload; ensure it's in the playlist (best-effort)
            vid = state[path.name]
            try:
                add_to_playlist(y, playlist_id, vid, position=idx-1)
            except Exception:
                pass
            say(f"Exists, skipping: {path.name}")
            continue

        # Upload
        vid = upload_mp4(y, path, title=title, desc=f"{album_name} — {title}", tags=[album_name], privacy=privacy)
        state[path.name] = vid
        uploaded += 1
        ok(f"Uploaded: {idx:02d}  →  {path.name}")

        # Add to playlist in track order
        add_to_playlist(y, playlist_id, vid, position=idx-1)
        added += 1
        ok(f"Uploaded & added: {idx:02d}  →  {path.name}")

        # Persist mapping after each video (so resume is safe)
        meta_path.write_text(json.dumps(state, indent=2))

    if uploaded == 0:
        ok("All tracks already uploaded; ensured playlist ordering.")
    else:
        ok(f"Album upload complete. New videos: {uploaded}, added to playlist: {added}")


def _album_tracks(album_root: Path) -> list[tuple[str,str]]:
    state = load_state(album_root)
    tracks = [(t["n"], t["title"]) for t in state.get("tracks", [])]
    if tracks: return tracks
    tracks = []
    for p in sorted((album_root/"Audio").glob("*")):
        m = re.match(r"^\s*(\d{2})\s*-\s*(.+)\.[^.]+$", p.name)
        if m: tracks.append((m.group(1), m.group(2)))
    return tracks

def action_youtube_auth(cfg:dict, args):
    prof = getattr(args, "profile", "default")
    # If --client points to a JSON, copy it into the profile dir
    if getattr(args, "client", None):
        src = Path(args.client).expanduser()
        dst = yt.profile_dir(prof)/"client_secret.json"
        dst.write_bytes(src.read_bytes())
        ok(f"Client secret installed: {dst}")
    # This will run local-server flow or console fallback inside yt.load_creds
    try:
        cl = yt.client(prof)
        ok("Auth complete.")
        say("Channel: " + yt.whoami(cl))
    except Exception as e:
        err(f"Auth failed: {e}")

def action_youtube_upload(cfg:dict, args):
    prof = getattr(args, "profile", "default")
    privacy = getattr(args, "privacy", "unlisted")
    album_root = find_album_root(Path(cfg["music_root"]).expanduser(), args.album)
    ensure_album_tree(album_root)

    # Ensure we have MP4s; if not, call your existing renderer
    videos = sorted((album_root/"Video").glob("*.mp4"))
    if not videos:
        warn("No MP4s in Video/. Rendering from Audio+Artwork…")
        action_video(cfg, argparse.Namespace(album=album_root.name, force=False))
        videos = sorted((album_root/"Video").glob("*.mp4"))
        if not videos:
            err("Still no videos to upload."); return

    tracks = _album_tracks(album_root)
    title_map = { f"{nn} - {title}": (nn, title) for nn, title in tracks }

    y = yt.client(prof)
    chan = yt.whoami(y)
    ok(f"Uploading to: {chan}")

    playlist_name = album_root.name
    pl_id = yt.ensure_playlist(y, playlist_name, privacy="public")  # playlist usually public
    ok(f"Playlist ready: {playlist_name} (id: {pl_id})")

    # Sort by numeric prefix to keep album order
    def _stem(p:Path): return re.sub(r"\.[^.]+$","",p.name)
    ordered = sorted(videos, key=lambda p: int(re.match(r"^\s*(\d{2})", _stem(p)).group(0)) if re.match(r"^\s*(\d{2})", _stem(p)) else 999)

    for idx, mp4 in enumerate(ordered):
        stem = _stem(mp4)
        # Title formatting
        nn_title = stem
        nn, title = (re.match(r"^\s*(\d{2})\s*-\s*(.+)$", stem).groups()
                     if re.match(r"^\s*(\d{2})\s*-\s*(.+)$", stem) else ("", stem))
        disp_title = f"{album_root.name} — {nn} — {title}" if nn else f"{album_root.name} — {title}"

        # Optional description: embed album + lyrics file if present
        lfile = (album_root/"Lyrics"/f"{nn} - {title}.txt") if nn else None
        lyrics = ""
        if lfile and lfile.exists():
            try:
                lyrics = lfile.read_text().strip()
            except Exception:
                pass
        desc = f"{album_root.name}\n\n{lyrics}" if lyrics else album_root.name

        vid = yt.upload_mp4(y, mp4, title=disp_title, desc=desc, privacy=privacy, tags=[album_root.name, "AlbumForge"])
        yt.add_to_playlist(y, pl_id, vid, position=idx)
        ok(f"Uploaded + added: {mp4.name}")

def _sim(a:str, b:str) -> float:
    a = re.sub(r"[^A-Za-z0-9]+", " ", a.lower()).strip()
    b = re.sub(r"[^A-Za-z0-9]+", " ", b.lower()).strip()
    return SequenceMatcher(None, a, b).ratio()

def have(cmd:str)->bool:
    return shutil.which(cmd) is not None

def is_wezterm()->bool:
    return os.environ.get("TERM_PROGRAM") == "WezTerm" or bool(os.environ.get("WEZTERM_EXECUTABLE") or os.environ.get("WT_SESSION"))

def clear_screen():
    try:
        if os.environ.get("AlbumForge_NOCLEAR"):
            return
        if os.name == "nt":
            os.system("cls")
        else:
            os.system("clear")
    except Exception:
        pass

def pause_then_clear():
    try:
        input("\nPress Enter to return to the main menu…")
    except EOFError:
        pass
    clear_screen()

def _safe_input(prompt:str) -> str:
    try:
        return input(prompt)
    except EOFError:
        return ""

def sanitize_title(name:str)->str:
    name = Path(name).stem
    name = re.sub(r"^\s*\d+\s*[-_.]\s*", "", name)
    name = name.replace("_", " ").strip()
    name = re.sub(r"\s+", " ", name)
    return name.title()

def _pick_player() -> list[str] | None:
    # prefer mpv, then vlc, then ffplay
    if have("mpv"):   return ["mpv", "--no-video", "--audio-display=no"]
    if have("vlc"):   return ["vlc", "--intf", "dummy", "--quiet"]
    if have("ffplay"):return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error"]
    return None

def set_album_cover_from_image(album_root: Path, image: Path, *, ext:str="png", mode:str="copy"):
    """Write album-root cover named after the folder: <AlbumName>.<ext>"""
    dst = album_root / f"{album_root.name}.{ext}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "move":
        shutil.move(str(image), str(dst))
    else:
        shutil.copy2(str(image), str(dst))
    ok(f"Album cover set: {dst.name}")
    return dst

def detect_best_album_cover(src_dir: Path, album_name: str, *, tracks: list[tuple[str,str]]|None=None,
                            lang:str="eng", psm:str|None="6") -> Path | None:
    """
    Heuristic: prefer filenames containing 'cover'/'front', else best title match.
    If tracks are provided, fall back to artmatch plan_matches to find highest score.
    """
    cand = [p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    if not cand: return None
    pri = []
    for p in cand:
        name = p.stem.lower()
        score = _sim(name, album_name)
        if any(k in name for k in ("cover","front","folder")):
            score += 0.25
        pri.append((score, p))
    pri.sort(reverse=True, key=lambda x: x[0])
    best = pri[0][1] if pri else None

    if tracks:
        # Build labels and ask artmatch for best overall match; use highest score as tie-breaker
        labels = [f"{nn} - {title}" for nn, title in tracks]
        pairs, leftovers = am.plan_matches(cand, labels, lang=lang, psm=psm, use_ocr=True, filename_thresh=0.0)
        if pairs:
            # choose the single highest scoring image across any track
            pairs.sort(key=lambda t: t[2], reverse=True)
            i, _j, sc, _m = pairs[0]
            # prefer artmatch pick if it significantly beats naive heuristic
            if best is None or sc >= 0.6:
                return cand[i]
    return best

def manage_album_menu(cfg:dict):
    def ask(prompt:str, default:str|None=None) -> str:
        suffix = f" [{default}]" if default is not None else ""
        resp = _safe_input(f"{prompt}{suffix}: ").strip()
        return resp if resp else (default if default is not None else "")

    def ask_yn(prompt:str, default:bool=False) -> bool:
        d = "Y/n" if default else "y/N"
        r = _safe_input(f"{prompt} ({d}): ").strip().lower()
        return (r in ("y","yes")) if r else default

    # light wrapper: prefer internal helper if present; else shell out to CLI
    def _youtube_upload_album(album_name:str, privacy:str="unlisted"):
        try:
            # If the project defines a python helper, use it
            fn = globals().get("youtube_upload_album")
            if callable(fn):
                return fn(album_name, privacy=privacy)
        except Exception:
            pass
        # Fallback: call the already-working CLI
        try:
            cmd = ["AlbumForge", "youtube", "upload", "--album", album_name, "--privacy", privacy]
            return subprocess.call(cmd)
        except Exception as e:
            err(f"YouTube upload failed to start: {e}")
            return 1

    album = ask("Album (blank to pick recent)") or None
    album_root = find_album_root(Path(cfg["music_root"]).expanduser(), album)
    ensure_album_tree(album_root)

    while True:
        clear_screen()
        a = len(list((album_root/"Audio").glob("*")))
        l = len(list((album_root/"Lyrics").glob("*.txt")))
        r = len(list((album_root/"Artwork").glob("*")))
        v = len(list((album_root/"Video").glob("*.mp4")))
        total = max(a,l,r,v,1)
        pct = int(round(100 * min(a,l,r,v) / total))
        say(f"=== Manage: {album_root.name} ===  ({pct}%)")
        say(f"A:{a}  L:{l}  R:{r}  V:{v}")
        say(
            "\n"
            "1) Play album\n"
            "2) Show album cover\n"
            "3) Detect & set album cover\n"
            "4) Status\n"
            "5) Regenerate artwork\n"
            "6) Render videos\n"
            "7) Delete album (DANGEROUS)\n"
            "8) Upload to YouTube\n"
            "b) Back"
        )
        choice = _safe_input("> ").strip().lower()

        if choice == "1":
            player = _pick_player()
            if not player:
                err("No supported player (mpv/vlc/ffplay) found.")
                _safe_input("Enter…"); continue
            plist = sorted((album_root/"Audio").glob("*"))
            if not plist:
                warn("No audio.")
                _safe_input("Enter…"); continue
            try:
                subprocess.call(player + [str(p) for p in plist])
            except Exception as e:
                err(f"Player failed: {e}")
            continue

        if choice == "2":
            covers = [p for p in album_root.glob(f"{album_root.name}.*") if p.suffix.lower() in IMG_EXTS]
            clear_screen()
            if covers:
                if not ti.preview(str(covers[0])):
                    warn("Preview failed. Install chafa or viu for text previews.")
            else:
                warn("No album cover found in album root.")
            _safe_input("Enter…")
            clear_screen()
            continue

        if choice == "3":
            src = ask("Source folder for cover detection", str(Path(cfg["downloads_dir"]).expanduser()))
            srcp = Path(src).expanduser()
            if not srcp.exists():
                err("No such folder."); _safe_input("Enter…"); continue
            # derive tracks for better matching
            state = load_state(album_root)
            tracks = [(t["n"], t["title"]) for t in state.get("tracks", [])]
            if not tracks:
                for p in sorted((album_root/"Audio").glob("*")):
                    m = re.match(r"^\s*(\d{2})\s*-\s*(.+)\.[^.]+$", p.name)
                    if m: tracks.append((m.group(1), m.group(2)))
            best = detect_best_album_cover(
                srcp, album_root.name, tracks=tracks,
                lang=cfg.get("ocr_lang","eng"), psm=cfg.get("ocr_psm","6")
            )
            if not best:
                warn("No candidate images found."); _safe_input("Enter…"); continue
            say(f"Best candidate: {best.name}")
            if ask_yn("Use this as album cover?", True):
                set_album_cover_from_image(
                    album_root, best, ext=cfg.get("image_ext","png"), mode="copy"
                )
            _safe_input("Enter…"); continue

        if choice == "4":
            action_status(cfg, argparse.Namespace(album=album_root.name))
            _safe_input("Enter…"); continue

        if choice == "5":
            src = ask("Source folder for artwork", str(Path(cfg["downloads_dir"]).expanduser()))
            amode = ask("Artwork ingest mode (copy/move)", "copy").lower()
            amode = amode if amode in ("copy","move") else "copy"
            action_art(
                cfg,
                argparse.Namespace(album=album_root.name, src=src, mode=amode, dry_run=False, force=False)
            )
            _safe_input("Enter…"); continue

        if choice == "6":
            action_video(cfg, argparse.Namespace(album=album_root.name, force=False))
            _safe_input("Enter…"); continue

        if choice == "7":
            warn("This will MOVE the entire album directory to a trash folder (non-destructive).")
            say(f"Type DELETE {album_root.name} to confirm.")
            if _safe_input("> ").strip() == f"DELETE {album_root.name}":
                try:
                    safe_trash_album(album_root)
                    ok("Album moved to trash.")
                    return
                except Exception as e:
                    err(f"Failed: {e}")
            else:
                warn("Delete aborted.")
            _safe_input("Enter…"); continue

        if choice == "8":
            # Upload to YouTube (creates/updates playlist with album name)
            vids = sorted((album_root/"Video").glob("*.mp4"))
            if not vids:
                warn("No rendered videos found.")
                if ask_yn("Render videos now?", True):
                    action_video(cfg, argparse.Namespace(album=album_root.name, force=False))
                    vids = sorted((album_root/"Video").glob("*.mp4"))
                else:
                    _safe_input("Enter…"); continue

            if not vids:
                err("Still no videos after render; aborting upload.")
                _safe_input("Enter…"); continue

            privacy = ask("YouTube privacy (public/unlisted/private)", "unlisted").strip().lower()
            if privacy not in ("public","unlisted","private"):
                warn("Invalid privacy; defaulting to 'unlisted'.")
                privacy = "unlisted"

            say(f"Uploading album '{album_root.name}' to YouTube as {privacy}…")
            rc = _youtube_upload_album(album_root.name, privacy=privacy)
            if rc not in (0, None):
                err("YouTube upload reported an error.")
            else:
                ok("YouTube upload finished (check terminal output for playlist id / links).")
            _safe_input("Enter…"); continue

        if choice in ("b","q","back"):
            return


# ----------
# Doctor 💉
# ----------
PKG_ALIASES = {
    "ffmpeg":   {"apt":"ffmpeg","pacman":"ffmpeg","dnf":"ffmpeg","zypper":"ffmpeg","brew":"ffmpeg","apk":"ffmpeg"},
    "tesseract":{"apt":"tesseract-ocr","pacman":"tesseract","dnf":"tesseract","zypper":"tesseract","brew":"tesseract","apk":"tesseract-ocr"},
    "chafa":    {"apt":"chafa","pacman":"chafa","dnf":"chafa","zypper":"chafa","brew":"chafa","apk":"chafa"},
    "viu":      {"apt":"viu","pacman":"viu","dnf":"viu","zypper":"viu","brew":"viu","apk":"viu"},
}

def _detect_pkgmgr()->str|None:
    for pm in ("pacman","apt","dnf","zypper","apk","brew"):
        if shutil.which(pm):
            return pm
    return None

def _install_cmd(pm:str, pkgs:list[str]) -> list[str] | None:
    if pm == "pacman":
        return ["sudo","pacman","-S","--needed","--noconfirm",*pkgs]
    if pm == "apt":
        # run via shell to chain update+install (caller joins + shell=True)
        return ["sudo","apt","update","-y","&&","sudo","apt","install","-y",*pkgs]
    if pm == "dnf":
        return ["sudo","dnf","install","-y",*pkgs]
    if pm == "zypper":
        return ["sudo","zypper","install","-y",*pkgs]
    if pm == "apk":
        return ["sudo","apk","add",*pkgs]
    if pm == "brew":
        return ["brew","install",*pkgs]
    return None

def doctor(interactive: bool = True) -> int:
    import sys, glob, os

    def _venv_active() -> bool:
        return sys.prefix != getattr(sys, "base_prefix", sys.prefix)

    def _externally_managed() -> bool:
        # Heuristic for PEP 668 (Arch & friends): EXTERNALLY-MANAGED marker exists
        paths = [
            "/usr/lib/python*/EXTERNALLY-MANAGED",
            "/usr/lib/python*/dist-packages/EXTERNALLY-MANAGED",
            "/usr/lib/python*/site-packages/EXTERNALLY-MANAGED",
        ]
        for pat in paths:
            if glob.glob(pat):
                return True
        return False

    def _pip_install(pkgs: list[str], allow_break=False) -> int:
        cmd = [sys.executable, "-m", "pip", "install"]
        if not _venv_active():
            cmd.append("--user")
            if allow_break:
                cmd.append("--break-system-packages")
        cmd += pkgs
        say(" ".join(cmd))
        try:
            return subprocess.call(cmd)
        except Exception as e:
            err(f"pip install failed: {e}")
            return 1

    def _reexec_now():
        from pathlib import Path
        import sys, os
        launcher = Path.home() / ".local" / "bin" / "AlbumForge"
        venv_py = Path.home() / ".config" / "AlbumForge" / ".venv" / "bin" / "python"

        if launcher.exists():
            ok("Reloading via launcher to activate local venv…")
            os.execv(str(launcher), ["AlbumForge", *sys.argv[1:]])
        elif venv_py.exists():
            ok("Reloading with venv interpreter…")
            os.execv(str(venv_py), [str(venv_py), "-m", "AlbumForge", *sys.argv[1:]])
        else:
            ok("Reloading current interpreter…")
            os.execv(sys.executable, [sys.executable] + sys.argv)

    def _create_local_venv_and_install(pkgs: list[str]) -> int:
        vdir = Path.home() / ".config" / "AlbumForge" / ".venv"
        vdir.parent.mkdir(parents=True, exist_ok=True)
        ok(f"Creating venv at {vdir} …")
        rc = subprocess.call([sys.executable, "-m", "venv", str(vdir)])
        if rc != 0:
            err(f"venv creation failed ({rc})"); return rc
        py  = str(vdir / "bin" / "python")
        say(f"{py} -m pip install {' '.join(pkgs)}")
        rc = subprocess.call([py, "-m", "pip", "install", *pkgs])
        if rc == 0:
            ok("Installed into local venv.")
            launcher = Path.home() / ".local" / "bin" / "AlbumForge"
            try:
                launcher.parent.mkdir(parents=True, exist_ok=True)
                body = f"""#!/usr/bin/env bash
source "{vdir}/bin/activate"
exec python -m AlbumForge "$@"
"""
                launcher.write_text(body)
                launcher.chmod(0o755)
                ok(f"Launcher written: {launcher} (uses local venv)")
            except Exception:
                pass
            _reexec_now()
        return rc

    ok("Doctor: scanning environment…")

    # ---------- system binaries ----------
    core_bins = {
        "ffmpeg": not have("ffmpeg"),
        "tesseract": not have("tesseract"),
    }
    opt_bins  = {
        "chafa": not have("chafa"), "viu": not have("viu"),
        "kitty": not have("kitty"), "wezterm": not have("wezterm"),
        "img2sixel": not have("img2sixel"), "mpv": not have("mpv"), "vlc": not have("vlc"),
    }

    missing_core_bins = [k for k, miss in core_bins.items() if miss]
    if missing_core_bins: warn("Missing core tools: " + ", ".join(missing_core_bins))
    else: ok("Core OK: ffmpeg & tesseract detected.")

    missing_opt_bins = [k for k, miss in opt_bins.items() if miss]
    if missing_opt_bins: warn("Optional tools missing (previews/players): " + ", ".join(missing_opt_bins))
    else: ok("Optional tools present (nice!).")

    # ---------- python modules ----------
    # Import-name → pip-package mapping
    wanted_mods = {
        "PIL": "pillow",                         # Pillow
        "rich": "rich",                          # console niceties
        "textual": "textual",                    # (if you still use the TUI bits)
        "google_auth_oauthlib": "google-auth-oauthlib",      # YouTube OAuth
        "googleapiclient": "google-api-python-client",       # YouTube Data API
    }

    missing_py: list[str] = []
    for import_name, pip_name in wanted_mods.items():
        try:
            __import__(import_name)
        except Exception:
            missing_py.append(pip_name)

    if missing_py:
        warn("Missing Python modules: " + ", ".join(sorted(set(missing_py))))
    else:
        ok("Python modules OK (pillow, rich, textual, google-auth-oauthlib, google-api-python-client).")

    if not interactive:
        return 0 if not (missing_core_bins or missing_py) else 1

    # ---------- install core system deps ----------
    if missing_core_bins:
        pm = _detect_pkgmgr()
        if not pm:
            warn("No known package manager detected (pacman/apt/dnf/zypper/apk/brew).")
        else:
            pkgs = []
            for m in missing_core_bins:
                alias = PKG_ALIASES.get(m, {})
                pkgs.append(alias.get(pm, m))
            say(f"\nDetected package manager: {pm}")
            say("I can install: " + " ".join(pkgs))
            if _safe_input("Proceed with system install? (y/N): ").strip().lower() in ("y","yes"):
                cmd = _install_cmd(pm, pkgs)
                if not cmd:
                    warn("Unsupported installer on this system."); return 1
                try:
                    rc = subprocess.call(" ".join(cmd), shell=True) if pm == "apt" else subprocess.call(cmd)
                    if rc == 0: ok("System install complete.")
                    else: err(f"Installer returned {rc}."); return rc
                except Exception as e:
                    err(f"Install failed: {e}"); return 1
            else:
                warn("Skipped system package installation.")

    # ---------- install Python deps (PEP668-aware) ----------
    def _venv_or_install(pkgs: list[str]) -> int:
        say("I can install Python modules: " + " ".join(pkgs))
        if _safe_input("Install into this environment with pip? (y/N): ").strip().lower() in ("y","yes"):
            rc = subprocess.call([sys.executable, "-m", "pip", "install", *pkgs])
            if rc == 0:
                ok("Python module install complete.")
                _reexec_now()
            else:
                err(f"pip exited with {rc}.")
            return rc
        else:
            warn("Skipped Python module installation.")
            return 1

    if missing_py:
        say("\nPython environment:")
        say(f"  Interpreter : {sys.executable}")
        say(f"  Venv active : {'yes' if _venv_active() else 'no'}")
        say(f"  Prefix      : {sys.prefix}")
        say(f"  BasePrefix  : {getattr(sys, 'base_prefix', sys.prefix)}")

        pep668 = _externally_managed()
        venv_active = _venv_active()

        if pep668 and not venv_active:
            pm = _detect_pkgmgr()
            if pm == "pacman":
                # Map pip package names to pacman equivalents where possible
                pac_pkgs = []
                for p in sorted(set(missing_py)):
                    if p == "pillow": pac_pkgs.append("python-pillow")
                    elif p == "rich": pac_pkgs.append("python-rich")
                    elif p == "textual": pac_pkgs.append("python-textual")
                    elif p == "google-auth-oauthlib": pac_pkgs.append("python-google-auth-oauthlib")
                    elif p == "google-api-python-client": pac_pkgs.append("python-google-api-python-client")
                    else: pac_pkgs.append(p)  # fallback
                say("Option A) Install via pacman: " + " ".join(pac_pkgs))
                say("Option B) Create a local venv just for AlbumForge (recommended).")
                say("Option C) Force pip with --break-system-packages (not recommended).")
                choice = _safe_input("Choose [A/B/C/N]: ").strip().lower()
                if choice == "a":
                    cmd = _install_cmd(pm, pac_pkgs)
                    rc = subprocess.call(cmd)
                    if rc == 0: _reexec_now()
                    return rc
                elif choice == "b":
                    return _create_local_venv_and_install(sorted(set(missing_py)))
                elif choice == "c":
                    return subprocess.call([sys.executable,"-m","pip","install","--user","--break-system-packages",*sorted(set(missing_py))])
                else:
                    warn("Skipped Python module installation.")
                    return 1
            else:
                say("Option A) Create a local venv (recommended).")
                say("Option B) Force pip with --break-system-packages.")
                choice = _safe_input("Choose [A/B/N]: ").strip().lower()
                if choice == "a":
                    return _create_local_venv_and_install(sorted(set(missing_py)))
                elif choice == "b":
                    return subprocess.call([sys.executable,"-m","pip","install","--user","--break-system-packages",*sorted(set(missing_py))])
                else:
                    warn("Skipped Python module installation.")
                    return 1
        else:
            return _venv_or_install(sorted(set(missing_py)))

    ok("Doctor finished: all green.")
    return 0


# -----------------------
# Core helpers & actions
# -----------------------
def _plan_audio_ingest(album_root: Path, src_files: list[Path], state: dict) -> list[tuple[Path, Path, str, str]]:
    """
    Return planned (src, dst, NN, Title) using the same seeded shuffle/numbering
    as the real ingest, but without writing anything.
    """
    audio_dir = album_root / "Audio"
    existing_audios = sorted([p for p in audio_dir.iterdir()
                              if p.is_file() and p.suffix.lower() in AUDIO_EXTS])
    existing_nn = [re.match(r"^\s*(\d{2})\s*-\s*", p.name).group(1)
                   for p in existing_audios if re.match(r"^\s*(\d{2})\s*-\s*", p.name)]
    existing_nn += [t.get("n","") for t in state.get("tracks", [])]

    nums_needed = next_free_numbers(existing_nn, len(src_files))

    # lock shuffle to existing seed so preview == actual
    seed = state.get("seed")
    if not seed:
        seed = random.randint(100000, 999999)
    rnd = random.Random(seed)
    files = list(src_files)
    rnd.shuffle(files)

    plan = []
    for idx, src in enumerate(files):
        nn = nums_needed[idx]
        title = sanitize_title(src.name)
        dst = (audio_dir / f"{nn} - {title}{src.suffix.lower()}")
        plan.append((src, dst, nn, title))
    return plan

def find_album_root(music_root:Path, album:str|None)->Path:
    if album:
        return (music_root/album).resolve()
    albums = sorted(
        [p for p in music_root.iterdir() if p.is_dir() and p.name != ".git" and not p.name.startswith(".")],
        key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not albums:
        raise SystemExit("No albums in music_root.")
    say("Select album (recent first):")
    for i, p in enumerate(albums[:20], 1):
        say(f"  {i:2d}) {p.name}")
    choice = _safe_input("> ").strip()
    idx = int(choice) if choice.isdigit() else 1
    return albums[idx-1]

def ensure_album_tree(root:Path):
    for d in ("Audio","Lyrics","Artwork","Video","Notes"):
        (root/d).mkdir(parents=True, exist_ok=True)

def load_state(album_root:Path)->dict:
    f = album_root/STATE_FILE_NAME
    if f.exists():
        try:
            return json.loads(f.read_text())
        except Exception:
            warn("State file unreadable; starting fresh.")
    return {"tracks": [], "seed": None, "moved_at": None}

def save_state(album_root:Path, state:dict):
    (album_root/STATE_FILE_NAME).write_text(json.dumps(state, indent=2))

def next_free_numbers(existing:list[str], count:int)->list[str]:
    used = set(int(n) for n in existing if n.isdigit())
    nums, n = [], 1
    while len(nums) < count:
        if n not in used:
            nums.append(f"{n:02d}")
        n += 1
    return nums

def list_audio_in_dir(path: Path, freshness_minutes: int = 0) -> list[Path]:
    files = [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
    if freshness_minutes > 0:
        import time as _t
        cutoff = _t.time() - (freshness_minutes * 60)
        files = [p for p in files if p.stat().st_mtime >= cutoff]
    return sorted(files)

def list_images_in_dir(path: Path) -> list[Path]:
    return sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])

def action_create(cfg:dict, args):
    """
    Create/ensure the album tree. If --populate is set:
      • Build a DRY plan for audio (exact NN - Title destinations) using a persisted seed
      • Run OCR to predict artwork matches WITHOUT moving files
      • Show a full summary (audio plan, artwork predictions, lyric/video estimates)
      • On YES, perform audio ingest, artwork ingest, lyrics, and (if art exists) render videos
    """
    music_root = Path(cfg["music_root"]).expanduser()
    music_root.mkdir(parents=True, exist_ok=True)
    target = music_root / args.name
    if target.exists():
        say(f"Album exists: {target}")
    else:
        ensure_album_tree(target)
        ok(f"Created album: {target}")

    # Load state and ensure it exists on disk
    state = load_state(target)
    save_state(target, state)

    # Simple create (no auto-populate)
    if not getattr(args, "populate", False):
        return

    # ------- Gather sources -------
    src = Path(getattr(args, "src", "") or cfg["downloads_dir"]).expanduser()
    aud_src = list_audio_in_dir(src, cfg.get("freshness_minutes", 0))
    img_src = list_images_in_dir(src)
    mode = getattr(args, "mode", "copy")
    art_mode = getattr(args, "art_mode", "copy")

    # ------- Build detailed plans (no side effects) -------
    if not state.get("seed"):
        state["seed"] = random.randint(100000, 999999)
        save_state(target, state)

    # Audio plan (exact NN - Title destinations)
    plan_audio = _plan_audio_ingest(target, aud_src, dict(state)) if aud_src else []

    # Prospective tracks = existing + planned (avoid NN duplicates, prefer existing)
    tracks_existing = [(t["n"], t["title"]) for t in state.get("tracks", [])]
    tracks_planned  = [(nn, title) for (_s, _d, nn, title) in plan_audio]
    seen = {nn for nn, _ in tracks_existing}
    prospective_tracks = tracks_existing + [(nn, t) for nn, t in tracks_planned if nn not in seen]

    # Artwork match preview via OCR (prediction only; no file moves)
    art_pairs = []
    if img_src and prospective_tracks:
        titles = [t for _, t in prospective_tracks]
        # use_ocr=True for a realistic preview
        art_pairs, _ = am.plan_matches(
            img_src, titles,
            lang=cfg.get("ocr_lang","eng"),
            psm=cfg.get("ocr_psm","6"),
            use_ocr=True
        )

    # Estimated lyric files to create
    lyrics_dir = target / "Lyrics"
    est_new_lyrics = 0
    for (_s, _d, nn, title) in plan_audio:
        if not (lyrics_dir / f"{nn} - {title}.txt").exists():
            est_new_lyrics += 1

    # Estimated videos (predicted artwork matches)
    est_video = len(art_pairs)

    # ------- Present the plan BEFORE confirmation -------
    say("\n" + "="*72)
    warn(" ⚠️  AUTO-POPULATE IS ABOUT TO RUN ")
    say(f" Source folder : {src}")
    say(f" Audio files   : {len(aud_src)}")
    say(f" Image files   : {len(img_src)} (used by 'art')")
    say(f" Mode          : {mode.upper()} (audio ingest)")
    say("="*72 + "\n")

    # Audio plan table
    if plan_audio:
        say("Audio ingest plan:")
        for srcp, dstp, nn, title in plan_audio:
            say(f"  {srcp.name}  ->  {dstp.relative_to(target)}  [{mode}]")
    else:
        say("Audio ingest plan: (no audio candidates found)")

    # Artwork plan table (predicted)
    if art_pairs:
        say("\nArtwork match plan (prediction):")
        for i, j, sc, _method in art_pairs:
            nn, title = prospective_tracks[j]
            dst = (target / "Artwork") / f"{nn} - {title}.{cfg.get('image_ext','png')}"
            say(f"  {img_src[i].name}  ->  {dst.relative_to(target)}  [score {sc:.2f}]")
    else:
        say("\nArtwork match plan: (no images or no tracks yet)")

    say(f"\nEstimated new lyrics files: {est_new_lyrics}")
    say(f"Estimated videos to render: {est_video}")

    say("\nMAKE SURE YOU HAVE ALL SONGS AND ARTWORK IN THE SOURCE FOLDER.")
    say("Type YES (all caps) to continue, anything else to abort.")
    proceed = getattr(args, "yes", False) or (_safe_input("> ").strip() == "YES")
    if not proceed:
        warn("Aborted by user. Album created, nothing ingested.")
        return

    # ------- Execute actual ingest -------
    action_move(cfg, argparse.Namespace(
        album=args.name, src=str(src),
        dry_run=False, force=getattr(args, "force", False),
        mode=mode
    ))
    action_art(cfg, argparse.Namespace(
        album=args.name, src=str(src),
        mode=art_mode, dry_run=False,
        force=getattr(args, "force", False)
    ))
    
    # Try to set album cover from the same source, if not already present
    root_cover = list((target).glob(f"{target.name}.*"))
    if not root_cover:
        # build track list for smarter cover detect
        _state = load_state(target)
        _tracks = [(t["n"], t["title"]) for t in _state.get("tracks", [])]
        best = detect_best_album_cover(
            Path(getattr(args,"src", "") or cfg["downloads_dir"]).expanduser(),
            target.name,
            tracks=_tracks,
            lang=cfg.get("ocr_lang","eng"),
            psm=cfg.get("ocr_psm","6")
        )
        if best:
            set_album_cover_from_image(target, best, ext=cfg.get("image_ext","png"), mode="copy")

    action_make_lyrics(cfg, argparse.Namespace(album=args.name))

    # Render if any art now exists
    if any((target / "Artwork").glob("*")):
        action_video(cfg, argparse.Namespace(album=args.name, force=False))

    action_status(cfg, argparse.Namespace(album=args.name))
    
def action_move(cfg:dict, args):
    album_root = find_album_root(Path(cfg["music_root"]).expanduser(), args.album)
    ensure_album_tree(album_root)
    state = load_state(album_root)

    src_dir = Path(getattr(args, "src", "") or cfg["downloads_dir"]).expanduser()
    src_files = list_audio_in_dir(src_dir, cfg.get("freshness_minutes", 0))
    if not src_files:
        warn(f"No audio files in {src_dir} to ingest.")
        return

    audio_dir = album_root/"Audio"
    existing_audios = sorted([p for p in audio_dir.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS])
    existing_nn = [re.match(r"^\s*(\d{2})\s*-\s*", p.name).group(1)
                   for p in existing_audios if re.match(r"^\s*(\d{2})\s*-\s*", p.name)]
    existing_nn += [t.get("n","") for t in state.get("tracks", [])]
    nums_needed = next_free_numbers(existing_nn, len(src_files))

    if not state.get("seed"):
        state["seed"] = random.randint(100000, 999999)
    rnd = random.Random(state["seed"])
    rnd.shuffle(src_files)

    plan = []
    for idx, src in enumerate(src_files):
        nn = nums_needed[idx]
        title = sanitize_title(src.name)
        dst = audio_dir / f"{nn} - {title}{src.suffix.lower()}"
        plan.append((src, dst, nn, title))

    mode = getattr(args, "mode", "move")
    say("Audio ingest plan:")
    for src, dst, nn, title in plan:
        say(f"  {src.name}  ->  {dst.relative_to(album_root)}  [{mode}]")

    if getattr(args, "dry_run", False):
        say("DRY-RUN: nothing changed.")
        return

    for src, dst, nn, title in plan:
        if dst.exists() and not getattr(args, "force", False):
            warn(f"Exists, skipping: {dst.name}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if mode == "copy":
            shutil.copy2(str(src), str(dst))
        else:
            shutil.move(str(src), str(dst))
        ok(f"{mode.title()}: {src.name} -> {dst.relative_to(album_root)}")

        # merge into state (warn if number collision with different title)
        existing = {t["n"]: t for t in state["tracks"]}
        if nn in existing and existing[nn].get("title") != title:
            warn(f"Track number {nn} already in state with different title "
                 f"('{existing[nn].get('title')}' vs '{title}'). Keeping original title.")
            # keep original title/lyrics mapping, but update audio filename
            existing[nn]["audio"] = dst.name
        else:
            existing[nn] = {
                "n": nn,
                "title": title,
                "audio": dst.name,
                "lyrics": f"{nn} - {title}.txt"
            }
        state["tracks"] = [existing[k] for k in sorted(existing.keys())]

    # create empty lyrics after ingest
    lyrics_dir = album_root/"Lyrics"
    for t in state["tracks"]:
        lpath = lyrics_dir / t["lyrics"]
        if not lpath.exists():
            lpath.write_text("")
            say(f"Created lyrics: {lpath.relative_to(album_root)}")

    state["moved_at"] = datetime.now(timezone.utc).isoformat()
    save_state(album_root, state)
    ok("Audio ingest complete.")

def action_art(cfg:dict, args):
    album_root = find_album_root(Path(cfg["music_root"]).expanduser(), args.album)
    ensure_album_tree(album_root)

    state = load_state(album_root)
    tracks = [(t["n"], t["title"]) for t in state.get("tracks", [])]
    if not tracks:
        tracks = []
        for p in sorted((album_root/"Audio").glob("*")):
            m = re.match(r"^\s*(\d{2})\s*-\s*(.+)\.[^.]+$", p.name)
            if m: tracks.append((m.group(1), m.group(2)))
    if not tracks:
        warn("No tracks known yet. Run 'move' first."); return

    src_dir = Path(getattr(args, "src", "") or cfg["downloads_dir"]).expanduser()
    mode = getattr(args, "mode", "copy")

    use_temp = (mode == "copy")
    temp_dir = None
    staged_src = src_dir

    try:
        if use_temp:
            temp_dir = Path(tempfile.mkdtemp(prefix="AlbumForge_art_"))
            for img in list_images_in_dir(src_dir):
                try:
                    shutil.copy2(str(img), str(temp_dir / img.name))
                except Exception:
                    pass
            staged_src = temp_dir

        am.run(
            album_root=album_root,
            src_dir=staged_src,
            tracks=tracks,
            out_ext=cfg.get("image_ext","png"),
            lang=cfg.get("ocr_lang","eng"),
            psm=cfg.get("ocr_psm","6"),
            dry_run=getattr(args, "dry_run", False),
            force=getattr(args, "force", False),
            say=say
        )
        ok(f"Artwork pass done. ({mode.upper()})")
    finally:
        if temp_dir and temp_dir.exists():
            try: shutil.rmtree(temp_dir)
            except Exception: pass

def action_video(cfg:dict, args):
    album_root = find_album_root(Path(cfg["music_root"]).expanduser(), args.album)
    ensure_album_tree(album_root)

    # tiny settle to avoid FS race when we just wrote Artwork/
    time.sleep(0.2)

    # If artwork is present, run renderer; otherwise bail cleanly
    audio = list((album_root/"Audio").glob("*"))
    art   = list((album_root/"Artwork").glob("*"))
    if not audio:
        warn("No audio to render."); return
    if not art:
        warn("No artwork found; skipping video render."); return

    rc = rv.run(album_root, force=getattr(args, "force", False), say=say)
    if rc == 0:
        ok("Video render complete.")
    else:
        err(f"One or more renders failed (exit {rc}). Check the logs above.")

def action_status(cfg:dict, args):
    album_root = find_album_root(Path(cfg["music_root"]).expanduser(), args.album)
    ensure_album_tree(album_root)

    audio = list((album_root/"Audio").glob("*"))
    lyrics = list((album_root/"Lyrics").glob("*.txt"))
    art    = list((album_root/"Artwork").glob("*"))
    video  = list((album_root/"Video").glob("*.mp4"))

    def stems(paths): return {re.sub(r"\.[^.]+$","",p.name): p for p in paths}
    aS, lS, rS, vS = stems(audio), stems(lyrics), stems(art), stems(video)

    all_stems = sorted(set(aS)|set(lS)|set(rS)|set(vS))
    say(f"Album: {album_root.name}")
    say(f"  Audio {len(aS)} | Lyrics {len(lS)} | Artwork {len(rS)} | Video {len(vS)}")

    missing = []
    for s in all_stems:
        miss = []
        if s not in aS: miss.append("Audio")
        if s not in lS: miss.append("Lyrics")
        if s not in rS: miss.append("Artwork")
        if s not in vS: miss.append("Video")
        if miss:
            missing.append((s, miss))
    if missing:
        warn("Gaps:")
        for s, m in missing:
            say(f"  {s}: missing {', '.join(m)}")
        total = max(len(aS), len(lS), len(rS), len(vS), 1)
        pct = int(round(100 * (len(all_stems) - len(missing)) / len(all_stems))) if all_stems else 100
        say(f"\nCompletion: ~{pct}% (per-stem coverage)")
    else:
        ok("All good — everything matched.")

def action_make_lyrics(cfg:dict, args):
    album_root = find_album_root(Path(cfg["music_root"]).expanduser(), args.album)
    ensure_album_tree(album_root)
    audio_dir = album_root/"Audio"
    lyrics_dir = album_root/"Lyrics"
    made = 0
    for p in sorted(audio_dir.glob("*")):
        m = re.match(r"^\s*(\d{2})\s*-\s*(.+)\.[^.]+$", p.name)
        if not m: continue
        nn, title = m.group(1), m.group(2)
        lpath = lyrics_dir / f"{nn} - {title}.txt"
        if not lpath.exists():
            lpath.write_text("")
            say(f"Created lyrics: {lpath.relative_to(album_root)}")
            made += 1
    ok(f"Lyrics check complete. Created {made} file(s).")

def action_preview(cfg:dict, args):
    target = Path(args.image).expanduser()
    if not target.exists():
        err(f"No such image: {target}")
        return
    if not ti.preview(str(target)):
        warn("No inline protocol available. Install chafa or viu for text previews (recommended).")

def action_preview_artwork(cfg:dict, args):
    album_root = find_album_root(Path(cfg["music_root"]).expanduser(), args.album)
    arts = sorted((album_root/"Artwork").glob("*"))
    if not arts:
        warn("No artwork found.")
        return
    say("Choose an artwork to preview:")
    for i,p in enumerate(arts,1):
        say(f"  {i:2d}) {p.name}")
    try:
        pick = input("> ").strip()
    except EOFError:
        err("No input available."); return
    try:
        idx = int(pick) if pick.isdigit() else 1
        target = arts[idx-1]
        if not ti.preview(str(target)):
            warn("Preview failed. Try installing chafa or viu for fallback rendering.")
    except Exception as e:
        err(f"Failed: {e}")


# -----------
# Interactive
# -----------
def run_menu(cfg:dict):
    def ask(prompt:str, default:str|None=None) -> str:
        suffix = f" [{default}]" if default is not None else ""
        try:
            resp = input(f"{prompt}{suffix}: ").strip()
        except EOFError:
            return default if default is not None else ""
        return resp if resp else (default if default is not None else "")

    def ask_yn(prompt:str, default:bool=False) -> bool:
        d = "Y/n" if default else "y/N"
        try:
            resp = input(f"{prompt} ({d}): ").strip().lower()
        except EOFError:
            return default
        if not resp: return default
        return resp in ("y","yes")

    def maybe_clear():
        if os.environ.get("AlbumForge_NOCLEAR"): 
            return
        clear_screen()

    while True:
        maybe_clear()
        music_root = Path(cfg["music_root"]).expanduser()
        music_root.mkdir(parents=True, exist_ok=True)
        albums = sorted([p for p in music_root.iterdir()
                         if p.is_dir() and p.name != ".git" and not p.name.startswith(".")])

        say("=== AlbumForge ===")
        say(f"music_root: {music_root}")
        say("Albums:")
        for p in albums[:12]:
            a = len(list((p/"Audio").glob("*")))
            l = len(list((p/"Lyrics").glob("*.txt")))
            r = len(list((p/"Artwork").glob("*")))
            v = len(list((p/"Video").glob("*.mp4")))
            total = max(a, l, r, v, 1)
            pct = int(round(100 * min(a, l, r, v) / total))
            say(f"  - {p.name:32s}  A:{a:2d} L:{l:2d} R:{r:2d} V:{v:2d}  ({pct:3d}%)")
        say("\n1) Create album\n2) Manage album\n3) Preview artwork\n7) Doctor (check/install deps)\nq) Quit")
        try:
            choice = input("> ").strip().lower()
        except EOFError:
            # No TTY? Just exit gracefully.
            return

        if choice == "1":
            name = ask("Album name")
            if not name:
                say("No name given."); pause_then_clear(); continue
            action_create(cfg, argparse.Namespace(name=name, populate=False))
            do_pop = ask_yn("Auto-populate from a source folder?", default=False)
            if do_pop:
                src_default = str(Path(cfg["downloads_dir"]).expanduser())
                src = ask("Source folder", src_default)
                mode = ask("Ingest mode (copy/move)", "copy").lower()
                mode = mode if mode in ("copy","move") else "copy"
                action_create(cfg, argparse.Namespace(
                    name=name, populate=True, src=src, mode=mode, art_mode="copy", yes=False,
                    dry_run=False, force=False
                ))
                pause_then_clear(); continue

            if ask_yn("Move music into album now?", default=True):
                src_default = str(Path(cfg["downloads_dir"]).expanduser())
                src = ask("Source folder for audio", src_default)
                mode = ask("Ingest mode (copy/move)", "move").lower()
                mode = mode if mode in ("copy","move") else "move"
                action_move(cfg, argparse.Namespace(album=name, src=src, mode=mode, dry_run=False, force=False))

            if ask_yn("Match/move artwork now?", default=True):
                src_default = str(Path(cfg["downloads_dir"]).expanduser())
                src = ask("Source folder for artwork", src_default)
                amode = ask("Artwork ingest mode (copy/move)", "copy").lower()
                amode = amode if amode in ("copy","move") else "copy"
                action_art(cfg, argparse.Namespace(album=name, src=src, mode=amode, dry_run=False, force=False))

            if ask_yn("Create missing lyric files now?", default=True):
                action_make_lyrics(cfg, argparse.Namespace(album=name))

            if ask_yn("Generate videos now?", default=False):
                action_video(cfg, argparse.Namespace(album=name, force=False))

            action_status(cfg, argparse.Namespace(album=name))
            pause_then_clear(); continue

        elif choice == "2":
            manage_album_menu(cfg); 
            pause_then_clear(); 
            continue

        elif choice == "3":
            album = None
            action_preview_artwork(cfg, argparse.Namespace(album=album))
            pause_then_clear(); 
            continue

        elif choice == "7":
            doctor(True)
            pause_then_clear(); 
            continue

        elif choice in ("q","quit","exit"):
            return
        else:
            say("…unknown choice. Try again.")
            pause_then_clear(); 
            continue


# -------
# CLI
# -------
def main():
    cfg = load_config()

    # build parser (let argparse resolve duplicates just in case)
    ap = argparse.ArgumentParser(
        prog=APP,
        description=f"{APP_TITLE} — single-file album workflow orchestrator",
        conflict_handler="resolve",
    )
    ap.add_argument("-V", "--version", action="store_true", help="Print version and exit")
    ap.add_argument("--menu", action="store_true", help="Open interactive menu")
    
    # subcommands
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("doctor", help="Check environment and offer to install missing deps")

    sp = sub.add_parser("create", help="Create/ensure album tree (optionally auto-populate)")
    sp.add_argument("name")
    sp.add_argument("--populate", action="store_true", help="After create, ingest audio/artwork from --src (or downloads)")
    sp.add_argument("--src", help="Source folder for ingest (default: config downloads_dir)")
    sp.add_argument("--mode", choices=("copy","move"), default="copy", help="How to ingest audio during --populate (default: copy)")
    sp.add_argument("--art-mode", choices=("copy","move"), dest="art_mode", default="copy", help="How to ingest artwork during --populate (default: copy)")
    sp.add_argument("--yes", action="store_true", help="Skip confirmation (DANGEROUS). Implies you read the warning.")
    sp.add_argument("--dry-run", action="store_true", help="Plan only; do not change files")
    sp.add_argument("--force", action="store_true", help="Overwrite existing numbered targets where applicable")

    sp = sub.add_parser("move", help="Ingest audio into album and create empty Lyrics")
    sp.add_argument("--album", help="Album name")
    sp.add_argument("--src", help="Source folder (default: config downloads_dir)")
    sp.add_argument("--mode", choices=("copy","move"), default="move", help="Copy or move files into album (default: move)")
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("--force", action="store_true")

    sp = sub.add_parser("art", help="Match & ingest artwork (OCR+fuzzy)")
    sp.add_argument("--album", help="Album name")
    sp.add_argument("--src", help="Source folder for artwork (default: config downloads_dir)")
    sp.add_argument("--mode", choices=("copy","move"), default="copy", help="Copy or move artwork into album (default: copy)")
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("--force", action="store_true")

    sp = sub.add_parser("video", help="Generate static MP4s for an album")
    sp.add_argument("--album", help="Album name")
    sp.add_argument("--force", action="store_true")

    sp = sub.add_parser("status", help="Show album status")
    sp.add_argument("--album", help="Album name")

    sp = sub.add_parser("lyrics", help="Create any missing lyrics files for existing audio")
    sp.add_argument("--album", help="Album name")

    sp = sub.add_parser("preview", help="Preview an image (inline if supported; fallback to chafa/viu)")
    sp.add_argument("image", help="Path to an image (png/jpg/webp)")
    
    sp = sub.add_parser("youtube", help="YouTube auth & uploads")
    sub_yt = sp.add_subparsers(dest="yt_cmd")

    spa = sub_yt.add_parser("auth", help="Authenticate and store tokens")
    spa.add_argument("--profile", default="default")
    spa.add_argument("--client", help="Path to OAuth client_secret.json (Desktop app)")

    spw = sub_yt.add_parser("whoami", help="Show current channel")
    spw.add_argument("--profile", default="default")

    spu = sub_yt.add_parser("upload", help="Upload an album to YouTube (creates/updates playlist)")
    spu.add_argument("--album", help="Album name")
    spu.add_argument("--profile", default="default")
    spu.add_argument("--privacy", choices=("private","unlisted","public"), default="unlisted")


    # ---- parse once everything is declared ----
    args = ap.parse_args()

    if args.version:
        print(f"{APP_TITLE} v{VERSION}")
        return

    if args.menu or not args.cmd:
        run_menu(cfg)
        return

    # dispatch
    if args.cmd == "doctor":        doctor(True)
    elif args.cmd == "create":      action_create(cfg, args)
    elif args.cmd == "move":        action_move(cfg, args)
    elif args.cmd == "art":         action_art(cfg, args)
    elif args.cmd == "video":       action_video(cfg, args)
    elif args.cmd == "status":      action_status(cfg, args)
    elif args.cmd == "lyrics":      action_make_lyrics(cfg, args)
    elif args.cmd == "preview":     action_preview(cfg, args)
    elif args.cmd == "youtube":
        if args.yt_cmd == "auth":
            action_youtube_auth(cfg, args)
        elif args.yt_cmd == "whoami":
            try:
                say("Channel: " + yt.whoami(yt.client(getattr(args,"profile","default"))))
            except Exception as e:
                err(f"Failed: {e}")
        elif args.yt_cmd == "upload":
            action_youtube_upload(cfg, args)
        else:
            ap.print_help()

    else:
        ap.print_help()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
