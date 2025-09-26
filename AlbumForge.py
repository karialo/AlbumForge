#!/usr/bin/env python3
# AlbumForge — Monolith v0.6.0
# Single-file tool for album workflows:
# - create/move/art/video/status/lyrics/preview + doctor (deps check/install)
# - Inline modules: term_images, render, artmatch
from __future__ import annotations
import argparse, json, os, random, re, shutil, subprocess, sys, shlex, tempfile, base64
from pathlib import Path
from datetime import datetime, timezone

APP = "albumforge"
APP_TITLE = "AlbumForge"
VERSION = "0.6.0"

# =========================
# Embedded modules (sources)
# =========================

_EMBED_TERM_IMAGES = r"""
# term_images (embedded)
from __future__ import annotations
import os, shutil, subprocess, base64, sys, os.path as op

def have(cmd:str)->bool: return shutil.which(cmd) is not None
def is_wezterm()->bool:
    return os.environ.get("TERM_PROGRAM") == "WezTerm" or bool(os.environ.get("WEZTERM_EXECUTABLE") or os.environ.get("WT_SESSION"))

def preview(path:str)->bool:
    '''
    Try (in order): Kitty graphics, iTerm2/WezTerm OSC 1337, SIXEL, chafa, viu.
    Returns True if *something* was shown.
    '''
    path = op.abspath(path)
    # 1) Kitty protocol (only renders if terminal supports it)
    if have("kitty"):
        try:
            subprocess.run(["kitty","+kitten","icat","--place","60x24@0x0",path], check=True)
            return True
        except Exception:
            pass
    # 2) iTerm2/WezTerm OSC 1337
    if is_wezterm():
        try:
            with open(path,"rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            sys.stdout.write(f"\\033]1337;File=name={op.basename(path)};inline=1;size={len(b64)}:{b64}\\a\\n")
            sys.stdout.flush()
            return True
        except Exception:
            pass
    # 3) SIXEL
    if have("img2sixel"):
        try:
            subprocess.run(["img2sixel", path], check=True)
            return True
        except Exception:
            pass
    # 4) Fallbacks — work everywhere
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
import shutil, subprocess, re, json, os
from pathlib import Path

AUDIO_EXTS = {".mp3", ".m4a", ".flac", ".wav", ".ogg", ".opus"}
IMG_EXTS   = {".png", ".jpg", ".jpeg", ".webp"}

def have(cmd:str) -> bool:
    return shutil.which(cmd) is not None

def _stem(p:Path) -> str:
    return re.sub(r"\\.[^.]+$","",p.name)

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
    if len(text) <= maxlen:
        return text
    if maxlen <= 1:
        return text[:maxlen]
    head = max(1, (maxlen - 1) // 2)
    tail = max(0, maxlen - head - 1)
    return text[:head] + "…" + text[-tail:] if tail else text[:head] + "…"

def _progress_line(title:str, audio:Path, out:Path, img:Path, ffmpeg_args:list[str], say=print) -> int:
    '''Run ffmpeg with -progress pipe:1, render single-line adaptive bar.'''
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
        pct  = 0.0 if pct < 0 else (100.0 if pct > 100.0 else pct)
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
            print("\\r" + line[:cols-1] + "\\x1b[K", end="", flush=True)
        else:
            print(line[:cols-1], flush=True)

    try:
        for line in proc.stdout:
            line = line.strip()
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
            print()
    return proc.returncode

def run(album_root:Path, *, force:bool=False, say=print):
    if not have("ffmpeg"):
        raise SystemExit("❌ ffmpeg not found in PATH.")
    audio_dir = album_root/"Audio"
    art_dir   = album_root/"Artwork"
    out_dir   = album_root/"Video"
    out_dir.mkdir(parents=True, exist_ok=True)
    audios = [p for p in audio_dir.glob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
    arts   = {_stem(p): p for p in art_dir.glob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS}
    if not audios:
        say("❌ No audio files found."); return 1
    for a in sorted(audios):
        stem = _stem(a)
        img = arts.get(stem)
        if not img:
            say(f"⚠️  No artwork for {a.name}, skipping."); continue
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
            say(f"❌ ffmpeg failed ({code}) for {stem}")
    say("✨ All done.")
    return 0
"""

_EMBED_ARTMATCH = r"""
# artmatch (embedded) — OCR + fuzzy best-fit image→track
from __future__ import annotations
import re, shutil, subprocess
from pathlib import Path
from difflib import SequenceMatcher

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

def have(cmd:str)->bool: return shutil.which(cmd) is not None

def _norm(s:str)->str:
    s = s.lower()
    s = re.sub(r"[^\\w\\s]", " ", s)
    s = re.sub(r"\\s+", " ", s).strip()
    return s

def _sim(a:str,b:str)->float:
    A, B = set(_norm(a).split()), set(_norm(b).split())
    token = 2*len(A&B)/(len(A)+len(B)) if A and B else 0.0
    ratio = SequenceMatcher(None, _norm(a), _norm(b)).ratio()
    return 0.6*ratio + 0.4*token

def _ocr_text(img:Path, lang:str, psm:str|None, timeout:int=8)->str:
    if not have("tesseract"): return ""
    cmd = ["tesseract", str(img), "stdout", "-l", lang]
    if psm: cmd += ["--psm", psm]
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=timeout).strip()
    except Exception:
        return ""

def plan_matches(images:list[Path], tracks:list[str], lang:str="eng", psm:str|None="6"):
    scores = []
    for img in images:
        combo = img.stem
        txt = _ocr_text(img, lang, psm)
        if txt: combo = f"{txt} {combo}"
        row = [_sim(combo, t) for t in tracks]
        scores.append(row)
    assigned = []
    used_t = set()
    for i, row in enumerate(scores):
        best_j, best_sc = None, -1.0
        for j, sc in enumerate(row):
            if j in used_t: continue
            if sc > best_sc:
                best_j, best_sc = j, sc
        if best_j is not None:
            assigned.append((i, best_j, best_sc))
            used_t.add(best_j)
    assigned.sort(key=lambda x: x[1])
    return assigned, scores

def run(album_root:Path, src_dir:Path, tracks:list[tuple[str,str]], *,  # (NN, Title)
        out_ext:str="png", lang:str="eng", psm:str|None="6",
        dry_run:bool=False, force:bool=False, say=print):
    out_dir = album_root/"Artwork"
    out_dir.mkdir(parents=True, exist_ok=True)
    images = sorted([p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])
    if not images:
        say("❌ No images found in source directory."); return 1
    track_titles = [title for _, title in tracks]
    pairs, _ = plan_matches(images, track_titles, lang=lang, psm=psm)
    say("Plan:")
    for i, j, sc in pairs:
        nn, title = tracks[j]
        dst = out_dir / f"{nn} - {title}.{out_ext}"
        say(f"  {images[i].name}  →  {dst.relative_to(album_root)}  [score {sc:.2f}]")
    if dry_run:
        say("DRY-RUN: no files moved.")
        return 0
    for i, j, sc in pairs:
        nn, title = tracks[j]
        dst = out_dir / f"{nn} - {title}.{out_ext}"
        if dst.exists() and not force:
            say(f"⏭️  Exists: {dst.name}"); continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(images[i]), str(dst))
        say(f"✓ {images[i].name} → {dst.relative_to(album_root)}")
    return 0
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

def clear_screen():
    try:
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

def sanitize_title(name:str)->str:
    name = Path(name).stem
    name = re.sub(r"^\s*\d+\s*[-_.]\s*", "", name)
    name = name.replace("_", " ").strip()
    name = re.sub(r"\s+", " ", name)
    return name.title()

def have(cmd:str)->bool:
    return shutil.which(cmd) is not None

def is_wezterm()->bool:
    return os.environ.get("TERM_PROGRAM") == "WezTerm" or bool(os.environ.get("WEZTERM_EXECUTABLE") or os.environ.get("WT_SESSION"))

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
        # run via shell to chain update+install
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

def doctor(interactive:bool=True) -> int:
    ok("Doctor: scanning environment…")
    missing = []
    need = {
        "ffmpeg": not have("ffmpeg"),
        "tesseract": not have("tesseract"),
        "chafa": not have("chafa"),
        "viu": not have("viu"),
    }
    for k, v in need.items():
        if v and k in ("ffmpeg","tesseract"):
            missing.append(k)

    if not missing:
        ok("Core OK: ffmpeg & tesseract detected.")
    else:
        warn("Missing core tools: " + ", ".join(missing))

    opt_missing = [k for k in ("chafa","viu") if need[k]]
    if opt_missing:
        warn("Optional tools missing (for nicer previews): " + ", ".join(opt_missing))
    else:
        ok("Optional preview tools present.")

    if interactive and missing:
        pm = _detect_pkgmgr()
        if not pm:
            warn("No known package manager detected (pacman/apt/dnf/zypper/apk/brew).")
            return 1
        pkgs = []
        for m in missing:
            alias = PKG_ALIASES.get(m, {})
            pkgs.append(alias.get(pm, m))
        say(f"\nDetected package manager: {pm}")
        say("I can install: " + " ".join(pkgs))
        yn = input("Proceed with install? (y/N): ").strip().lower()
        if yn not in ("y","yes"):
            warn("Skipped installation.")
            return 1
        cmd = _install_cmd(pm, pkgs)
        if not cmd:
            warn("Unsupported installer on this system.")
            return 1
        try:
            if pm == "apt":
                shell_cmd = " ".join(cmd)
                rc = subprocess.call(shell_cmd, shell=True)
            else:
                rc = subprocess.call(cmd)
            if rc == 0:
                ok("Install complete. Re-run doctor if needed.")
            else:
                err(f"Installer returned {rc}.")
                return rc
        except Exception as e:
            err(f"Install failed: {e}")
            return 1
    return 0

# -----------------------
# Core helpers & actions
# -----------------------
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
    choice = input("> ").strip()
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
        import time
        cutoff = time.time() - (freshness_minutes * 60)
        files = [p for p in files if p.stat().st_mtime >= cutoff]
    return sorted(files)

def list_images_in_dir(path: Path) -> list[Path]:
    return sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])

def action_create(cfg:dict, args):
    music_root = Path(cfg["music_root"]).expanduser()
    music_root.mkdir(parents=True, exist_ok=True)
    target = music_root/args.name
    if target.exists():
        say(f"Album exists: {target}")
    else:
        ensure_album_tree(target)
        ok(f"Created album: {target}")
    state = load_state(target)
    save_state(target, state)

    if getattr(args, "populate", False):
        src = Path(getattr(args, "src", "") or cfg["downloads_dir"]).expanduser()
        aud = list_audio_in_dir(src, cfg.get("freshness_minutes", 0))
        imgs = list_images_in_dir(src)
        mode = getattr(args, "mode", "copy")
        say("\n" + "="*72)
        warn(" ⚠️  AUTO-POPULATE IS ABOUT TO RUN ")
        say(f" Source folder : {src}")
        say(f" Audio files   : {len(aud)}")
        say(f" Image files   : {len(imgs)} (used by 'art')")
        say(f" Mode          : {mode.upper()} (audio ingest)")
        say("="*72 + "\n")
        if not getattr(args, "yes", False):
            say("MAKE SURE YOU HAVE ALL SONGS AND ARTWORK IN THE SOURCE FOLDER.")
            say("Type YES (all caps) to continue, anything else to abort.")
            if input("> ").strip() != "YES":
                warn("Aborted by user. Album created, nothing ingested.")
                return
        action_move(cfg, argparse.Namespace(
            album=args.name, src=str(src),
            dry_run=getattr(args, "dry_run", False),
            force=getattr(args, "force", False),
            mode=mode
        ))
        art_mode = getattr(args, "art_mode", "copy")
        action_art(cfg, argparse.Namespace(
            album=args.name, src=str(src),
            mode=art_mode,
            dry_run=getattr(args, "dry_run", False),
            force=getattr(args, "force", False)
        ))
        action_make_lyrics(cfg, argparse.Namespace(album=args.name))
        album_root = find_album_root(Path(cfg["music_root"]).expanduser(), args.name)
        has_art = any((album_root/"Artwork").glob("*"))
        if has_art:
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
    existing_nn = [re.match(r"^\s*(\d{2})\s*-\s*", p.name).group(1) for p in existing_audios if re.match(r"^\s*(\d{2})\s*-\s*", p.name)]
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
        existing = {t["n"]: t for t in state["tracks"]}
        existing[nn] = {"n": nn, "title": title, "audio": dst.name, "lyrics": f"{nn} - {title}.txt"}
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
            temp_dir = Path(tempfile.mkdtemp(prefix="albumforge_art_"))
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
    rv.run(album_root, force=getattr(args, "force", False), say=say)

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
    pick = input("> ").strip()
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
        resp = input(f"{prompt}{suffix}: ").strip()
        return resp if resp else (default if default is not None else "")

    def ask_yn(prompt:str, default:bool=False) -> bool:
        d = "Y/n" if default else "y/N"
        resp = input(f"{prompt} ({d}): ").strip().lower()
        if not resp: return default
        return resp in ("y","yes")

    while True:
        clear_screen()
        music_root = Path(cfg["music_root"]).expanduser()
        music_root.mkdir(parents=True, exist_ok=True)
        albums = sorted([p for p in music_root.iterdir()
                         if p.is_dir() and p.name != ".git" and not p.name.startswith(".")])

        say(f"=== {APP_TITLE} ===")
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
        say("\n1) Create album\n2) Move music\n3) Generate artwork\n4) Generate videos\n5) Status\n6) Preview artwork\n7) Doctor (check/install deps)\nq) Quit")
        choice = input("> ").strip().lower()

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
            album = ask("Album (blank to pick recent)") or None
            src_default = str(Path(cfg["downloads_dir"]).expanduser())
            src = ask("Source folder for audio", src_default)
            mode = ask("Ingest mode (copy/move)", "move").lower()
            mode = mode if mode in ("copy","move") else "move"
            action_move(cfg, argparse.Namespace(album=album, src=src, mode=mode, dry_run=False, force=False))
            pause_then_clear(); continue

        elif choice == "3":
            album = ask("Album (blank to pick recent)") or None
            src_default = str(Path(cfg["downloads_dir"]).expanduser())
            src = ask("Source folder for artwork", src_default)
            amode = ask("Artwork ingest mode (copy/move)", "copy").lower()
            amode = amode if amode in ("copy","move") else "copy"
            action_art(cfg, argparse.Namespace(album=album, src=src, mode=amode, dry_run=False, force=False))
            pause_then_clear(); continue

        elif choice == "4":
            album = ask("Album (blank to pick recent)") or None
            action_video(cfg, argparse.Namespace(album=album, force=False))
            pause_then_clear(); continue

        elif choice == "5":
            album = ask("Album (blank to pick recent)") or None
            action_status(cfg, argparse.Namespace(album=album))
            pause_then_clear(); continue

        elif choice == "6":
            album = ask("Album (blank to pick recent)") or None
            action_preview_artwork(cfg, argparse.Namespace(album=album))
            pause_then_clear(); continue

        elif choice == "7":
            doctor(True)
            pause_then_clear(); continue

        elif choice in ("q","quit","exit"):
            return
        else:
            say("…unknown choice. Try again.")
            pause_then_clear(); continue

# -------
# CLI
# -------
def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(prog=APP, description=f"{APP_TITLE} — single-file album workflow orchestrator")
    ap.add_argument("-V","--version", action="store_true", help="Print version and exit")
    ap.add_argument("--menu", action="store_true", help="Open interactive menu")

    sub = ap.add_subparsers(dest="cmd")

    sp = sub.add_parser("doctor", help="Check environment and offer to install missing deps")

    sp = sub.add_parser("create", help="Create/ensure album tree (optionally auto-populate)")
    sp.add_argument("name")
    sp.add_argument("--populate", action="store_true", help="After create, ingest audio/artwork from --src (or downloads)")
    sp.add_argument("--src", help="Source folder for ingest (default: config downloads_dir)")
    sp.add_argument("--mode", choices=("copy","move"), default="copy", help="How to ingest audio during --populate (default: copy)")
    sp.add_argument("--art-mode", choices=("copy","move"), dest="art_mode", default="copy", help="How to ingest artwork during --populate (default: copy)")
    sp.add_argument("--yes", action="store_true", help='Skip confirmation (DANGEROUS). Implies you read the warning.')
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

    args = ap.parse_args()
    if args.version:
        print(f"{APP_TITLE} v{VERSION}"); return
    if args.menu or not args.cmd:
        run_menu(cfg); return

    if args.cmd == "doctor":    doctor(True)
    elif args.cmd == "create":  action_create(cfg, args)
    elif args.cmd == "move":    action_move(cfg, args)
    elif args.cmd == "art":     action_art(cfg, args)
    elif args.cmd == "video":   action_video(cfg, args)
    elif args.cmd == "status":  action_status(cfg, args)
    elif args.cmd == "lyrics":  action_make_lyrics(cfg, args)
    elif args.cmd == "preview": action_preview(cfg, args)
    else:
        ap.print_help()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
