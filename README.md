# 🎶 Album Forge  
*A tiny orchestrator for your album workflow.*  

![Demo header](images/01.png)  

---

## What is this?  
Album Forge takes your messy **Downloads folder full of songs, artwork, and chaos**, and transforms it into a clean, album-ready structure:  
```
Album/
├── Audio/NN - Title.ext
├── Lyrics/NN - Title.txt
├── Artwork/NN - Title.png
├── Video/NN - Title.mp4
├── Notes/
└── .AlbumForge.json
```


It:  
- Moves audio → `Audio/` with proper numbering  
- Creates empty lyric stubs → `Lyrics/`  
- Matches artwork → `Artwork/` (OCR + fuzzy magic)  
- Generates static MP4s → `Video/` (with ffmpeg)  
- Reports what’s missing with `status`  

Think: *Downloads gremlin → tidy album wizard.* 

---

## Features  
- `create` — scaffold a new album folder  
- `move` — relocate audio from `~/Downloads` → album, generate lyric stubs  
- `art` — OCR + fuzzy match images to track titles  
- `video` — render static MP4s for YouTube  
- `status` — check if you’ve got Audio/Lyrics/Artwork/Video for every track  
- `--menu` — friendly interactive menu for the indecisive  

---

## Installation  

Clone this repo, then:  

```bash
pip install -r requirements.txt
# or run directly with
python -m AlbumForge --menu
```

For convenience, `bin/AlbumForge` is a tiny launcher script. Add `bin/` to your `$PATH` or symlink it into `~/.local/bin`.  

---

## Configuration  

First run will generate:  
`~/.config/AlbumForge/config.toml`  

Example:  

```toml
music_root = "/home/you/Music"
downloads_dir = "/home/you/Downloads"
ocr_lang = "eng"
ocr_psm = "6"
image_ext = "png"
freshness_minutes = 0
editor = "nano"
```

---

## Workflow  

1. **Create an album**  
```bash
AlbumForge create "My Album"
```  

![Menu demo](images/02.png)  

2. **Move audio & create lyric stubs**  
```bash
AlbumForge move --album "My Album"
```  

Empty text files appear in `Lyrics/`, ready for your words.  

![Created album](images/03.png)  

3. **Drop artwork into Downloads → match to tracks**  
```bash
AlbumForge art --album "My Album"
```  

OCR + fuzzy finds the best fit, places `NN - Title.png` in `Artwork/`.  

![Lyrics stub demo](images/04.png)  

4. **Generate MP4s**  
```bash
AlbumForge video --album "My Album"
```  

ffmpeg stitches artwork + audio into `Video/NN - Title.mp4`.  

![Artwork demo](images/05.png)  

5. **Check status**  
```bash
AlbumForge status --album "My Album"
```  

Quick report on missing pieces.  

![Video demo](images/06.png)  

---

## Cheatsheet  

```bash
AlbumForge -V          # show version
AlbumForge --menu      # interactive menu
AlbumForge create NAME
AlbumForge move   --album NAME
AlbumForge art    --album NAME
AlbumForge video  --album NAME
AlbumForge status --album NAME
```

---

## Dependencies  
- Python 3.11+  
- ffmpeg (for video)  
- Tesseract OCR + language packs (for artwork)  

---

## Why?  
Because wrangling albums by hand is boring.  
This script makes it fun, fast, and a little bit magical. 

---

