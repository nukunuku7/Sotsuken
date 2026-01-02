# editor/grid_utils.py
import os
import re
import time
import json
from datetime import datetime
from PyQt5.QtGui import QGuiApplication

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROFILE_DIR = os.path.join(ROOT_DIR, "config", "projector_profiles")
os.makedirs(PROFILE_DIR, exist_ok=True)

def log(msg: str):
    print(f"[DEBUG] {msg}")

def sanitize_filename(display_name: str, mode: str):
    """
    Produce stable filename from a normalized display identifier.
    We expect display_name to be a virtual id like 'D1' or a short token.
    Result: 'D1_perspective_points.json'
    """
    # keep it simple and predictable
    safe = re.sub(r'[^\w\-]+', '_', display_name)  # allow letters, numbers, underscore, hyphen
    safe = safe.strip("_")
    return f"{safe}_{mode}_points.json"


# --- 仮想IDルール: 左から順に D1, D2, ... を振る ---
def get_virtual_id(display_name: str) -> str:
    """
    Robust normalization:
      - If already 'D{n}' => return as-is
      - Try to extract a DISPLAY number from typical forms:
          '\\\\.\\DISPLAY1', '\\\\.\\DISPLAY2', 'DISPLAY1', 'Display 1', etc.
      - If not, fall back to mapping by left-to-right order via QGuiApplication.screens()
    Always returns a string (may fall back to original if nothing matched).
    """
    if not isinstance(display_name, str):
        return str(display_name)

    # if already virtual id
    if re.match(r"^D\d+$", display_name, re.IGNORECASE):
        return display_name.upper()

    # quick-clean copy to search for DISPLAY\d+
    cleaned = display_name.upper()
    # remove common escape sequences and dots for easier matching
    cleaned_simple = re.sub(r"[\\\.]", "", cleaned)  # remove "\" and "."
    m = re.search(r"DISPLAY\s*0*([1-9]\d*)", cleaned_simple)
    if m:
        return f"D{int(m.group(1))}"

    # try to find any trailing digits after word DISPLAY
    m2 = re.search(r"DISPLAY(\d+)", cleaned)
    if m2:
        return f"D{int(m2.group(1))}"

    # fallback: try to map by QGuiApplication order (left-to-right)
    try:
        screens = QGuiApplication.screens()
        if screens:
            ordered = sorted(screens, key=lambda s: s.geometry().x())
            for idx, s in enumerate(ordered, start=1):
                try:
                    if s.name() == display_name:
                        return f"D{idx}"
                except Exception:
                    continue
    except Exception:
        pass

    # last fallback: return sanitized form of the original name (safe)
    # but prefer to keep readable token
    fallback = re.sub(r'[^\w]+', '_', display_name).strip("_")
    return fallback or display_name

def virtual_to_screen_name(virt: str):
    """D1 -> QScreen.name() by left-to-right order. Returns None if not found."""
    if not isinstance(virt, str) or not re.match(r"^D\d+$", virt, re.IGNORECASE):
        return None
    try:
        screens = QGuiApplication.screens()
        if not screens:
            return None
        ordered = sorted(screens, key=lambda s: s.geometry().x())
        idx = int(virt[1:]) - 1
        if 0 <= idx < len(ordered):
            return ordered[idx].name()
    except Exception:
        pass
    return None

def get_point_path(display_name: str, mode: str = "perspective") -> str:
    """
    Normalize to virtual ID and return full path for JSON.
    """
    virt = get_virtual_id(display_name)
    filename = sanitize_filename(virt, mode)
    return os.path.join(PROFILE_DIR, filename)

def _alternate_filenames_for(display_name: str, mode: str):
    """
    Return a set of filename paths that might represent the "same" display,
    produced by older/buggy flows — used to clean up duplicates.
    We'll include:
      - sanitized(virt)
      - sanitized(original raw)
      - sanitized(stripped dots/backslashes version)
    """
    variants = set()
    virt = get_virtual_id(display_name)
    variants.add(os.path.join(PROFILE_DIR, sanitize_filename(virt, mode)))

    # raw sanitized
    raw_safe = re.sub(r'[^\w\-]+', '_', display_name).strip("_")
    if raw_safe:
        variants.add(os.path.join(PROFILE_DIR, f"{raw_safe}_{mode}_points.json"))

    # stripped form (remove backslashes and leading dots)
    cleaned = re.sub(r"[\\\.]+", "", display_name)
    cleaned_safe = re.sub(r'[^\w\-]+', '_', cleaned).strip("_")
    if cleaned_safe:
        variants.add(os.path.join(PROFILE_DIR, f"{cleaned_safe}_{mode}_points.json"))

    # also include legacy patterns starting with "._" or "__._"
    legacy_forms = []
    for v in list(variants):
        b = os.path.basename(v)
        if not b.startswith("._") and not b.startswith("__._"):
            legacy_forms.append(os.path.join(PROFILE_DIR, "._" + b))
            legacy_forms.append(os.path.join(PROFILE_DIR, "__._" + b))
    for lf in legacy_forms:
        variants.add(lf)

    return variants

def remove_alternate_files(display_name: str, mode: str):
    """
    Remove any alternate/gibberish filenames for the same display+mode,
    except the canonical virt-file which we will write.
    """
    virt = get_virtual_id(display_name)
    canonical = os.path.join(PROFILE_DIR, sanitize_filename(virt, mode))
    alts = _alternate_filenames_for(display_name, mode)
    for p in alts:
        # don't remove canonical
        if os.path.abspath(p) == os.path.abspath(canonical):
            continue
        try:
            if os.path.exists(p):
                os.remove(p)
                log(f"[CLEANUP] removed alternate file: {p}")
        except Exception as e:
            log(f"[WARN] failed to remove alt file {p}: {e}")

def save_points(display_name: str, points: list, mode: str = "perspective"):
    """
    Always normalize display_name to virtual ID and write canonical file.
    Before writing, remove alternate/legacy filenames that may have been
    created by older/buggy flows.
    """
    virt = get_virtual_id(display_name)
    path = os.path.join(PROFILE_DIR, sanitize_filename(virt, mode))

    # remove any weird duplicates before writing
    try:
        remove_alternate_files(display_name, mode)
    except Exception as e:
        log(f"[WARN] remove_alternate_files failed: {e}")

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(points, f, indent=2, ensure_ascii=False)
        log(f"[SAVE] saved points -> {path}")
    except Exception as e:
        log(f"[ERROR] Failed to save points: {e}")

def load_points(display_name: str, mode: str = "perspective"):
    """
    Load by normalized virtual id file (canonical).
    """
    virt = get_virtual_id(display_name)
    path = os.path.join(PROFILE_DIR, sanitize_filename(virt, mode))
    if not os.path.exists(path):
        log(f"[DEBUG] グリッドファイルが存在しません: {path}")
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        log(f"[LOAD] loaded points <- {path}")
        return data
    except Exception as e:
        log(f"[ERROR] Failed to load points: {e}")
        return None

# --- helper to get geometry by either PyQt name or virtual id ---
def get_screen_geometry(display_identifier: str):
    """
    If display_identifier is a virtual ID (Dn), map to screen by left-to-right order.
    If it's a PyQt name, find screen by name.
    Returns (width, height) tuple or (1920,1080) fallback.
    """
    try:
        screens = QGuiApplication.screens()
    except Exception:
        screens = []
    if not screens:
        return 1920, 1080
    ordered = sorted(screens, key=lambda s: s.geometry().x())

    # if virtual id
    if isinstance(display_identifier, str) and re.match(r"^D\d+$", display_identifier, re.IGNORECASE):
        try:
            idx = int(display_identifier[1:]) - 1
            if 0 <= idx < len(ordered):
                geo = ordered[idx].geometry()
                return geo.width(), geo.height()
        except Exception:
            pass

    # else attempt by PyQt name
    for s in ordered:
        try:
            if s.name() == display_identifier:
                geo = s.geometry()
                return geo.width(), geo.height()
        except Exception:
            continue

    return 1920, 1080

def generate_grid_points(display_name: str, divisions: int = 10) -> list:
    """
    Generate only BORDER POINTS inside a centered 10% rectangle.
    (外周 36 点, 位置は画面中央の幅・高さ 10% の範囲)
    """

    w, h = get_screen_geometry(display_name)

    # --- 中央10%の矩形を計算 ---
    scale = 0.10  # 10%
    cx, cy = w / 2, h / 2
    half_w, half_h = (w * scale) / 2, (h * scale) / 2

    left, right = cx - half_w, cx + half_w
    top, bottom = cy - half_h, cy + half_h

    # --- 外周36点を生成（generate_border_points を改造して利用） ---
    pts = []

    step = 1.0 / (divisions - 1)

    # top
    for i in range(divisions):
        x = left + (right - left) * (i * step)
        y = top
        pts.append([x, y])

    # right
    for i in range(1, divisions - 1):
        x = right
        y = top + (bottom - top) * (i * step)
        pts.append([x, y])

    # bottom
    for i in reversed(range(divisions)):
        x = left + (right - left) * (i * step)
        y = bottom
        pts.append([x, y])

    # left
    for i in reversed(range(1, divisions - 1)):
        x = left
        y = top + (bottom - top) * (i * step)
        pts.append([x, y])

    log(f"[INIT_BORDER] Display={display_name} Border10%Grid=({left:.1f},{top:.1f})~({right:.1f},{bottom:.1f}) → {len(pts)} points")
    return pts

def generate_perspective_points(display_name: str) -> list:
    """
    4-point rectangle in center. display_name can be virtualID or PyQt name.
    """
    w, h = get_screen_geometry(display_name)
    scale = 0.10
    cx, cy = w / 2, h / 2
    half_w, half_h = (w * scale) / 2, (h * scale) / 2

    points = [
        [cx - half_w, cy - half_h],
        [cx + half_w, cy - half_h],
        [cx + half_w, cy + half_h],
        [cx - half_w, cy + half_h],
    ]
    log(f"[INIT_4PT] Display={display_name}  Rect=({cx-half_w:.1f},{cy-half_h:.1f})~({cx+half_w:.1f},{cy+half_h:.1f})")
    return points

def create_display_grid(display_name: str, mode: str = "warp_map"):
    """
    Create initial grid only if file doesn't exist.
    display_name may be PyQt name or virtual id.
    """
    existing_points = load_points(display_name, mode)
    if existing_points:
        log(f"[SKIP_INIT] 既存グリッドを再利用: {display_name} ({len(existing_points)}点)")
        return existing_points

    if mode == "warp_map":
        points = generate_grid_points(display_name)
    else:
        points = generate_perspective_points(display_name)

    # Save canonical (will remove alternate legacy names)
    save_points(display_name, points, mode)
    log(f"✔ グリッド生成: {display_name} → {len(points)}点（モード: {mode}）")
    return points

def cleanup_old_profiles():
    """
    Tidy up clearly broken legacy filenames in PROFILE_DIR.
    We remove files that look like malformed artifacts such as starting with '._' or '__._',
    or known legacy patterns. This is safe to run repeatedly.
    """
    for f in os.listdir(PROFILE_DIR):
        if not f.endswith(".json"):
            continue
        old_path = os.path.join(PROFILE_DIR, f)

        # remove files beginning with '._' or '__._' that likely are artifacts
        base = os.path.basename(old_path)
        if base.startswith("._") or base.startswith("__._"):
            try:
                os.remove(old_path)
                log(f"[CLEANUP] removed malformed name: {base}")
            except Exception as e:
                log(f"[WARN] failed to remove {base}: {e}")
            continue

        # If filename contains a raw DISPLAY pattern (like DISPLAY1_perspective_points.json), remove it
        if re.search(r"DISPLAY\d+_.*_points\.json", base, re.IGNORECASE):
            try:
                os.remove(old_path)
                log(f"[CLEANUP] removed raw-display file: {base}")
            except Exception as e:
                log(f"[WARN] failed to remove {base}: {e}")
            continue

    log("🧹 ファイルクリーンアップ完了")

def auto_generate_from_environment(mode="warp_map", displays=None):
    """
    If displays is None, use non-primary screens.
    displays is a list of PyQt names or virtual IDs; create_display_grid will normalize.
    """
    try:
        screens = QGuiApplication.screens()
    except Exception:
        screens = []

    if not displays:
        primary = None
        try:
            primary = QGuiApplication.primaryScreen().name()
        except Exception:
            primary = None
        displays = [s.name() for s in screens if s.name() != primary]

    cleanup_old_profiles()
    for name in displays:
        create_display_grid(name, mode)
    print(f"🎉 選択ディスプレイ（{len(displays)}台）のグリッドを生成完了。")

# temp dir for coordination (file-based IPC between editor processes)
TEMP_DIR = os.path.join(ROOT_DIR, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

def _atomic_write(path, data):
    """Atomic write helper for small text/JSON files."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(data)
    os.replace(tmp, path)

def _read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _write_json(path, obj):
    _atomic_write(path, json.dumps(obj, ensure_ascii=False, indent=2))

# queue file path (left-to-right order of virtual ids)
def _queue_path():
    return os.path.join(TEMP_DIR, "editor_queue.json")

def _session_path(virt):
    return os.path.join(TEMP_DIR, f"editor_session_{virt}.json")

def create_editor_queue(pyqt_names):
    """
    Create the editor queue file from a list of PyQt display names.
    The function normalizes to virtual IDs (D1...) and writes them left->right.
    Overwrites any existing queue.
    """
    virts = []
    for name in pyqt_names:
        v = get_virtual_id(name)
        virts.append(v)
    # ensure order left-to-right by mapping using QGuiApplication screens
    try:
        screens = QGuiApplication.screens() or []
        ordered = sorted(screens, key=lambda s: s.geometry().x())
        ordered_names = [s.name() for s in ordered]
        # sort virts by left-to-right according to screen order
        def keyfn(v):
            sn = virtual_to_screen_name(v) or v
            try:
                return ordered_names.index(sn)
            except ValueError:
                return 9999
        virts = sorted(list(dict.fromkeys(virts)), key=keyfn)
    except Exception:
        pass

    queue = {
        "queue": virts,
        "created_at": datetime.utcnow().isoformat()
    }
    _write_json(_queue_path(), queue)
    log(f"[QUEUE] created queue: {virts}")
    return virts

def read_editor_queue():
    q = _read_json(_queue_path())
    if not q:
        return []
    return q.get("queue", [])

def init_editor_session(virt, total_points):
    """
    Create a session file for a given virt.
    Fields:
      - virt
      - total_points
      - current_index  (next point index to be claimed; starts from 0)
      - done_count     (how many points done)
      - finished       (bool)
      - last_update
    """
    path = _session_path(virt)
    sess = {
        "virt": virt,
        "total_points": int(total_points),
        "current_index": 0,
        "done_count": 0,
        "finished": False,
        "last_update": datetime.utcnow().isoformat()
    }
    _write_json(path, sess)
    log(f"[SESSION] init {virt} (total={total_points})")
    return sess

def get_session(virt):
    path = _session_path(virt)
    return _read_json(path)

def claim_next_point(virt):
    """
    Try to claim the next point index for this virt.
    Returns:
      - integer index (0-based) if this editor is allowed to take the next point now
      - None if it's not this editor's turn (queue prevents it)
    Behavior:
      - Only the virt at the head of the queue may claim points.
      - Within its session, it will receive current_index and current_index will be left unchanged
        until mark_point_done is called (so the editor knows which index to drag).
    """
    queue = read_editor_queue()
    if not queue:
        # no queue -> allow immediate claiming (single-editor mode)
        q_head = None
    else:
        q_head = queue[0]

    if q_head and q_head != virt:
        # not our turn
        return None

    sess = get_session(virt)
    if not sess:
        return None

    if sess.get("finished"):
        return None

    cur = int(sess.get("current_index", 0))
    total = int(sess.get("total_points", 0))
    if cur >= total:
        # nothing left
        return None

    # return current index but do NOT advance it here;
    # advance happens only when mark_point_done is called.
    return cur

def mark_point_done(virt, index):
    """
    Called by editor when the editing of index is completed (on_release).
    Advances done_count and current_index.
    If all points done, marks finished and pops virt from queue.
    """
    path = _session_path(virt)
    sess = get_session(virt)
    if not sess:
        log(f"[SESSION] mark_point_done: no session for {virt}")
        return

    try:
        total = int(sess.get("total_points", 0))
        cur = int(sess.get("current_index", 0))
        done = int(sess.get("done_count", 0))
    except Exception:
        log(f"[ERROR] invalid session values for {virt}")
        return

    # Ensure index matches expected (best-effort)
    if index != cur:
        # allow out-of-order but log
        log(f"[SESSION] warning: {virt} reported done index {index} but current_index={cur}")

    done += 1
    cur += 1

    sess["done_count"] = done
    sess["current_index"] = cur
    sess["last_update"] = datetime.utcnow().isoformat()
    if cur >= total:
        sess["finished"] = True

    _write_json(path, sess)
    log(f"[SESSION] {virt} done index={index} -> cur={cur}/{total}")

    # If finished, advance queue
    if sess.get("finished"):
        _advance_queue_after_finish(virt)

def _advance_queue_after_finish(virt):
    """Internal: remove virt from head of queue (if present) when finished."""
    p = _queue_path()
    q = _read_json(p)
    if not q:
        return
    queue = q.get("queue", [])
    if queue and queue[0] == virt:
        queue = queue[1:]
        q["queue"] = queue
        q["last_update"] = datetime.utcnow().isoformat()
        _write_json(p, q)
        log(f"[QUEUE] popped finished virt: {virt} -> next: {queue[:3]}")

def mark_session_finished(virt):
    """Force mark session as finished and advance queue."""
    sess = get_session(virt)
    if not sess:
        return
    sess["finished"] = True
    sess["last_update"] = datetime.utcnow().isoformat()
    _write_json(_session_path(virt), sess)
    _advance_queue_after_finish(virt)

def cleanup_sessions_and_queue():
    """
    Helper to remove stale session/queue files older than some threshold.
    (Not called automatically — call from main for housekeeping if desired.)
    """
    now = time.time()
    threshold = 60 * 60 * 24  # 1 day
    # sessions
    for f in os.listdir(TEMP_DIR):
        if f.startswith("editor_session_") or f == "editor_queue.json":
            p = os.path.join(TEMP_DIR, f)
            try:
                m = os.path.getmtime(p)
                if now - m > threshold:
                    os.remove(p)
                    log(f"[CLEANUP] removed stale coordinator file: {f}")
            except Exception:
                pass
