import threading
import time
from datetime import datetime
from pathlib import Path

from pose_hud.capture import WindowsVRChatCapture
from pose_hud.cli._keys import key_events
from pose_hud.mapping import RoomMapper
from pose_hud.mapping_render import render_map
from pose_hud.reader import PoseReader


def _key_thread(pause_evt: threading.Event, stop_evt: threading.Event) -> None:
    for ch in key_events():
        if ch == " ":
            (pause_evt.clear if pause_evt.is_set() else pause_evt.set)()
        elif ch in ("q", "Q", "\x1b", "\x03"):  # q / ESC / Ctrl+C
            stop_evt.set()
            return


def main() -> None:
    reader = PoseReader(source=WindowsVRChatCapture())
    mapper = RoomMapper()
    pause_evt = threading.Event()
    stop_evt = threading.Event()

    reader.start()
    threading.Thread(
        target=_key_thread, args=(pause_evt, stop_evt), daemon=True
    ).start()
    print("recording... SPACE=一時停止/再開  q=保存して終了")

    last_t = None
    last_report = time.monotonic()
    was_paused = False
    try:
        while not stop_evt.is_set():
            pose = reader.get_latest()
            if pose is not None and pose.time_ms != last_t:
                last_t = pose.time_ms
                if pause_evt.is_set():
                    if not was_paused:
                        mapper.break_segment()
                        was_paused = True
                else:
                    was_paused = False
                    mapper.add_pose(pose)
            now = time.monotonic()
            if now - last_report >= 1.0:
                w, d = mapper.dimensions()
                state = "PAUSED" if pause_evt.is_set() else "rec   "
                print(
                    f"  [{state}] pts={len(mapper):5d}  seg={mapper.num_segments}  "
                    f"bbox={w:5.2f}x{d:5.2f}m  path={mapper.path_length():6.2f}m"
                )
                last_report = now
            time.sleep(0.005)
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()

    if len(mapper) == 0:
        print("no trajectory collected; nothing saved.")
        return

    out_dir = Path("maps") / datetime.now().strftime("%Y%m%d_%H%M%S")
    npz = mapper.save(out_dir / "room")
    s = mapper.to_dict()
    print(
        f"\nroom: {s['width_x_m']:.2f} x {s['depth_z_m']:.2f} m  "
        f"(path {s['path_length_m']:.2f} m, {s['points']} pts, {s['segments']} seg)"
    )
    print(f"saved: {npz}  {npz.with_suffix('.json')}")
    print(f"map:   {render_map(mapper, out_dir / 'room')}")


if __name__ == "__main__":
    main()
