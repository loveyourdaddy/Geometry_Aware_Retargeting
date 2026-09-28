"""Viewer that plays a list of motions in order."""
import glfw

from pymovis.vis.app import MyApp
from etc.etc import render_result
from test_utils.inference import run_network

PLAYBACK_SPEED = 0.8  # same playback speed MotionApp applies to the first motion
FRAME_JUMP = 10


class PlaylistApp(MyApp):
    """Play motion_pairs in order. ←/→: previous/next motion, ↑/↓: ±10 frames while paused."""

    def __init__(self, args, setup, motion_pairs):
        self.args = args
        self.setup = setup
        self.motion_pairs = motion_pairs
        self.index = 0
        self.cache = {}

        characters, motions = self.load(0, scale_fps=False)
        super().__init__(characters, motions, args, setup.net)
        self.show_title()

    def load(self, index, scale_fps=True):
        """(characters, motions) to render for the index-th motion. Inference runs only the first time."""
        if index not in self.cache:
            setup = self.setup
            name0, name1 = self.motion_pairs[index]
            print(f"> motion [{index + 1}/{len(self.motion_pairs)}]: {name0}")
            out_motion0, out_motion1, src_motion0, src_motion1 = run_network(
                self.args, setup, name0, name1)
            characters, motions = render_result(
                self.args,
                setup.src_char0, setup.src_char1, setup.tgt_char0, setup.tgt_char1,
                src_motion0, src_motion1, out_motion0, out_motion1,
            )
            if scale_fps:
                motions[0].fps = PLAYBACK_SPEED * motions[0].fps
            self.cache[index] = (characters, motions)
        return self.cache[index]

    def set_motion(self, index):
        self.index = index % len(self.motion_pairs)
        _, motions = self.load(self.index)
        self.motions = self.motion = motions
        self.example_motion = motions[0]
        self.frame = 0
        self.prev_frame = -1
        self.playing = True
        glfw.set_time(0)
        self.show_title()

    def show_title(self):
        name0, _ = self.motion_pairs[self.index]
        glfw.set_window_title(
            glfw.get_current_context(),
            f"[{self.index + 1}/{len(self.motion_pairs)}] {name0}")

    def jump_frame(self, offset):
        last = len(self.example_motion) - 1
        self.frame = min(max(self.frame + offset, 0), last)
        glfw.set_time(self.frame / self.example_motion.fps)

    def key_callback(self, window, key, scancode, action, mods):
        if key in (glfw.KEY_LEFT, glfw.KEY_RIGHT):
            if action == glfw.PRESS:
                self.set_motion(self.index + (1 if key == glfw.KEY_RIGHT else -1))
            return
        if key in (glfw.KEY_UP, glfw.KEY_DOWN):
            if action == glfw.PRESS and not self.playing:
                self.jump_frame(FRAME_JUMP if key == glfw.KEY_UP else -FRAME_JUMP)
            return
        super().key_callback(window, key, scancode, action, mods)

    def update(self):
        """Move to the next motion when one ends; stop after the last."""
        if self.playing and self.frame == len(self.example_motion) - 1:
            if self.index < len(self.motion_pairs) - 1:
                self.set_motion(self.index + 1)
            else:
                self.playing = False
