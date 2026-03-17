"""
Module de visualisation 3D de points lumineux
=============================================

Version dédiée à la mesure des temps de rendu pour l'étape "Mesure du temps initial".
Le fichier original visualizer3d.py est laissé intact.
"""

import ctypes
import time

import numpy as np
import sdl2
import sdl2.ext
from OpenGL.GL import *
from OpenGL.GLU import *


class Visualizer3D:
    def __init__(self, points, colors, luminosities, bounds, visible=True, vsync=True):
        self.points = np.array(points, dtype=np.float32)
        self.colors = np.array(colors, dtype=np.float32)
        self.luminosities = np.array(luminosities, dtype=np.float32)
        self.bounds = bounds

        self.window_width = 1024
        self.window_height = 768
        self.window = None
        self.gl_context = None
        self.visible = visible
        self.vsync = vsync

        self.camera_distance = 5.0
        self.camera_rotation_x = 0.0
        self.camera_rotation_y = 0.0
        self.zoom_factor = 1.0

        self.mouse_dragging = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.mouse_sensitivity = 0.3

        self.running = False

        self.vbo_vertices = None
        self.vbo_colors = None
        self.vbo_needs_update = True

        self.center = np.array(
            [
                (bounds[0][0] + bounds[0][1]) / 2.0,
                (bounds[1][0] + bounds[1][1]) / 2.0,
                (bounds[2][0] + bounds[2][1]) / 2.0,
            ],
            dtype=np.float32,
        )

        self.scale = max(
            bounds[0][1] - bounds[0][0],
            bounds[1][1] - bounds[1][0],
            bounds[2][1] - bounds[2][0],
        )

        self._init_sdl()
        self._init_opengl()
        self._init_vbo()

    def _init_sdl(self):
        if sdl2.SDL_Init(sdl2.SDL_INIT_VIDEO) != 0:
            raise RuntimeError(f"Erreur SDL_Init: {sdl2.SDL_GetError()}")

        sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_CONTEXT_MAJOR_VERSION, 2)
        sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_CONTEXT_MINOR_VERSION, 1)
        sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_DOUBLEBUFFER, 1)
        sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_DEPTH_SIZE, 24)

        window_flags = sdl2.SDL_WINDOW_OPENGL
        if self.visible:
            window_flags |= sdl2.SDL_WINDOW_SHOWN
        else:
            window_flags |= sdl2.SDL_WINDOW_HIDDEN

        self.window = sdl2.SDL_CreateWindow(
            b"Visualisation 3D - Points Lumineux",
            sdl2.SDL_WINDOWPOS_CENTERED,
            sdl2.SDL_WINDOWPOS_CENTERED,
            self.window_width,
            self.window_height,
            window_flags,
        )

        if not self.window:
            raise RuntimeError(f"Erreur création fenêtre: {sdl2.SDL_GetError()}")

        self.gl_context = sdl2.SDL_GL_CreateContext(self.window)
        if not self.gl_context:
            raise RuntimeError(f"Erreur création contexte GL: {sdl2.SDL_GetError()}")

        sdl2.SDL_GL_SetSwapInterval(1 if self.vsync else 0)

    def _init_opengl(self):
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        glEnable(GL_POINT_SMOOTH)
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
        glPointSize(3.0)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect_ratio = self.window_width / self.window_height
        gluPerspective(45.0, aspect_ratio, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def _init_vbo(self):
        self.vbo_vertices = glGenBuffers(1)
        self.vbo_colors = glGenBuffers(1)
        self._update_vbo()

    def _update_vbo(self):
        colors_with_luminosity = (
            self.colors * self.luminosities[:, np.newaxis] / 255.0
        ).astype(np.float32)

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_vertices)
        glBufferData(GL_ARRAY_BUFFER, self.points.nbytes, self.points, GL_DYNAMIC_DRAW)

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_colors)
        glBufferData(
            GL_ARRAY_BUFFER,
            colors_with_luminosity.nbytes,
            colors_with_luminosity,
            GL_DYNAMIC_DRAW,
        )

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        self.vbo_needs_update = False

    def _setup_camera(self):
        glLoadIdentity()
        distance = self.camera_distance / self.zoom_factor
        glTranslatef(0.0, 0.0, -distance)
        glRotatef(self.camera_rotation_x, 1.0, 0.0, 0.0)
        glRotatef(self.camera_rotation_y, 0.0, 1.0, 0.0)
        glTranslatef(-self.center[0], -self.center[1], -self.center[2])

    def _render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self._setup_camera()

        if self.vbo_needs_update:
            self._update_vbo()

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_vertices)
        glVertexPointer(3, GL_FLOAT, 0, None)

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_colors)
        glColorPointer(3, GL_FLOAT, 0, None)

        glDrawArrays(GL_POINTS, 0, len(self.points))

        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        sdl2.SDL_GL_SwapWindow(self.window)

    def _handle_events(self):
        event = sdl2.SDL_Event()

        while sdl2.SDL_PollEvent(ctypes.byref(event)) != 0:
            if event.type == sdl2.SDL_QUIT:
                return False
            elif event.type == sdl2.SDL_KEYDOWN:
                if event.key.keysym.sym == sdl2.SDLK_ESCAPE:
                    return False
            elif event.type == sdl2.SDL_MOUSEBUTTONDOWN:
                if event.button.button == sdl2.SDL_BUTTON_LEFT:
                    self.mouse_dragging = True
                    self.last_mouse_x = event.button.x
                    self.last_mouse_y = event.button.y
            elif event.type == sdl2.SDL_MOUSEBUTTONUP:
                if event.button.button == sdl2.SDL_BUTTON_LEFT:
                    self.mouse_dragging = False
            elif event.type == sdl2.SDL_MOUSEMOTION:
                if self.mouse_dragging:
                    dx = event.motion.x - self.last_mouse_x
                    dy = event.motion.y - self.last_mouse_y
                    self.camera_rotation_y += dx * self.mouse_sensitivity
                    self.camera_rotation_x += dy * self.mouse_sensitivity
                    self.camera_rotation_x = np.clip(self.camera_rotation_x, -89.0, 89.0)
                    self.last_mouse_x = event.motion.x
                    self.last_mouse_y = event.motion.y
            elif event.type == sdl2.SDL_MOUSEWHEEL:
                if event.wheel.y > 0:
                    self.zoom_factor *= 1.1
                elif event.wheel.y < 0:
                    self.zoom_factor /= 1.1
                self.zoom_factor = np.clip(self.zoom_factor, 0.1, 10.0)

        return True

    def update_points(self, points, colors=None, luminosities=None):
        self.points = np.array(points, dtype=np.float32)

        if colors is not None:
            self.colors = np.array(colors, dtype=np.float32)

        if luminosities is not None:
            self.luminosities = np.array(luminosities, dtype=np.float32)

        self.vbo_needs_update = True

    def run(self, updater=None, dt=0.001, max_frames=None, metrics_callback=None):
        self.running = True

        print("Contrôles :")
        print("  - Clic gauche + déplacement souris : rotation de la caméra")
        print("  - Molette de la souris : zoom")
        print("  - ESC ou fermeture de fenêtre : quitter")

        previous_frame_end = time.perf_counter()
        frame_count = 0

        while self.running:
            self.running = self._handle_events()

            render_start = time.perf_counter()
            self._render()
            render_end = time.perf_counter()

            update_start = time.perf_counter()
            if updater is not None:
                self.update_points(updater(dt))
            frame_end = time.perf_counter()

            render_ms = (render_end - render_start) * 1000.0
            update_ms = (frame_end - update_start) * 1000.0
            total_ms = (frame_end - previous_frame_end) * 1000.0
            previous_frame_end = frame_end

            if metrics_callback is not None:
                metrics_callback(frame_count, render_ms, update_ms, total_ms)

            print(
                f"Frame {frame_count:04d} | Render: {render_ms:8.3f} ms | "
                f"Update: {update_ms:8.3f} ms | Total: {total_ms:8.3f} ms",
                end="\r",
            )

            frame_count += 1
            if max_frames is not None and frame_count >= max_frames:
                self.running = False

        print()
        self.cleanup()

    def cleanup(self):
        if self.vbo_vertices is not None:
            glDeleteBuffers(1, [self.vbo_vertices])
        if self.vbo_colors is not None:
            glDeleteBuffers(1, [self.vbo_colors])

        if self.gl_context:
            sdl2.SDL_GL_DeleteContext(self.gl_context)

        if self.window:
            sdl2.SDL_DestroyWindow(self.window)

        sdl2.SDL_Quit()
