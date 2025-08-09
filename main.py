#!/usr/bin/env python3

import math

import random

import sys

from dataclasses import dataclass

from typing import List, Tuple



import numpy as np



from PySide6 import QtCore, QtWidgets, QtGui

from PySide6.QtCore import Qt, QTimer

from PySide6.QtGui import QVector3D, QColor

from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QComboBox, QSpinBox, QSlider, QLineEdit



# Qt3D

from PySide6 import Qt3DCore, Qt3DExtras, Qt3DRender, Qt3DInput





U, D, F, B, L, R = 1, 2, 4, 8, 16, 32



@dataclass

class Move:

    axis: str  # 'x'|'y'|'z'

    index: int  # slice index

    k: int      # quarter-turns (1..3)



def blend_colors(bits: int) -> QColor:

    # Face colors similar to classic cube

    face_colors = {

        U: QColor('#ffd60a'),  # yellow

        D: QColor('#ffffff'),  # white

        F: QColor('#00a651'),  # green

        B: QColor('#0057ff'),  # blue

        L: QColor('#ff8c00'),  # orange

        R: QColor('#ff0030'),  # red

    }

    rs=gs=bs=cnt=0

    for bit, col in face_colors.items():

        if bits & bit:

            rs += col.redF(); gs += col.greenF(); bs += col.blueF(); cnt += 1

    if cnt == 0:

        return QColor('#111111')

    r = max(0.0, min(1.0, rs/cnt))

    g = max(0.0, min(1.0, gs/cnt))

    b = max(0.0, min(1.0, bs/cnt))

    c = QColor()

    c.setRgbF(r,g,b,1.0)

    return c



class VoxelRubiks:

    """Outer-shell cubelets only; each cubelet keeps its own color/mask and moves in 3D integer grid."""

    def __init__(self, n=20):

        assert isinstance(n, int) and n>=2

        self.n = n

        self.half = (n-1)/2.0

        # Build outer coordinates and initial face mask per cubelet (order is stable)

        self.coords = []  # list of (x,y,z) integers per cubelet (entity order)

        self.masks  = []  # corresponding bitmask (color) per cubelet

        for z in range(n):

            for y in range(n):

                for x in range(n):

                    if x in (0,n-1) or y in (0,n-1) or z in (0,n-1):

                        self.coords.append([x,y,z])

                        mask = 0

                        if z==0:     mask |= U

                        if z==n-1:   mask |= D

                        if y==0:     mask |= F

                        if y==n-1:   mask |= B

                        if x==0:     mask |= L

                        if x==n-1:   mask |= R

                        self.masks.append(mask)

        self.coords = np.array(self.coords, dtype=np.int16)  # shape (M,3)

        self.masks  = np.array(self.masks, dtype=np.uint8)

        self.solved = self.coords.copy()

        self.history: List[Move] = []



    def reset(self):

        self.coords[:] = self.solved

        self.history.clear()



    def scramble(self, moves=24, seed=None):

        if seed is not None:

            random.seed(seed)

        axes = ['x','y','z']

        seq = []

        for _ in range(moves):

            axis = random.choice(axes)

            index = random.randint(0, self.n-1)

            k = random.choice([1,2,3])

            seq.append(Move(axis,index,k))

        self.history.extend(seq)

        return seq



    def inverse_history(self) -> List[Move]:

        inv = []

        for m in reversed(self.history):

            inv_k = (4 - (m.k % 4)) % 4

            if inv_k:

                inv.append(Move(m.axis, m.index, inv_k))

        return inv



    def _slice_mask(self, axis, index):

        if axis=='x': return (self.coords[:,0]==index)

        if axis=='y': return (self.coords[:,1]==index)

        return (self.coords[:,2]==index)



    def _rotate_slice_90(self, axis, index, dir_sign=+1):

        """Commit a 90° rotation for the slice; updates integer coords in-place."""

        mask = self._slice_mask(axis, index)

        pts = self.coords[mask].astype(np.float64)

        pts -= self.half

        a = dir_sign * (math.pi/2.0)

        c, s = math.cos(a), math.sin(a)

        if axis=='x':

            y = pts[:,1].copy(); z = pts[:,2].copy()

            pts[:,1] = c*y - s*z

            pts[:,2] = s*y + c*z

        elif axis=='y':

            x = pts[:,0].copy(); z = pts[:,2].copy()

            pts[:,0] =  c*x + s*z

            pts[:,2] = -s*x + c*z

        else: # 'z'

            x = pts[:,0].copy(); y = pts[:,1].copy()

            pts[:,0] = c*x - s*y

            pts[:,1] = s*x + c*y

        snapped = np.rint(pts + self.half).astype(np.int16)

        self.coords[mask] = snapped



class CubeletEntity:

    """Holds per-cubelet Transform and color; shares a global mesh for performance."""

    def __init__(self, parent_entity: Qt3DCore.QEntity, mesh: Qt3DExtras.QCuboidMesh, color: QColor, pos: Tuple[float,float,float], scale: float=0.92):

        self.entity = Qt3DCore.QEntity(parent_entity)

        self.transform = Qt3DCore.QTransform()

        self.material = Qt3DExtras.QPhongMaterial(self.entity)

        self.material.setAmbient(color)

        self.material.setDiffuse(color)

        self.entity.addComponent(mesh)

        self.entity.addComponent(self.material)

        self.entity.addComponent(self.transform)

        self.set_position(pos)

        self.set_scale(scale)



    def set_position(self, pos: Tuple[float,float,float]):

        self.transform.setTranslation(QVector3D(*pos))



    def set_scale(self, s: float):

        self.transform.setScale(s)



class VoxelQtScene(QtWidgets.QWidget):

    """Qt3D scene + UI controls + animation loop."""

    def __init__(self, n=20, parent=None):

        super().__init__(parent)

        self.n = n

        self.model = VoxelRubiks(n)

        self.half = (n-1)/2.0



        # --- Build UI layout with a Qt3DWindow inside a QWidget container ---

        self.view = Qt3DExtras.Qt3DWindow()

        self.view.defaultFrameGraph().setClearColor(QColor('#0b0b0e'))

        container = QtWidgets.QWidget.createWindowContainer(self.view, self)

        container.setMinimumSize(640, 480)

        container.setFocusPolicy(Qt.StrongFocus)



        # Controls panel

        panel = QtWidgets.QWidget(self)

        ph = QHBoxLayout(panel); ph.setContentsMargins(8,8,8,8); ph.setSpacing(8)



        self.axisBox = QComboBox(); self.axisBox.addItems(['x','y','z'])

        self.sliceSpin = QSpinBox(); self.sliceSpin.setRange(0, n-1)

        self.kBox = QComboBox(); self.kBox.addItems(['90°','180°','270°'])

        self.turnBtn = QPushButton('Turn')

        self.scrambleSpin = QSpinBox(); self.scrambleSpin.setRange(1, 500); self.scrambleSpin.setValue(32)

        self.seedEdit = QLineEdit(); self.seedEdit.setPlaceholderText('seed (optional)')

        self.scrambleBtn = QPushButton('Scramble')

        self.solveBtn = QPushButton('Solve')

        self.resetBtn = QPushButton('Reset')

        self.speedSlider = QSlider(Qt.Horizontal); self.speedSlider.setRange(1, 50); self.speedSlider.setValue(25)



        for w,label in [(self.axisBox,'Axis'), (self.sliceSpin,'Slice'), (self.kBox,'Quarter-turns'),

                        (self.scrambleSpin,'Moves'), (self.seedEdit,'Seed'), (self.speedSlider,'Speed')]:

            lab = QLabel(label); lab.setStyleSheet('color:#bbb')

            ph.addWidget(lab); ph.addWidget(w)

        for b in (self.turnBtn,self.scrambleBtn,self.solveBtn,self.resetBtn):

            ph.addWidget(b)



        # Layout

        root = QVBoxLayout(self)

        root.setContentsMargins(0,0,0,0); root.setSpacing(0)

        root.addWidget(panel)

        root.addWidget(container, 1)



        # --- Build 3D scene ---

        self.rootEntity = Qt3DCore.QEntity()

        self.view.setRootEntity(self.rootEntity)



        # Camera

        self.camera = self.view.camera()

        self.camera.lens().setPerspectiveProjection(45.0, 16/9, 0.1, 1000)

        self.camera.setPosition(QVector3D(n*1.3, n*1.2, n*1.4))

        self.camera.setViewCenter(QVector3D(0,0,0))



        # Controls

        cam_ctrl = Qt3DExtras.QOrbitCameraController(self.rootEntity)

        cam_ctrl.setLinearSpeed(100)

        cam_ctrl.setLookSpeed(180)

        cam_ctrl.setCamera(self.camera)



        # Light

        self.lightEntity = Qt3DCore.QEntity(self.rootEntity)

        self.light = Qt3DRender.QPointLight(self.lightEntity)

        self.light.setColor(QColor('white'))

        self.light.setIntensity(1.2)

        self.lightTransform = Qt3DCore.QTransform()

        self.lightTransform.setTranslation(QVector3D(n*2.0, n*2.0, n*2.0))

        self.lightEntity.addComponent(self.light)

        self.lightEntity.addComponent(self.lightTransform)



        # Shared cube mesh

        self.cubeMesh = Qt3DExtras.QCuboidMesh(self.rootEntity)

        self.cubeMesh.setXExtent(1.0); self.cubeMesh.setYExtent(1.0); self.cubeMesh.setZExtent(1.0)



        # Build entities for each outer cubelet

        self.cubelets: List[CubeletEntity] = []

        for i, (x,y,z) in enumerate(self.model.coords):

            mask = int(self.model.masks[i])

            color = blend_colors(mask)

            pos = (x-self.half, y-self.half, z-self.half)

            ent = CubeletEntity(self.rootEntity, self.cubeMesh, color, pos, scale=0.92)

            self.cubelets.append(ent)



        # --- Animation state ---

        self.queue: List[Tuple[str,int,int]] = []  # expanded quarter-turns: (axis,index,dir=+1)

        self.animating = False

        self.frame = 0

        self.frames_per_turn = 12

        self.speed = 1.0  # multiplier

        self.current_move = None  # (axis,index,dir)

        self.moving_indices = None  # numpy mask of moving cubelets



        # Timer

        self.timer = QtCore.QTimer(self)

        self.timer.setInterval(16)

        self.timer.timeout.connect(self._on_tick)

        self.timer.start()



        # Wire controls

        self.turnBtn.clicked.connect(self.on_turn_clicked)

        self.scrambleBtn.clicked.connect(self.on_scramble_clicked)

        self.solveBtn.clicked.connect(self.on_solve_clicked)

        self.resetBtn.clicked.connect(self.on_reset_clicked)

        self.speedSlider.valueChanged.connect(self.on_speed_changed)



    # --- Controls handlers ---

    def on_speed_changed(self, value: int):

        # Map 1..50 to 0.25..4.0

        self.speed = 0.25 + (value-1) * (3.75/49.0)



    def on_turn_clicked(self):

        axis = self.axisBox.currentText()

        index = self.sliceSpin.value()

        k = self.kBox.currentIndex()+1

        self.enqueue_move(axis, index, k)



    def on_scramble_clicked(self):

        moves = self.scrambleSpin.value()

        seed_text = self.seedEdit.text().strip()

        seed = None

        if seed_text:

            # FNV-1a 32-bit hash

            h = 2166136261

            for ch in seed_text:

                h ^= ord(ch); h = (h * 16777619) & 0xffffffff

            seed = h

        seq = self.model.scramble(moves, seed)

        for m in seq:

            self.enqueue_move(m.axis, m.index, m.k, record=False)  # already recorded



    def on_solve_clicked(self):

        inv = self.model.inverse_history()

        for m in inv:

            self.enqueue_move(m.axis, m.index, m.k, record=False)  # don't extend history



        # Clear history after we scheduled the solve sequence

        self.model.history.clear()



    def on_reset_clicked(self):

        self.queue.clear(); self.animating=False; self.current_move=None

        self.model.reset()

        # Snap visuals to solved

        for i, (x,y,z) in enumerate(self.model.coords):

            self.cubelets[i].set_position((x-self.half, y-self.half, z-self.half))



    # --- Animation queue ---

    def enqueue_move(self, axis: str, index: int, k: int, record=True):

        steps = k % 4

        for _ in range(steps):

            self.queue.append((axis, index, +1))

        if record:

            self.model.history.append(Move(axis, index, steps))



    def _start_next_turn(self):

        if not self.queue:

            self.animating = False

            self.current_move = None

            self.moving_indices = None

            return

        self.animating = True

        self.frame = 0

        axis, index, dir_sign = self.queue.pop(0)

        self.current_move = (axis, index, dir_sign)

        # Precompute which cubelets are in the slice

        mask = self.model._slice_mask(axis, index)

        self.moving_indices = np.where(mask)[0]



    def _on_tick(self):

        if not self.animating and self.queue:

            self._start_next_turn()

        if not self.animating:

            return



        axis, index, dir_sign = self.current_move

        # Interp 0..1 across the quarter-turn

        total_frames = max(1, int(self.frames_per_turn / self.speed))

        t01 = self.frame / max(1, total_frames-1)

        angle = dir_sign * (math.pi/2.0) * t01



        # Update positions for moving slice

        h = self.half

        c, s = math.cos(angle), math.sin(angle)



        # For performance: update only moving cubelets' transforms

        for i in self.moving_indices:

            x, y, z = self.model.coords[i]

            # base pos

            px, py, pz = (x-h, y-h, z-h)

            if axis == 'x':

                ny = c*py - s*pz

                nz = s*py + c*pz

                nx = px

            elif axis == 'y':

                nx =  c*px + s*pz

                nz = -s*px + c*pz

                ny = py

            else:  # 'z'

                nx = c*px - s*py

                ny = s*px + c*py

                nz = pz

            self.cubelets[i].set_position((nx, ny, nz))



        self.frame += 1

        if self.frame >= total_frames:

            # Commit rotation: snap integer coords and update visuals to new bases

            self.model._rotate_slice_90(axis, index, dir_sign)

            for i in self.moving_indices:

                x,y,z = self.model.coords[i]

                self.cubelets[i].set_position((x-h, y-h, z-h))

            # Next

            self._start_next_turn()



class MainWindow(QMainWindow):

    def __init__(self, n=20):

        super().__init__()

        self.setWindowTitle('20×20×20 Voxel Cube — Qt6/Qt3D')

        self.resize(1200, 800)

        self.scene = VoxelQtScene(n=n)

        self.setCentralWidget(self.scene)



def main():

    app = QApplication(sys.argv)

    # High-DPI friendly

    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)

    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    w = MainWindow(n=20)

    w.show()

    sys.exit(app.exec())



if __name__ == '__main__':

    main()



