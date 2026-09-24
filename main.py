import sys
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                              QLineEdit, QComboBox, QLabel, QPushButton, QCheckBox, 
                              QSpinBox, QDoubleSpinBox, QGroupBox, QFormLayout, 
                              QMessageBox, QSplitter, QFrame)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QColor, QPalette
import pyqtgraph as pg
from pyqtgraph import PlotWidget, PlotItem, InfiniteLine, FillBetweenItem, TargetItem
from pyqtgraph.graphicsItems.TextItem import TextItem

from calculus_engine import CalculusEngine, PRESET_FUNCTIONS


class MathTextItem(TextItem):
    """Custom text item for displaying math formulas"""
    def __init__(self, text="", color='white', font_size=12):
        super().__init__(text, color=color)
        self.setFont(QFont('Consolas', font_size))


class TangentPoint(TargetItem):
    """Draggable point on the curve for tangent line"""
    def __init__(self, engine, plot_widget, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine = engine
        self.plot_widget = plot_widget
        self.movable = True
        self.symbol = 'o'
        self.symbolSize = 12
        self.symbolBrush = pg.mkBrush('yellow')
        self.symbolPen = pg.mkPen('black', width=2)
        self._last_pos = None
        
    def mouseDragEvent(self, ev):
        super().mouseDragEvent(ev)
        if ev.isFinish():
            self.update_tangent()
    
    def mouseClickEvent(self, ev):
        super().mouseClickEvent(ev)
        self.update_tangent()
    
    def update_tangent(self):
        pos = self.getPos()
        x_val = pos[0]
        y_val = self.engine.evaluate_at_point(x_val)
        if not np.isnan(y_val):
            self.setPos(x_val, y_val)
            self.plot_widget.update_tangent_line(x_val, y_val)


class CalculusPlotWidget(PlotWidget):
    """Main plotting widget with interactive calculus features"""
    
    def __init__(self, engine, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine = engine
        
        # Configure plot
        self.showGrid(x=True, y=True, alpha=0.3)
        self.setLabel('bottom', 'x')
        self.setLabel('left', 'y')
        self.setMouseEnabled(x=True, y=True)
        self.enableAutoRange(x=False, y=False)
        self.setXRange(-10, 10)
        self.setYRange(-10, 10)
        
        # Plot items
        self.curve_item = self.plot([], [], pen=pg.mkPen('cyan', width=2), name='f(x)')
        self.derivative_curve = self.plot([], [], pen=pg.mkPen('magenta', width=2, style=Qt.PenStyle.DashLine), name="f'(x)")
        self.derivative_curve.setVisible(False)
        
        # Tangent line
        self.tangent_line = InfiniteLine(angle=0, movable=False, 
                                          pen=pg.mkPen('yellow', width=2, style=Qt.PenStyle.DashLine))
        self.tangent_line.setVisible(False)
        self.addItem(self.tangent_line)
        
        # Tangent point
        self.tangent_point = TangentPoint(engine, self, pos=(0, 0))
        self.tangent_point.setVisible(False)
        self.addItem(self.tangent_point)
        
        # Integral shade
        self.integral_shade = None
        self.integral_a = None
        self.integral_b = None
        
        # Asymptote lines
        self.asymptote_lines = []
        
        # Slope label
        self.slope_label = MathTextItem("", color='yellow', font_size=11)
        self.slope_label.setVisible(False)
        self.addItem(self.slope_label)
        
        # Connect signals
        self.scene().sigMouseMoved.connect(self.on_mouse_moved)
        
    def update_plot(self, x_min=-10, x_max=10, num_points=1000):
        """Update the main function plot"""
        x_vals = np.linspace(x_min, x_max, num_points)
        y_vals = self.engine.evaluate(x_vals)
        
        # Mask out infinite/nan values for cleaner plotting
        mask = np.isfinite(y_vals)
        x_plot = x_vals[mask]
        y_plot = y_vals[mask]
        
        self.curve_item.setData(x_plot, y_plot)
        
        # Update derivative if visible
        if self.derivative_curve.isVisible():
            dy_vals = self.engine.evaluate_derivative(x_vals)
            mask_d = np.isfinite(dy_vals)
            self.derivative_curve.setData(x_vals[mask_d], dy_vals[mask_d])
        
        # Update asymptotes
        self.update_asymptotes()
        
    def update_derivative_plot(self, visible: bool):
        """Toggle derivative plot visibility"""
        self.derivative_curve.setVisible(visible)
        if visible:
            self.update_plot(self.getViewBox().viewRange()[0][0], self.getViewBox().viewRange()[0][1])
    
    def update_asymptotes(self):
        """Draw vertical asymptote lines"""
        # Remove old asymptotes
        if hasattr(self, 'asymptote_lines'):
            for line in self.asymptote_lines:
                self.removeItem(line)
        self.asymptote_lines = []
        
        # Add new ones
        view_range = self.getViewBox().viewRange()
        x_min, x_max = view_range[0]
        y_min, y_max = view_range[1]
        
        for asym in self.engine.asymptotes:
            if x_min <= asym <= x_max:
                line = InfiniteLine(pos=asym, angle=90, 
                                    pen=pg.mkPen('red', width=1, style=Qt.PenStyle.DashLine))
                self.addItem(line)
                self.asymptote_lines.append(line)
    
    def update_tangent_line(self, x_val, y_val):
        """Update tangent line at point (x_val, y_val)"""
        slope = self.engine.derivative_at_point(x_val)
        if np.isnan(slope) or np.isnan(y_val):
            self.tangent_line.setVisible(False)
            self.tangent_point.setVisible(False)
            self.slope_label.setVisible(False)
            return
        
        # Tangent line: y - y0 = m(x - x0) => y = m*x + (y0 - m*x0)
        intercept = y_val - slope * x_val
        self.tangent_line.setPos(intercept)
        self.tangent_line.setAngle(np.degrees(np.arctan(slope)))
        self.tangent_line.setVisible(True)
        
        self.tangent_point.setPos(x_val, y_val)
        self.tangent_point.setVisible(True)
        
        # Update slope label
        self.slope_label.setText(f"Slope: {slope:.4f}\nf'({x_val:.2f}) = {slope:.4f}")
        self.slope_label.setPos(x_val + 0.5, y_val + 0.5)
        self.slope_label.setVisible(True)
    
    def set_integral_bounds(self, a: float, b: float):
        """Set and display definite integral bounds with shading"""
        # Remove old shade
        if self.integral_shade:
            self.removeItem(self.integral_shade)
        
        self.integral_a = min(a, b)
        self.integral_b = max(a, b)
        
        # Create fill between curve and x-axis
        x_vals = np.linspace(self.integral_a, self.integral_b, 200)
        y_vals = self.engine.evaluate(x_vals)
        
        # Create zero line
        y_zero = np.zeros_like(x_vals)
        
        # Fill between curve and zero
        self.integral_shade = FillBetweenItem(
            pg.PlotCurveItem(x_vals, y_vals, pen=None),
            pg.PlotCurveItem(x_vals, y_zero, pen=None),
            brush=pg.mkBrush(0, 255, 0, 80)
        )
        self.addItem(self.integral_shade)
        
        # Add vertical boundary lines
        self.integral_line_a = InfiniteLine(pos=self.integral_a, angle=90, 
                                             pen=pg.mkPen('green', width=2, style=Qt.PenStyle.DotLine))
        self.integral_line_b = InfiniteLine(pos=self.integral_b, angle=90, 
                                             pen=pg.mkPen('green', width=2, style=Qt.PenStyle.DotLine))
        self.addItem(self.integral_line_a)
        self.addItem(self.integral_line_b)
    
    def clear_integral(self):
        """Remove integral shading and lines"""
        if self.integral_shade:
            self.removeItem(self.integral_shade)
            self.integral_shade = None
        if hasattr(self, 'integral_line_a') and self.integral_line_a:
            self.removeItem(self.integral_line_a)
        if hasattr(self, 'integral_line_b') and self.integral_line_b:
            self.removeItem(self.integral_line_b)
    
    def on_mouse_moved(self, pos):
        """Handle mouse movement for coordinate display"""
        if self.sceneBoundingRect().contains(pos):
            mouse_point = self.getViewBox().mapSceneToView(pos)
            x = mouse_point.x()
            y = self.engine.evaluate_at_point(x)
            if not np.isnan(y):
                self.setTitle(f"x = {x:.3f},  f(x) = {y:.3f}")
    
    def viewRangeChanged(self, view, range):
        """Called when view range changes"""
        if hasattr(self, 'asymptote_lines'):
            self.update_asymptotes()
        super().viewRangeChanged(view, range)


class VirtualKeypad(QWidget):
    """Virtual keyboard for math input"""
    
    key_pressed = pyqtSignal(str)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        
        # Row 1: Functions
        row1 = QHBoxLayout()
        for btn_text in ['sin(', 'cos(', 'tan(', 'sqrt(', 'log(', 'exp(', 'abs(', 'pi']:
            btn = QPushButton(btn_text)
            btn.clicked.connect(lambda checked, t=btn_text: self.key_pressed.emit(t))
            btn.setMaximumHeight(35)
            row1.addWidget(btn)
        layout.addLayout(row1)
        
        # Row 2: Operators and constants
        row2 = QHBoxLayout()
        for btn_text in ['+', '-', '*', '/', '^', '(', ')', 'e']:
            btn = QPushButton(btn_text)
            btn.clicked.connect(lambda checked, t=btn_text: self.key_pressed.emit(t))
            btn.setMaximumHeight(35)
            row2.addWidget(btn)
        layout.addLayout(row2)
        
        # Row 3: Calculus
        row3 = QHBoxLayout()
        for btn_text in ['d/dx', '∫', '∫_a^b', 'lim', '∞', 'dx', '=']:
            btn = QPushButton(btn_text)
            btn.clicked.connect(lambda checked, t=btn_text: self.key_pressed.emit(t))
            btn.setMaximumHeight(35)
            row3.addWidget(btn)
        layout.addLayout(row3)
        
        # Row 4: Navigation
        row4 = QHBoxLayout()
        for btn_text in ['←', '→', 'Del', 'Clear']:
            btn = QPushButton(btn_text)
            btn.clicked.connect(lambda checked, t=btn_text: self.key_pressed.emit(t))
            btn.setMaximumHeight(35)
            row4.addWidget(btn)
        layout.addLayout(row4)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.engine = CalculusEngine()
        self.init_ui()
        self.setup_connections()
        
        # Load default function
        self.function_input.setText("x") # -----------------------------------
        self.on_function_changed()
    
    def init_ui(self):
        self.setWindowTitle("Calculus Graphing Calculator")
        self.setGeometry(100, 100, 1400, 900)
        
        # Main widget and splitter
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(400)
        left_panel.setMinimumWidth(350)
        
        # Function Input Group
        input_group = QGroupBox("Function Input")
        input_layout = QVBoxLayout(input_group)
        
        # Preset dropdown
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["Custom"] + list(PRESET_FUNCTIONS.keys()))
        self.preset_combo.currentTextChanged.connect(self.on_preset_changed)
        preset_layout.addWidget(self.preset_combo)
        input_layout.addLayout(preset_layout)
        
        # Function input
        self.function_input = QLineEdit()
        self.function_input.setPlaceholderText("Enter f(x) = ...")
        self.function_input.setFont(QFont('Consolas', 14))
        self.function_input.setMinimumHeight(45)
        input_layout.addWidget(self.function_input)
        
        # Validation label
        self.validation_label = QLabel("")
        self.validation_label.setStyleSheet("color: red; font-size: 11px;")
        self.validation_label.setWordWrap(True)
        input_layout.addWidget(self.validation_label)
        
        # LaTeX display
        self.latex_label = QLabel("")
        self.latex_label.setStyleSheet("color: #aaa; font-size: 12px; background: #222; padding: 5px;")
        self.latex_label.setWordWrap(True)
        self.latex_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        input_layout.addWidget(self.latex_label)
        
        left_layout.addWidget(input_group)
        
        # Virtual Keypad
        self.keypad = VirtualKeypad()
        left_layout.addWidget(self.keypad)
        
        # Calculus Operations Group
        calc_group = QGroupBox("Calculus Operations")
        calc_layout = QVBoxLayout(calc_group)
        
        # Derivative
        deriv_layout = QHBoxLayout()
        self.show_derivative_cb = QCheckBox("Show f'(x)")
        self.show_derivative_cb.setChecked(False)
        deriv_layout.addWidget(self.show_derivative_cb)
        self.derivative_latex = QLabel("")
        self.derivative_latex.setStyleSheet("color: magenta; font-size: 11px;")
        self.derivative_latex.setWordWrap(True)
        deriv_layout.addWidget(self.derivative_latex)
        calc_layout.addLayout(deriv_layout)
        
        # Indefinite Integral
        int_layout = QVBoxLayout()
        self.indefinite_latex = QLabel("∫ f(x) dx = ")
        self.indefinite_latex.setStyleSheet("color: lightgreen; font-size: 11px;")
        self.indefinite_latex.setWordWrap(True)
        int_layout.addWidget(self.indefinite_latex)
        calc_layout.addLayout(int_layout)
        
        # Definite Integral
        def_int_layout = QFormLayout()
        self.integral_a = QDoubleSpinBox()
        self.integral_a.setRange(-100, 100)
        self.integral_a.setDecimals(3)
        self.integral_a.setValue(0)
        self.integral_a.setSingleStep(0.5)
        
        self.integral_b = QDoubleSpinBox()
        self.integral_b.setRange(-100, 100)
        self.integral_b.setDecimals(3)
        self.integral_b.setValue(2)
        self.integral_b.setSingleStep(0.5)
        
        def_int_layout.addRow("∫ from a:", self.integral_a)
        def_int_layout.addRow("to b:", self.integral_b)
        
        self.compute_integral_btn = QPushButton("Compute Definite Integral")
        def_int_layout.addRow(self.compute_integral_btn)
        
        self.definite_result = QLabel("")
        self.definite_result.setStyleSheet("color: yellow; font-size: 12px; font-weight: bold;")
        self.definite_result.setWordWrap(True)
        def_int_layout.addRow(self.definite_result)
        
        calc_layout.addLayout(def_int_layout)
        
        # Tangent point control
        tangent_layout = QFormLayout()
        self.tangent_x = QDoubleSpinBox()
        self.tangent_x.setRange(-100, 100)
        self.tangent_x.setDecimals(3)
        self.tangent_x.setValue(1)
        self.tangent_x.setSingleStep(0.1)
        tangent_layout.addRow("Tangent at x:", self.tangent_x)
        
        self.show_tangent_cb = QCheckBox("Show Tangent Line")
        tangent_layout.addRow(self.show_tangent_cb)
        
        self.tangent_info = QLabel("")
        self.tangent_info.setStyleSheet("color: yellow; font-size: 11px;")
        self.tangent_info.setWordWrap(True)
        tangent_layout.addRow(self.tangent_info)
        
        calc_layout.addLayout(tangent_layout)
        
        left_layout.addWidget(calc_group)
        
        # View Controls
        view_group = QGroupBox("View Controls")
        view_layout = QFormLayout(view_group)
        
        self.x_min = QDoubleSpinBox()
        self.x_min.setRange(-1000, 1000)
        self.x_min.setValue(-10)
        self.x_min.setSingleStep(1)
        
        self.x_max = QDoubleSpinBox()
        self.x_max.setRange(-1000, 1000)
        self.x_max.setValue(10)
        self.x_max.setSingleStep(1)
        
        self.y_min = QDoubleSpinBox()
        self.y_min.setRange(-1000, 1000)
        self.y_min.setValue(-10)
        self.y_min.setSingleStep(1)
        
        self.y_max = QDoubleSpinBox()
        self.y_max.setRange(-1000, 1000)
        self.y_max.setValue(10)
        self.y_max.setSingleStep(1)
        
        view_layout.addRow("X min:", self.x_min)
        view_layout.addRow("X max:", self.x_max)
        view_layout.addRow("Y min:", self.y_min)
        view_layout.addRow("Y max:", self.y_max)
        
        self.auto_range_btn = QPushButton("Auto Range")
        view_layout.addRow(self.auto_range_btn)
        
        left_layout.addWidget(view_group)
        left_layout.addStretch()
        
        # Right panel - Plot
        self.plot_widget = CalculusPlotWidget(self.engine)
        
        splitter.addWidget(left_panel)
        splitter.addWidget(self.plot_widget)
        splitter.setSizes([380, 1020])
        
        # Apply dark theme
        self.apply_dark_theme()
    
    def apply_dark_theme(self):
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
        dark_palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.Base, QColor(20, 20, 20))
        dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(40, 40, 40))
        dark_palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.Button, QColor(50, 50, 50))
        dark_palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
        dark_palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
        self.setPalette(dark_palette)
        
        # Style for buttons
        self.setStyleSheet("""
            QPushButton {
                background-color: #444;
                border: 1px solid #666;
                border-radius: 4px;
                padding: 5px;
                color: white;
            }
            QPushButton:hover {
                background-color: #555;
            }
            QPushButton:pressed {
                background-color: #333;
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: #222;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 5px;
                color: white;
            }
            QGroupBox {
                border: 1px solid #555;
                border-radius: 5px;
                margin-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
        """)
    
    def setup_connections(self):
        # Function input
        self.function_input.textChanged.connect(self.on_function_text_changed)
        self.function_input.returnPressed.connect(self.on_function_changed)
        
        # Keypad
        self.keypad.key_pressed.connect(self.on_keypad_key)
        
        # Derivative
        self.show_derivative_cb.toggled.connect(self.on_derivative_toggled)
        
        # Definite integral
        self.compute_integral_btn.clicked.connect(self.on_compute_definite_integral)
        
        # Tangent
        self.show_tangent_cb.toggled.connect(self.on_tangent_toggled)
        self.tangent_x.valueChanged.connect(self.on_tangent_x_changed)
        
        # View
        self.x_min.valueChanged.connect(self.on_view_changed)
        self.x_max.valueChanged.connect(self.on_view_changed)
        self.y_min.valueChanged.connect(self.on_view_changed)
        self.y_max.valueChanged.connect(self.on_view_changed)
        self.auto_range_btn.clicked.connect(self.on_auto_range)
        
        # Debounce timer for live validation
        self.validation_timer = QTimer()
        self.validation_timer.setSingleShot(True)
        self.validation_timer.timeout.connect(self.validate_input)
    
    def on_preset_changed(self, text):
        if text in PRESET_FUNCTIONS:
            self.function_input.setText(PRESET_FUNCTIONS[text])
            self.on_function_changed()
    
    def on_function_text_changed(self):
        # Live validation with debounce
        self.validation_timer.start(300)
        text = self.function_input.text()
        is_valid, error = self.engine.validate_syntax(text)
        if not is_valid and text:
            self.validation_label.setText(f"⚠ {error}")
        else:
            self.validation_label.setText("")
    
    def validate_input(self):
        text = self.function_input.text()
        is_valid, error = self.engine.validate_syntax(text)
        if not is_valid and text:
            self.validation_label.setText(f"⚠ {error}")
        else:
            self.validation_label.setText("")
    
    def on_function_changed(self):
        text = self.function_input.text()
        success, error = self.engine.parse_expression(text)
        
        if success:
            self.validation_label.setText("")
            self.validation_label.setStyleSheet("color: green; font-size: 11px;")
            self.validation_label.setText("✓ Valid expression")
            
            # Update LaTeX display
            self.latex_label.setText(f"f(x) = {self.engine.get_function_latex()}")
            
            # Update derivative display
            self.derivative_latex.setText(f"f'(x) = {self.engine.get_derivative_latex()}")
            
            # Update indefinite integral
            integral_latex = self.engine.get_integral_latex()
            if integral_latex:
                self.indefinite_latex.setText(f"∫ f(x) dx = {integral_latex} + C")
            else:
                self.indefinite_latex.setText("∫ f(x) dx = (no closed form)")
            
            # Update plot
            self.plot_widget.update_plot(
                self.x_min.value(), self.x_max.value()
            )
            
            # Update derivative if enabled
            if self.show_derivative_cb.isChecked():
                self.plot_widget.update_derivative_plot(True)
            
            # Update tangent if enabled
            if self.show_tangent_cb.isChecked():
                self.on_tangent_x_changed(self.tangent_x.value())
        else:
            self.validation_label.setStyleSheet("color: red; font-size: 11px;")
            self.validation_label.setText(f"✗ {error}")
            self.latex_label.setText("")
            self.derivative_latex.setText("")
            self.indefinite_latex.setText("∫ f(x) dx = ")
            self.curve_item.setData([], [])
    
    def on_keypad_key(self, key):
        """Handle virtual keypad input"""
        cursor_pos = self.function_input.cursorPosition()
        text = self.function_input.text()
        
        if key == '←':
            if cursor_pos > 0:
                self.function_input.setCursorPosition(cursor_pos - 1)
        elif key == '→':
            if cursor_pos < len(text):
                self.function_input.setCursorPosition(cursor_pos + 1)
        elif key == 'Del':
            if cursor_pos < len(text):
                new_text = text[:cursor_pos] + text[cursor_pos+1:]
                self.function_input.setText(new_text)
                self.function_input.setCursorPosition(cursor_pos)
        elif key == 'Clear':
            self.function_input.clear()
        elif key == 'd/dx':
            self.function_input.insert("diff(")
        elif key == '∫':
            self.function_input.insert("integrate(")
        elif key == '∫_a^b':
            self.function_input.insert("integrate(, x, a, b)")
            self.function_input.setCursorPosition(cursor_pos + 10)
        elif key == 'lim':
            self.function_input.insert("limit(")
        elif key == '∞':
            self.function_input.insert("oo")
        elif key == 'dx':
            self.function_input.insert(", x)")
        elif key == '=':
            self.on_function_changed()
        else:
            self.function_input.insert(key)
    
    def on_derivative_toggled(self, checked):
        self.plot_widget.update_derivative_plot(checked)
        if checked:
            self.plot_widget.update_plot(
                self.x_min.value(), self.x_max.value()
            )
    
    def on_compute_definite_integral(self):
        a = self.integral_a.value()
        b = self.integral_b.value()
        
        result, numeric = self.engine.definite_integral(a, b)
        
        if result is not None:
            if isinstance(numeric, str):  # Error message
                self.definite_result.setText(f"Error: {numeric}")
                self.definite_result.setStyleSheet("color: red; font-size: 12px;")
            else:
                self.definite_result.setText(f"∫[{a:.3f}, {b:.3f}] f(x) dx = {result} ≈ {numeric:.6f}")
                self.definite_result.setStyleSheet("color: yellow; font-size: 12px; font-weight: bold;")
                self.plot_widget.set_integral_bounds(a, b)
        else:
            self.definite_result.setText("Could not compute integral")
            self.definite_result.setStyleSheet("color: red; font-size: 12px;")
    
    def on_tangent_toggled(self, checked):
        self.plot_widget.tangent_line.setVisible(checked)
        self.plot_widget.tangent_point.setVisible(checked)
        self.plot_widget.slope_label.setVisible(checked)
        
        if checked:
            self.on_tangent_x_changed(self.tangent_x.value())
        else:
            self.tangent_info.setText("")
    
    def on_tangent_x_changed(self, x_val):
        if not self.show_tangent_cb.isChecked():
            return
        
        y_val = self.engine.evaluate_at_point(x_val)
        slope = self.engine.derivative_at_point(x_val)
        
        if np.isnan(y_val) or np.isnan(slope):
            self.tangent_info.setText("Tangent undefined at this point")
            self.plot_widget.tangent_line.setVisible(False)
            self.plot_widget.tangent_point.setVisible(False)
            self.plot_widget.slope_label.setVisible(False)
        else:
            self.tangent_info.setText(f"Point: ({x_val:.3f}, {y_val:.3f})\nSlope f'({x_val:.3f}) = {slope:.4f}\nLine: y = {slope:.4f}x + {y_val - slope*x_val:.4f}")
            self.plot_widget.update_tangent_line(x_val, y_val)
    
    def on_view_changed(self):
        self.plot_widget.setXRange(self.x_min.value(), self.x_max.value())
        self.plot_widget.setYRange(self.y_min.value(), self.y_max.value())
    
    def on_auto_range(self):
        # Compute y range from function
        x_vals = np.linspace(self.x_min.value(), self.x_max.value(), 500)
        y_vals = self.engine.evaluate(x_vals)
        finite_y = y_vals[np.isfinite(y_vals)]
        
        if len(finite_y) > 0:
            y_min = float(np.min(finite_y))
            y_max = float(np.max(finite_y))
            margin = (y_max - y_min) * 0.1
            if margin == 0:
                margin = 1
            self.y_min.setValue(y_min - margin)
            self.y_max.setValue(y_max + margin)
        
        self.on_view_changed()


def main():
    app = QApplication(sys.argv)
    
    # Set application font
    font = QFont('Segoe UI', 9)
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()