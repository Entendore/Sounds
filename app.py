import sys
import numpy as np
import sounddevice as sd
import queue
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QSlider, QLabel, QGridLayout)
from PySide6.QtCore import QTimer, Qt
import pyqtgraph as pg

pg.setConfigOption('usePySide6', True)
pg.setConfigOptions(antialias=True)

class NoteAxisItem(pg.AxisItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    def tickStrings(self, values, scale, spacing):
        return [self.notes[int(v) % 12] if 0 <= int(v) < 12 else '' for v in values]

class ShortsReadyVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Shorts & Video Ready - Math of Sound")
        self.setStyleSheet("background-color: #0F0F1A; color: #FFFFFF;")
        
        # Audio & Math Params
        self.fs = 44100
        self.chunk_size = 4096 
        self.time = np.arange(self.chunk_size) / self.fs
        self.current_freq = 440.0
        
        # History Buffers
        self.spec_history = np.zeros((100, self.chunk_size // 2))
        self.chroma_history = np.zeros((100, 12))
        
        # Audio Queue & State
        self.audio_queue = queue.Queue()
        self.current_mode = "Sine"
        self.lissajous_delay = 50 
        self.is_short_mode = False
        
        self.init_widgets()
        self.build_horizontal_layout() # Start in standard YouTube mode
        self.init_audio()
        self.init_timer()

    def init_widgets(self):
        # 1. Big HUD Label (Crucial for Shorts text readability)
        self.hud_label = QLabel("440.0 Hz | A4")
        self.hud_label.setAlignment(Qt.AlignCenter)
        self.hud_label.setStyleSheet("font-size: 48px; font-weight: bold; color: #00BFFF; font-family: Arial;")
        
        # 2. Reactor Circle (Visual hook for Short-form)
        self.reactor_plot = pg.PlotWidget()
        self.reactor_plot.setBackground('#0F0F1A')
        self.reactor_plot.setAspectLocked(True)
        self.reactor_plot.setXRange(-1.5, 1.5)
        self.reactor_plot.setYRange(-1.5, 1.5)
        self.reactor_plot.hideAxis('left')
        self.reactor_plot.hideAxis('bottom')
        self.reactor_curve = self.reactor_plot.plot(pen=pg.mkPen(color='#FF00FF', width=4))
        
        # 3. Time Domain (Waveform)
        self.time_plot = pg.PlotWidget(title="Time Domain")
        self.time_plot.setBackground('#0F0F1A')
        self.time_plot.showGrid(x=True, y=True, alpha=0.2)
        self.time_plot.setYRange(-1.5, 1.5)
        self.time_curve = self.time_plot.plot(pen=pg.mkPen(color='#00BFFF', width=3))
        
        # 4. Phase Space (Lissajous)
        self.phase_plot = pg.PlotWidget(title="Phase Space")
        self.phase_plot.setBackground('#0F0F1A')
        self.phase_plot.setAspectLocked(True)
        self.phase_plot.setXRange(-1.5, 1.5)
        self.phase_plot.setYRange(-1.5, 1.5)
        self.phase_curve = self.phase_plot.plot(pen=pg.mkPen(color='#FFAA00', width=2))
        
        # 5. Frequency Spectrum (FFT)
        self.freq_plot = pg.PlotWidget(title="FFT")
        self.freq_plot.setBackground('#0F0F1A')
        self.freq_plot.showGrid(x=True, y=True, alpha=0.2)
        self.freq_plot.setLogMode(x=True, y=False)
        self.freq_plot.setXRange(np.log10(20), np.log10(20000), padding=0)
        self.freq_plot.setYRange(0, 0.1)
        self.freq_curve = self.freq_plot.plot(pen=pg.mkPen(color='#FF55FF', width=3))
        
        # 6. Chromagram
        chroma_axis = NoteAxisItem(orientation='left')
        self.chroma_plot = pg.PlotWidget(title="Chromagram", axisItems={'left': chroma_axis})
        self.chroma_plot.setBackground('#0F0F1A')
        self.chroma_plot.setYRange(0, 11, padding=0)
        self.chroma_image = pg.ImageItem()
        self.chroma_plot.addItem(self.chroma_image)
        self.chroma_image.setColorMap(pg.colormap.get('viridis'))
        
        # 7. Spectrogram
        self.spec_plot = pg.PlotWidget(title="Spectrogram")
        self.spec_plot.setBackground('#0F0F1A')
        self.spec_plot.setLogMode(x=True, y=False)
        self.spec_image = pg.ImageItem()
        self.spec_plot.addItem(self.spec_image)
        self.spec_image.setColorMap(pg.colormap.get('inferno'))
        
        # --- Controls ---
        self.btn_mic = QPushButton("🎙️ Mic")
        self.btn_sine = QPushButton("〰️ Sine")
        self.btn_harmonics = QPushButton("🎹 Harmonics")
        self.btn_short = QPushButton("📱 Toggle Short Mode (9:16)")
        self.btn_hide = QPushButton("👁️ Hide UI")
        
        for btn in [self.btn_mic, self.btn_sine, self.btn_harmonics, self.btn_short, self.btn_hide]:
            btn.setStyleSheet("background-color: #2A2A40; padding: 15px; font-size: 14px; border-radius: 5px;")
            
        self.btn_mic.clicked.connect(lambda: self.set_mode("Mic"))
        self.btn_sine.clicked.connect(lambda: self.set_mode("Sine"))
        self.btn_harmonics.clicked.connect(lambda: self.set_mode("Harmonics"))
        self.btn_short.clicked.connect(self.toggle_short_mode)
        self.btn_hide.clicked.connect(self.hide_ui)
        
        # --- Slider ---
        self.freq_slider = QSlider(Qt.Horizontal)
        self.freq_slider.setMinimum(20)
        self.freq_slider.setMaximum(2000)
        self.freq_slider.setValue(440)
        self.freq_slider.valueChanged.connect(self.update_frequency)

    def clear_layouts(self):
        # Helper to safely destroy current layout before rebuilding
        if self.centralWidget().layout() is not None:
            widget = self.centralWidget()
            layout = widget.layout()
            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().setParent(None)
            # Delete the layout itself
            QWidget().setLayout(layout)

    def build_horizontal_layout(self):
        self.is_short_mode = False
        self.resize(1280, 720) # 16:9
        self.clear_layouts()
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        grid = QGridLayout()
        
        # Row 0: Top plots
        grid.addWidget(self.time_plot, 0, 0)
        grid.addWidget(self.phase_plot, 0, 1)
        # Row 1: Middle plots
        grid.addWidget(self.freq_plot, 1, 0)
        grid.addWidget(self.chroma_plot, 1, 1)
        # Row 2: Bottom full-width
        grid.addWidget(self.spec_plot, 2, 0, 1, 2)
        
        main_layout.addLayout(grid, stretch=4)
        
        # Controls
        controls = QHBoxLayout()
        for w in [self.btn_mic, self.btn_sine, self.btn_harmonics, self.btn_short, self.btn_hide]:
            controls.addWidget(w)
        main_layout.addLayout(controls, stretch=1)
        
        # Slider
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(self.hud_label)
        slider_layout.addWidget(self.freq_slider)
        main_layout.addLayout(slider_layout, stretch=1)

    def build_vertical_layout(self):
        self.is_short_mode = True
        self.resize(540, 960) # 9:16 (scaled down for desktop window)
        self.clear_layouts()
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # In Shorts, we prioritize the most colorful/reactive elements
        main_layout.addWidget(self.hud_label, stretch=1)    # Top: Big Text
        main_layout.addWidget(self.spec_plot, stretch=3)   # Upper Mid: Spectrogram
        main_layout.addWidget(self.chroma_plot, stretch=2) # Lower Mid: Chromagram
        main_layout.addWidget(self.phase_plot, stretch=2)  # Bottom Mid: Lissajous
        main_layout.addWidget(self.time_plot, stretch=2)   # Lower: Waveform
        
        # Controls overlaid at the very bottom
        controls = QHBoxLayout()
        for w in [self.btn_mic, self.btn_sine, self.btn_harmonics, self.btn_short, self.btn_hide]:
            controls.addWidget(w)
        main_layout.addLayout(controls, stretch=1)

    def toggle_short_mode(self):
        if self.is_short_mode:
            self.build_horizontal_layout()
        else:
            self.build_vertical_layout()

    def update_frequency(self, val):
        self.current_freq = float(val)
        note = self.freq_to_note(self.current_freq)
        self.hud_label.setText(f"{self.current_freq:.1f} Hz | {note}")
        if self.current_freq > 0:
            self.lissajous_delay = int(self.fs / self.current_freq)

    def freq_to_note(self, freq):
        if freq == 0: return "N/A"
        A4 = 440.0
        C0 = A4 * pow(2, -4.75)
        name = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        half_steps = round(12 * np.log2(freq / C0))
        octave = half_steps // 12
        n = half_steps % 12
        return f"{name[n]}{octave}"

    def hide_ui(self):
        # Hides buttons/slider for clean recording
        # Finds the last two layouts added and hides their widgets
        main_layout = self.centralWidget().layout()
        if main_layout.count() >= 3:
            for i in [main_layout.count()-1, main_layout.count()-2]:
                item = main_layout.itemAt(i)
                if item.layout():
                    for j in range(item.layout().count()):
                        w = item.layout().itemAt(j).widget()
                        if w: w.setVisible(not w.isVisible())

    def init_audio(self):
        try:
            self.stream = sd.InputStream(
                samplerate=self.fs, channels=1, dtype='float32',
                blocksize=self.chunk_size, callback=self.audio_callback
            )
            self.stream.start()
        except Exception as e:
            print(f"Microphone error: {e}")
            self.current_mode = "Sine"

    def audio_callback(self, indata, frames, time, status):
        if status: print(status)
        self.audio_queue.put(indata[:, 0].copy())

    def set_mode(self, mode):
        self.current_mode = mode
        styles = {"Mic": "#0066CC", "Sine": "#009933", "Harmonics": "#990099"}
        for btn, m in [(self.btn_mic, "Mic"), (self.btn_sine, "Sine"), (self.btn_harmonics, "Harmonics")]:
            btn.setStyleSheet(f"background-color: {styles[m] if m==mode else '#2A2A40'}; padding: 15px; font-size: 14px; border-radius: 5px;")

    def compute_chroma(self, fft_magnitudes, freqs):
        valid_idx = np.where((freqs > 20) & (freqs < 5000))[0]
        valid_freqs = freqs[valid_idx]
        valid_mags = fft_magnitudes[valid_idx]
        midi_notes = 69 + 12 * np.log2(valid_freqs / 440.0)
        pitch_classes = np.mod(np.round(midi_notes).astype(int), 12)
        chroma = np.zeros(12)
        for pc in range(12):
            chroma[pc] = np.sum(valid_mags[pitch_classes == pc])
        return chroma

    def process_data(self, audio_data):
        # 1. Time Domain
        self.time_curve.setData(self.time, audio_data)
        
        # 2. Phase Space
        if len(audio_data) > self.lissajous_delay:
            self.phase_curve.setData(audio_data[:-self.lissajous_delay], audio_data[self.lissajous_delay:])
            
        # 3. Reactor Circle (RMS Volume pulse)
        rms = np.sqrt(np.mean(audio_data**2))
        t_circle = np.linspace(0, 2*np.pi, 100)
        radius = 0.5 + rms * 5.0 # Pulse based on volume
        x_circ = radius * np.cos(t_circle)
        y_circ = radius * np.sin(t_circle)
        self.reactor_curve.setData(x_circ, y_circ)
        
        # 4. FFT Math
        window = np.hanning(len(audio_data))
        fft_result = np.fft.rfft(audio_data * window)
        fft_magnitudes = np.abs(fft_result) / self.chunk_size
        freqs = np.fft.rfftfreq(self.chunk_size, 1.0 / self.fs)
        
        self.freq_curve.setData(freqs, fft_magnitudes)
        
        # 5. Spectrogram
        self.spec_history = np.roll(self.spec_history, -1, axis=0)
        self.spec_history[-1] = fft_magnitudes
        self.spec_image.setImage(self.spec_history, levels=(0, 0.02))
        
        # 6. Chromagram
        chroma = self.compute_chroma(fft_magnitudes, freqs)
        self.chroma_history = np.roll(self.chroma_history, -1, axis=0)
        self.chroma_history[-1] = chroma
        self.chroma_image.setImage(self.chroma_history.T, levels=(0, np.max(chroma)*2 + 0.001))
        
        # 7. Pitch Detection
        if self.current_mode == "Mic":
            peak_idx = np.argmax(fft_magnitudes[1:]) + 1
            peak_freq = freqs[peak_idx]
            if fft_magnitudes[peak_idx] > 0.005:
                note = self.freq_to_note(peak_freq)
                self.hud_label.setText(f"{peak_freq:.1f} Hz | {note}")
            else:
                self.hud_label.setText("Silence...")

    def update_loop(self):
        if self.current_mode == "Mic":
            try:
                self.process_data(self.audio_queue.get_nowait())
            except queue.Empty:
                pass
        else:
            if not hasattr(self, 'math_time'): self.math_time = 0
            t = np.linspace(self.math_time, self.math_time + (self.chunk_size/self.fs), self.chunk_size)
            self.math_time += self.chunk_size / self.fs
            
            f = self.current_freq
            if self.current_mode == "Sine":
                audio_data = 0.8 * np.sin(2 * np.pi * f * t)
            else: # Harmonics
                h1 = 0.6 * np.sin(2 * np.pi * f * t)
                h2 = 0.3 * np.sin(2 * np.pi * (2*f) * t)
                h3 = 0.15 * np.sin(2 * np.pi * (3*f) * t)
                audio_data = h1 + h2 + h3
                
            self.process_data(audio_data)

    def init_timer(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_loop)
        self.timer.start(16)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ShortsReadyVisualizer()
    window.show()
    sys.exit(app.exec())