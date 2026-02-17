import sys
import json
import time
import datetime
from threading import Lock
from enum import Enum
from contextlib import contextmanager

# GUI Imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QPushButton, QTextEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QMessageBox, QTabWidget, QGroupBox, QFileDialog,
    QSplitter, QProgressBar, QCheckBox
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer, QSettings
from PyQt6.QtGui import QFont

# Serial Imports
import serial
import serial.tools.list_ports


# ==================== CONFIGURATION ====================
class Config:
    """Centralized configuration"""
    DEFAULT_BAUD = 115200

    # Reduced timeout to prevent UI blocking
    RUNTIME_TIMEOUT = 10
    SERIAL_TIMEOUT = 0.1  

    # FIX: Defaults changed to False (Standard for CP210x/CH340/FTDI drivers)
    DTR_INVERTED_DEFAULT = False
    RTS_INVERTED_DEFAULT = False

    # I2C scan range (inclusive).
    I2C_SCAN_RANGE = (1, 126)

    # I2C Device Database
    I2C_DEVICES = {
        0x1C: "ADXL345 Accel", 0x1D: "ADXL345 Accel (Alt)",
        0x1E: "HMC5883L Compass", 0x20: "PCF8574 I/O Expander",
        0x23: "BH1750 Light", 0x27: "LCD Display",
        0x34: "AXP192 Power", 0x35: "AXP202 Power",
        0x38: "PCF8574A I/O", 0x39: "TSL2561 Light",
        0x3C: "OLED Display (SSD1306)", 0x3D: "OLED Display (Alt)",
        0x40: "Si7021 Temp/Humid", 0x48: "ADS1115 ADC",
        0x4A: "MAX44009 Light", 0x50: "EEPROM (24C32)",
        0x51: "RTC DS3231", 0x52: "EEPROM (Alt)",
        0x53: "ADXL345 (Alt)", 0x57: "EEPROM (Alt)",
        0x5A: "MPR121 Touch", 0x68: "MPU6050 IMU / RTC DS1307",
        0x69: "MPU6050 (Alt)", 0x76: "BME280 Temp/Press/Humid",
        0x77: "BME280 (Alt) / BMP180",
    }

    @staticmethod
    def get_safe_gpio_pins(chip_name: str) -> list[int]:
        gpio_map = {
            'ESP32':    [4, 5, 13, 14, 16, 17, 18, 19, 21, 22, 23, 25, 26, 27, 32, 33],
            'ESP32-S2': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 33, 34, 35, 36, 37, 38],
            'ESP32-S3': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 35, 36, 37, 38],
            'ESP32-C3': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 18, 19],
            'ESP32-C6': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 18, 19],
            'ESP32-H2': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
        return gpio_map.get(chip_name, [4, 5, 13, 14])


# ==================== WORKFLOW STATE ====================
class WorkflowState(Enum):
    INITIAL         = 0  
    AGENT_SAVED     = 1  
    AGENT_UPLOADED  = 2  
    RUNTIME_READY   = 3  


# ==================== UTILITY FUNCTIONS ====================
@contextmanager
def serial_port_safe(port, baud=115200, timeout=0.1):
    ser = None
    try:
        ser = serial.Serial(port, baudrate=baud, timeout=timeout)
        yield ser
    finally:
        if ser and ser.is_open:
            try:
                ser.close()
            except Exception:
                pass


def port_present(port: str) -> bool:
    try:
        ports = [p.device for p in serial.tools.list_ports.comports()]
        return port in ports
    except Exception:
        return False


def _apply_inversion(asserted: bool, inverted: bool) -> bool:
    return (not asserted) if inverted else asserted


def set_reset_lines(ser: serial.Serial, *, en_reset: bool, gpio0_boot: bool, dtr_inverted: bool, rts_inverted: bool):
    ser.rts = _apply_inversion(en_reset, rts_inverted)
    ser.dtr = _apply_inversion(gpio0_boot, dtr_inverted)


def perform_reset(ser: serial.Serial, strategy: str, *, dtr_inverted: bool, rts_inverted: bool) -> bool:
    try:
        # Strategy "classic": Standard NodeMCU reset (DTR=GPIO0, RTS=EN)
        if strategy == 'classic':
            # 1. Assert EN (Reset) while keeping GPIO0 High (Boot)
            set_reset_lines(ser, en_reset=True,  gpio0_boot=False, dtr_inverted=dtr_inverted, rts_inverted=rts_inverted)
            time.sleep(0.1)
            # 2. Release EN (Run)
            set_reset_lines(ser, en_reset=False, gpio0_boot=False, dtr_inverted=dtr_inverted, rts_inverted=rts_inverted)

        # Strategy "usb_jtag": Pulse both lines (common for native USB PHY)
        elif strategy == 'usb_jtag':
            ser.dtr = False
            ser.rts = False
            time.sleep(0.1)
            ser.dtr = True
            ser.rts = True
            time.sleep(0.1)
            ser.dtr = False
            ser.rts = False

        time.sleep(0.5)
        return True
    except Exception as e:
        print(f"Reset failed: {e}")
        return False


def _brace_match_slice(buffer: str, start: int) -> str | None:
    depth = 0
    in_str = False
    esc = False

    for i in range(start, len(buffer)):
        ch = buffer[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return buffer[start:i + 1]
    return None


def extract_first_json_object(buffer: str, marker: str) -> tuple[dict | None, str]:
    """
    Robust JSON extractor. 
    FIX: Logic improved to avoid aggressively trimming headers if JSON isn't complete yet.
    """
    idx = buffer.find(marker)
    if idx < 0:
        # Keep the last portion of the buffer to avoid cutting a partial marker
        if len(buffer) > 65536:
            buffer = buffer[-4096:] 
        return None, buffer

    # Look backwards from the marker to find the opening '{'
    candidates = []
    pos = idx
    for _ in range(64): # Scan backwards a reasonable amount
        pos = buffer.rfind("{", 0, pos)
        if pos < 0:
            break
        candidates.append(pos)
        if pos == 0:
            break
        pos -= 1

    for start in candidates:
        slice_ = _brace_match_slice(buffer, start)
        if not slice_:
            continue
        try:
            obj = json.loads(slice_)
        except json.JSONDecodeError:
            continue
        
        # Verify the marker is actually inside this valid JSON object
        if marker not in slice_:
            continue

        end_index = start + len(slice_)
        remaining = buffer[end_index:]
        return obj, remaining

    # If we found the marker but couldn't parse a valid object yet,
    # we must be careful not to trim the start of the object.
    # Keep buffer starting from the earliest candidate brace.
    if candidates:
        earliest = min(candidates)
        trimmed = buffer[earliest:]
    else:
        # Marker found but no brace? Weird. Just keep near the marker.
        trimmed = buffer[max(0, idx - 100):]
        
    return None, trimmed


# ==================== WORKER THREAD ====================
class RuntimeAgent(QThread):
    data_received = pyqtSignal(dict)
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)

    def __init__(self, port: str, chip_type: str, strategy: str, dtr_inverted: bool, rts_inverted: bool):
        super().__init__()
        self.port = port
        self.chip_type = chip_type
        self.strategy = strategy # FIX: Strategy passed from UI
        self.dtr_inverted = dtr_inverted
        self.rts_inverted = rts_inverted
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        ser = None
        try:
            if not port_present(self.port):
                self.log_signal.emit(f"Port {self.port} is not present.")
                return

            self.progress_signal.emit(10)

            # FIX: Use short timeout for non-blocking reads
            try:
                ser = serial.Serial(self.port, Config.DEFAULT_BAUD, timeout=Config.SERIAL_TIMEOUT)
            except Exception as e:
                self.log_signal.emit(f"Port open failed: {e}")
                return

            with ser:
                self.log_signal.emit(
                    f"Reset strategy={self.strategy} | DTR inv={self.dtr_inverted} | RTS inv={self.rts_inverted}"
                )
                self.log_signal.emit("Resetting board...")
                self.progress_signal.emit(30)

                if not perform_reset(
                    ser, self.strategy,
                    dtr_inverted=self.dtr_inverted,
                    rts_inverted=self.rts_inverted
                ):
                    self.log_signal.emit("Warning: Reset logic encountered error")

                try:
                    ser.reset_input_buffer()
                except Exception:
                    pass

                self.progress_signal.emit(50)

                start_time = time.time()
                buffer = ""

                # FIX: Loop checks _is_running constantly
                while self._is_running:
                    # Check timeout
                    if (time.time() - start_time) > Config.RUNTIME_TIMEOUT:
                        self.log_signal.emit("Timeout: No agent report received.")
                        break

                    # Update progress
                    elapsed = time.time() - start_time
                    progress = 50 + int((elapsed / Config.RUNTIME_TIMEOUT) * 50)
                    self.progress_signal.emit(min(progress, 98))

                    # Read non-blocking
                    try:
                        # read() respects timeout, or reads available bytes
                        chunk = ser.read(ser.in_waiting or 1).decode("utf-8", errors="ignore")
                        if chunk:
                            buffer += chunk
                            
                            # Fast path for simple JSON lines
                            if '"type":"audit_report"' in chunk or '"type":"audit_report"' in buffer:
                                # Try full extraction
                                obj, buffer = extract_first_json_object(buffer, '"type":"audit_report"')
                                if obj is not None:
                                    self.progress_signal.emit(100)
                                    self.data_received.emit(obj)
                                    return
                    except Exception:
                        pass
                    
                    # Prevent CPU spinning
                    time.sleep(0.01)

        except Exception as e:
            if self._is_running:
                self.log_signal.emit(f"Runtime scan error: {e}")


# ==================== MAIN WINDOW ====================
class ESPDetective(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ESP32 Board Detective v2.7 (Fixed)")
        self.resize(900, 750)

        self.workflow_state = WorkflowState.INITIAL
        self.runtime_data = {}

        self.chip_type = "ESP32"
        self.reset_strategy = "classic" # Default

        self.dtr_inverted = Config.DTR_INVERTED_DEFAULT
        self.rts_inverted = Config.RTS_INVERTED_DEFAULT

        self._agent_saved = False

        self.workers = []
        self.workers_lock = Lock()
        self.settings = QSettings("ESPDetective", "BoardAnalyzer")

        self._runtime_worker: RuntimeAgent | None = None

        self.setup_ui()
        self.update_workflow_state(WorkflowState.INITIAL)
        self.restore_settings()

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        self.setup_workflow_indicator(main_layout)
        self.setup_port_selector(main_layout)

        splitter = QSplitter(Qt.Orientation.Vertical)
        self.tabs = QTabWidget()
        self.setup_tabs()
        splitter.addWidget(self.tabs)

        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        log_header = QHBoxLayout()
        log_header.addWidget(QLabel("<b>System Log:</b>"))
        log_header.addStretch()
        self.btn_clear_log = QPushButton("Clear Log")
        self.btn_clear_log.clicked.connect(lambda: self.log_area.clear())
        log_header.addWidget(self.btn_clear_log)
        log_layout.addLayout(log_header)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setFont(QFont("Consolas", 9))
        log_layout.addWidget(self.log_area)

        splitter.addWidget(log_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

    def setup_workflow_indicator(self, parent_layout):
        group = QGroupBox("Workflow Progress")
        layout = QHBoxLayout()
        self.step_labels = []
        steps = ["1. Generate Agent", "2. Upload via Arduino IDE", "3. Runtime Inspect"]

        for i, step_text in enumerate(steps):
            label = QLabel(step_text)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("padding: 8px; border: 1px solid #ccc; border-radius: 3px;")
            self.step_labels.append(label)
            layout.addWidget(label)
            if i < len(steps) - 1:
                layout.addWidget(QLabel("→"))

        group.setLayout(layout)
        parent_layout.addWidget(group)

    def setup_port_selector(self, parent_layout):
        layout = QHBoxLayout()
        layout.addWidget(QLabel("<b>Target Port:</b>"))
        self.port_combo = QComboBox()
        self.port_combo.setMinimumWidth(200)
        layout.addWidget(self.port_combo)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_ports)
        layout.addWidget(self.refresh_btn)
        
        self.port_status = QLabel("?")
        layout.addWidget(self.port_status)

        layout.addSpacing(15)
        
        # Chip Selection
        layout.addWidget(QLabel("<b>Chip:</b>"))
        self.chip_combo = QComboBox()
        self.chip_combo.addItems([
            "ESP32", "ESP32-S2", "ESP32-S3", "ESP32-C3", "ESP32-C6", "ESP32-H2"
        ])
        self.chip_combo.currentTextChanged.connect(self._on_chip_changed)
        layout.addWidget(self.chip_combo)

        layout.addSpacing(15)

        # FIX: Reset Method Selector
        layout.addWidget(QLabel("<b>Reset Method:</b>"))
        self.reset_combo = QComboBox()
        self.reset_combo.addItems(["Classic (UART)", "USB-JTAG (Native)"])
        self.reset_combo.currentTextChanged.connect(self._on_reset_strategy_changed)
        layout.addWidget(self.reset_combo)

        layout.addSpacing(15)

        # Inversion Controls
        self.cb_dtr_inv = QCheckBox("Inv DTR")
        self.cb_dtr_inv.setToolTip("Invert DTR (GPIO0) polarity")
        self.cb_dtr_inv.setChecked(self.dtr_inverted)
        self.cb_dtr_inv.stateChanged.connect(self._on_inversion_changed)
        layout.addWidget(self.cb_dtr_inv)

        self.cb_rts_inv = QCheckBox("Inv RTS")
        self.cb_rts_inv.setToolTip("Invert RTS (EN) polarity")
        self.cb_rts_inv.setChecked(self.rts_inverted)
        self.cb_rts_inv.stateChanged.connect(self._on_inversion_changed)
        layout.addWidget(self.cb_rts_inv)

        layout.addStretch()
        parent_layout.addLayout(layout)

        self.refresh_ports()
        self.port_timer = QTimer()
        self.port_timer.timeout.connect(self.check_port_status)
        self.port_timer.start(2000)
        self.port_combo.currentTextChanged.connect(lambda _: self.check_port_status())

    def _on_chip_changed(self, chip_name: str):
        self.chip_type = chip_name

    def _on_reset_strategy_changed(self, text: str):
        if "Classic" in text:
            self.reset_strategy = "classic"
        else:
            self.reset_strategy = "usb_jtag"
        self.log(f"Reset strategy set to: {self.reset_strategy}")

    def _on_inversion_changed(self):
        self.dtr_inverted = self.cb_dtr_inv.isChecked()
        self.rts_inverted = self.cb_rts_inv.isChecked()
        self.log(f"Reset inversion: DTR={self.dtr_inverted}, RTS={self.rts_inverted}")

    def setup_tabs(self):
        self.tab1 = QWidget()
        self.setup_tab1()
        self.tabs.addTab(self.tab1, "1. Generate Agent")

        self.tab2 = QWidget()
        self.setup_tab2()
        self.tabs.addTab(self.tab2, "2. Runtime Inspector")

    def setup_tab1(self):
        layout = QVBoxLayout()
        layout.addWidget(QLabel(
            "<b>Step 1:</b> Configure and save the Hardware Agent sketch, "
            "then open it in Arduino IDE and flash it to your board."
        ))

        options_group = QGroupBox("Agent Options")
        options_layout = QVBoxLayout()
        self.cb_i2c_scan = QCheckBox("Enable I2C Scan")
        self.cb_i2c_scan.setChecked(True)
        options_layout.addWidget(self.cb_i2c_scan)

        self.cb_gpio_detect = QCheckBox("Enable GPIO Heuristic")
        self.cb_gpio_detect.setChecked(True)
        options_layout.addWidget(self.cb_gpio_detect)

        self.cb_power_monitor = QCheckBox("Detect Power Monitor (AXP)")
        self.cb_power_monitor.setChecked(True)
        options_layout.addWidget(self.cb_power_monitor)
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

        self.btn_save = QPushButton("Save 'ESP_Agent.ino'")
        self.btn_save.clicked.connect(self.save_sketch)
        layout.addWidget(self.btn_save)

        layout.addWidget(QLabel(
            "After saving, open the .ino file in Arduino IDE, select your board and port, "
            "then click Upload. Once flashing is complete, click the button below."
        ))

        self.btn_mark_uploaded = QPushButton("✓  I have uploaded the agent (mark Step 2 complete)")
        self.btn_mark_uploaded.clicked.connect(self.mark_agent_uploaded)
        self.btn_mark_uploaded.setEnabled(False)
        layout.addWidget(self.btn_mark_uploaded)

        layout.addStretch()
        self.tab1.setLayout(layout)

    def save_sketch(self):
        fname, _ = QFileDialog.getSaveFileName(
            self, "Save Agent", "ESP_Agent.ino", "Arduino (*.ino)"
        )
        if fname:
            try:
                with open(fname, 'w', encoding="utf-8") as f:
                    f.write(self.generate_agent_code())
            except Exception as e:
                self.handle_error(f"Failed to save sketch:\n{e}")
                return

            self.log(f"Saved to {fname}")
            self._agent_saved = True
            self.update_workflow_state(WorkflowState.AGENT_SAVED)
            self.btn_mark_uploaded.setEnabled(True)
            QMessageBox.information(
                self, "Saved",
                "File saved.\n\nOpen it in Arduino IDE, select your board/port, and click Upload.\n"
                "Return here and click 'I have uploaded the agent' when done."
            )

    def mark_agent_uploaded(self):
        self.update_workflow_state(WorkflowState.AGENT_UPLOADED)
        self.tabs.setCurrentIndex(1)

    def generate_agent_code(self) -> str:
        i2c_enabled   = self.cb_i2c_scan.isChecked()
        gpio_enabled  = self.cb_gpio_detect.isChecked()
        power_enabled = self.cb_power_monitor.isChecked()
        chip_name     = self.chip_type
        safe_pins     = Config.get_safe_gpio_pins(chip_name)
        safe_pins_str = ", ".join(str(p) for p in safe_pins)

        code = r'''/* ESP32 Hardware Audit Agent v2.7 */
#include <Wire.h>
#include <Arduino.h>
#include "esp_chip_info.h"
#include "esp_system.h"

static void print_bool(bool v) { Serial.print(v ? "true" : "false"); }

static String i2c_devices = "[]";
static bool axp_detected = false;
static int axp_addr = -1;
static String gpio_pullups = "[]";
static String gpio_pulldowns = "[]";

'''
        if gpio_enabled:
            code += r'''static bool stable_read(int pin, int mode, bool expect_high) {
  pinMode(pin, mode);
  delay(2);
  int ok = 0;
  for (int i = 0; i < 6; i++) {
    int v = digitalRead(pin);
    if (expect_high ? v : !v) ok++;
    delay(1);
  }
  pinMode(pin, INPUT);
  return ok >= 5;
}

'''
        code += r'''static void emit_report(const esp_chip_info_t& chip_info, uint32_t flash_size) {
  Serial.print("{\"type\":\"audit_report\"");
  Serial.print(",\"model_id\":"); Serial.print((int)chip_info.model);
  Serial.print(",\"cores\":"); Serial.print((int)chip_info.cores);
  Serial.print(",\"revision\":"); Serial.print((int)chip_info.revision);
  Serial.print(",\"flash_bytes\":"); Serial.print((uint32_t)flash_size);
  Serial.print(",\"i2c_found\":"); Serial.print(i2c_devices);
  Serial.print(",\"gpio_pullups\":"); Serial.print(gpio_pullups);
  Serial.print(",\"gpio_pulldowns\":"); Serial.print(gpio_pulldowns);
  Serial.print(",\"axp_detected\":"); print_bool(axp_detected);
  Serial.print(",\"axp_addr\":"); Serial.print(axp_addr);
  Serial.println("}");
}

void setup() {
  Serial.begin(115200);
  delay(400);

  esp_chip_info_t chip_info;
  esp_chip_info(&chip_info);

  uint32_t flash_size = 0;
#if defined(ARDUINO_ARCH_ESP32)
  flash_size = ESP.getFlashChipSize();
#endif

'''
        if i2c_enabled:
            code += rf'''
  Wire.begin();
  i2c_devices = "[";
  int i2c_count = 0;
  for (uint8_t address = {Config.I2C_SCAN_RANGE[0]}; address <= {Config.I2C_SCAN_RANGE[1]}; address++) {{
    Wire.beginTransmission(address);
    if (Wire.endTransmission() == 0) {{
      if (i2c_count > 0) i2c_devices += ",";
      i2c_devices += String((int)address);
      i2c_count++;
'''
            if power_enabled:
                code += r'''      if (address == 0x34 || address == 0x35) {
        axp_detected = true;
        axp_addr = (int)address;
      }
'''
            code += r'''    }
  }
  i2c_devices += "]";
'''

        if gpio_enabled:
            code += rf'''
  const int gpio_pins[] = {{{safe_pins_str}}};
  const int num_pins = sizeof(gpio_pins) / sizeof(gpio_pins[0]);
  if (num_pins > 0) {{
      int pullups[64];  int pu_n = 0;
      int pulldowns[64]; int pd_n = 0;
    
      for (int i = 0; i < num_pins; i++) {{
        int pin = gpio_pins[i];
        if (stable_read(pin, INPUT_PULLUP, true)  && pu_n < 64) pullups[pu_n++]    = pin;
    #if defined(ARDUINO_ARCH_ESP32)
        if (stable_read(pin, INPUT_PULLDOWN, false) && pd_n < 64) pulldowns[pd_n++] = pin;
    #endif
      }}
    
      gpio_pullups = "[";
      for (int i = 0; i < pu_n; i++) {{ if (i) gpio_pullups += ","; gpio_pullups += String(pullups[i]); }}
      gpio_pullups += "]";
    
      gpio_pulldowns = "[";
      for (int i = 0; i < pd_n; i++) {{ if (i) gpio_pulldowns += ","; gpio_pulldowns += String(pulldowns[i]); }}
      gpio_pulldowns += "]";
  }}
'''

        code += r'''
  // Emit immediately once setup is complete
  emit_report(chip_info, flash_size);
}

void loop() {
  // Re-emit once per second so the host can't miss it.
  static uint32_t last_ms = 0;
  uint32_t now = millis();
  if (now - last_ms >= 1000) {
    last_ms = now;

    esp_chip_info_t chip_info;
    esp_chip_info(&chip_info);

    uint32_t flash_size = 0;
#if defined(ARDUINO_ARCH_ESP32)
    flash_size = ESP.getFlashChipSize();
#endif

    emit_report(chip_info, flash_size);
  }
}
'''
        return code

    def setup_tab2(self):
        layout = QVBoxLayout()
        layout.addWidget(QLabel("<b>Step 3:</b> Click Run to read the agent report from your board over serial."))

        self.btn_runtime = QPushButton("Run Runtime Scan")
        self.btn_runtime.clicked.connect(self.run_runtime)
        layout.addWidget(self.btn_runtime)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(2)
        self.results_table.setHorizontalHeaderLabels(["Component", "Details"])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.results_table)

        self.btn_export = QPushButton("Export JSON")
        self.btn_export.clicked.connect(self.export_results)
        self.btn_export.setEnabled(False)
        layout.addWidget(self.btn_export)

        self.tab2.setLayout(layout)

    def run_runtime(self):
        if self._runtime_worker is not None and self._runtime_worker.isRunning():
            QMessageBox.information(self, "Busy", "A runtime scan is already running. Please wait for it to finish.")
            return

        port = self.port_combo.currentText().strip()
        if not port:
            return

        self.set_ui_busy(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # FIX: Pass all config to worker
        worker = RuntimeAgent(
            port, self.chip_type, self.reset_strategy,
            self.dtr_inverted, self.rts_inverted
        )
        self._runtime_worker = worker

        self._add_worker(worker)
        worker.log_signal.connect(self.log)
        worker.data_received.connect(self.display_runtime)
        worker.progress_signal.connect(self.progress_bar.setValue)
        worker.finished.connect(self.runtime_finished)
        worker.finished.connect(lambda w=worker: self._remove_worker(w))
        worker.finished.connect(self._runtime_worker_finished)
        worker.start()

    def _runtime_worker_finished(self):
        self._runtime_worker = None

    def display_runtime(self, data):
        self.runtime_data = data
        self.log("Report received!")

        i2c_raw = data.get('i2c_found', [])
        if isinstance(i2c_raw, list):
            i2c_names = [f"0x{addr:02X} ({Config.I2C_DEVICES.get(addr, 'Unknown')})" for addr in i2c_raw]
            i2c_display = ", ".join(i2c_names) if i2c_names else "None"
        else:
            i2c_display = str(i2c_raw)
        
        # FIX: Ensure all values are strings for QTableWidgetItem
        rows = [
            ("Cores",            str(data.get('cores', 'N/A'))),
            ("Revision",         str(data.get('revision', 'N/A'))),
            ("Flash",            f"{data.get('flash_bytes', 0) / 1048576:.2f} MB"),
            ("I2C Devices",      i2c_display),
            ("GPIO Pullups",     str(data.get('gpio_pullups', []))),
            ("GPIO Pulldowns",   str(data.get('gpio_pulldowns', []))),
            ("AXP Power",        str(data.get('axp_detected', 'False'))),
        ]
        self.results_table.setRowCount(len(rows))
        for i, (k, v) in enumerate(rows):
            self.results_table.setItem(i, 0, QTableWidgetItem(k))
            self.results_table.setItem(i, 1, QTableWidgetItem(v))

        self.btn_export.setEnabled(True)
        self.update_workflow_state(WorkflowState.RUNTIME_READY)

    def runtime_finished(self):
        self.set_ui_busy(False)
        self.progress_bar.setVisible(False)

    def export_results(self):
        if not self.runtime_data:
            return
        fname, _ = QFileDialog.getSaveFileName(self, "Export", "results.json", "JSON (*.json)")
        if fname:
            try:
                with open(fname, 'w', encoding="utf-8") as f:
                    json.dump(self.runtime_data, f, indent=2)
            except Exception as e:
                self.handle_error(f"Failed to export JSON:\n{e}")

    def _add_worker(self, w):
        with self.workers_lock:
            self.workers.append(w)

    def _remove_worker(self, w):
        with self.workers_lock:
            if w in self.workers:
                self.workers.remove(w)

    def log(self, msg):
        self.log_area.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}")

    def refresh_ports(self):
        cur = self.port_combo.currentText()
        self.port_combo.clear()
        for p in serial.tools.list_ports.comports():
            self.port_combo.addItem(p.device)
        if cur:
            self.port_combo.setCurrentText(cur)
        self.check_port_status()

    def check_port_status(self):
        p = self.port_combo.currentText()
        if p:
            self.port_status.setText("Present" if port_present(p) else "Missing")
        else:
            self.port_status.setText("No port selected")

    def set_ui_busy(self, busy: bool):
        self.btn_runtime.setEnabled(not busy)
        self.btn_save.setEnabled(not busy)
        if hasattr(self, "btn_mark_uploaded"):
            self.btn_mark_uploaded.setEnabled(self._agent_saved and not busy)

        self.cb_dtr_inv.setEnabled(not busy)
        self.cb_rts_inv.setEnabled(not busy)
        self.chip_combo.setEnabled(not busy)
        self.reset_combo.setEnabled(not busy)

    def handle_error(self, msg):
        self.log(f"ERROR: {msg}")
        QMessageBox.critical(self, "Error", msg)

    def update_workflow_state(self, state: WorkflowState):
        self.workflow_state = state
        for i, lbl in enumerate(self.step_labels):
            if i < state.value:
                lbl.setStyleSheet(
                    "border: 2px solid #4CAF50; background: #C8E6C9; "
                    "font-weight: bold; padding: 8px; border-radius: 3px;"
                )
            else:
                lbl.setStyleSheet("border: 1px solid #ccc; padding: 8px; border-radius: 3px;")

    def restore_settings(self):
        geo = self.settings.value("geometry")
        if geo:
            self.restoreGeometry(geo)

    def closeEvent(self, e):
        with self.workers_lock:
            for w in self.workers:
                w.stop()
                w.wait(500)
        self.settings.setValue("geometry", self.saveGeometry())
        e.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ESPDetective()
    window.show()
    sys.exit(app.exec())
