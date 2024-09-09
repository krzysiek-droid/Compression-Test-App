import sys
import math
import chardet
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QFileDialog, QLabel, QVBoxLayout, QWidget,
                             QTableWidget, QTableWidgetItem, QHBoxLayout, QDialog, QFormLayout, QComboBox,
                             QDialogButtonBox, QSplitter, QLineEdit, QCheckBox, QGridLayout, QLayoutItem)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from matplotlib import pyplot as plt

class PlotSettingsDialog(QDialog):
    def init(self, parent: QWidget = None):
        super(PlotSettingsDialog, self).__init__()
        print(f"PlotSettingsDialog initialization... args: ()")
        self.setWindowTitle("Ustawienia Wykresu")
        self.columns = None

    def init_ui(self):
        layout = QFormLayout(self)

        self.x_combobox = QComboBox(self)
        self.y_combobox = QComboBox(self)
        self.x_combobox.addItems(self.columns)
        self.y_combobox.addItems(self.columns)

        layout.addRow("Oś X:", self.x_combobox)
        layout.addRow("Oś Y:", self.y_combobox)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def get_settings(self):
        return self.x_combobox.currentText(), self.y_combobox.currentText()

    def pass_column_names(self, column_names: list):
        self.columns = column_names
        self.init_ui()


class PlotControlDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.plot_controls = None
        self.setWindowTitle("Ustawienia Wykresu")
        self.init_ui()

    def init_ui(self):
        layout = QFormLayout(self)

        self.x_min_edit = QLineEdit(self)
        self.x_max_edit = QLineEdit(self)
        self.y_min_edit = QLineEdit(self)
        self.y_max_edit = QLineEdit(self)
        self.x_ticks_edit = QLineEdit(self)
        self.y_ticks_edit = QLineEdit(self)
        self.grid_checkbox = QCheckBox("Pokaż linie pomocnicze", self)

        layout.addRow("Min. oś X:", self.x_min_edit)
        layout.addRow("Max. oś X:", self.x_max_edit)
        layout.addRow("Min. oś Y:", self.y_min_edit)
        layout.addRow("Max. oś Y:", self.y_max_edit)
        layout.addRow("Ticki oś X:", self.x_ticks_edit)
        layout.addRow("Ticki oś Y:", self.y_ticks_edit)
        layout.addRow("", self.grid_checkbox)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        self.setLayout(layout)

    def get_settings(self):
        self.plot_controls = {
            'x_min': self.x_min_edit.text(),
            'x_max': self.x_max_edit.text(),
            'y_min': self.y_min_edit.text(),
            'y_max': self.y_max_edit.text(),
            'x_ticks': self.x_ticks_edit.text(),
            'y_ticks': self.y_ticks_edit.text(),
            'grid': self.grid_checkbox.isChecked()
        }

    def accept(self) -> None:
        self.get_settings()
        super(PlotControlDialog, self).accept()


class CSVViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.zero_point_row = None
        self.sample_info = None
        self.df = None
        self.initial_df = None
        self.regression_line = None
        self.regression_point_1 = None
        self.regression_point_2 = None
        self.motion_conn_id = None
        self.press_conn_id = None
        self.hvlines = []
        self.modified_df = None
        self.current_y_axis = None
        self.current_x_axis = None
        self.layout = QHBoxLayout(self)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.layout.addWidget(self.splitter)

        self.segment_1 = QWidget()
        self.segment_1_layout = QVBoxLayout()
        self.segment_1.setLayout(self.segment_1_layout)


        self.sample_info_widget = QWidget(self)
        self.sample_info_layout = QVBoxLayout(self.sample_info_widget)
        self.segment_1_layout.addWidget(self.sample_info_widget)
        self.sample_info_widget.setFixedSize(300, 150)

        self.sample_info_lbl = QLabel('No sample added yet.')
        self.sample_info_lbl.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.sample_info_lbl.setFixedHeight(40)
        self.sample_info_layout.addWidget(self.sample_info_lbl)

        self.table = QTableWidget(self)
        self.segment_1_layout.addWidget(self.table)
        self.table.setMinimumWidth(300)

        self.plot_widget = QWidget(self)
        self.plot_layout = QVBoxLayout(self.plot_widget)

        self.info_label = QLabel('Load plot data first.')
        self.info_label.setFixedHeight(50)
        self.info_label.setFont(QFont("Calibri", 14))
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.plot_layout.addWidget(self.info_label)

        self.figure, self.ax = plt.subplots()
        self.canvas = self.figure.canvas
        self.hline = None
        self.vline = None
        self.plot_layout.addWidget(self.canvas)

        self.plot_button = QPushButton("Ustawienia Wykresu", self)
        self.plot_button.clicked.connect(self.open_plot_settings)
        self.plot_layout.addWidget(self.plot_button)

        self.plot_control_button = QPushButton("Sterowanie Wykresem", self)
        self.plot_control_button.clicked.connect(self.open_plot_control)
        self.plot_layout.addWidget(self.plot_control_button)

        self.plot_scale_button = QPushButton("Wyzeruj Wykres", self)
        self.plot_scale_button.clicked.connect(self.scale_data)
        self.plot_layout.addWidget(self.plot_scale_button)

        self.plot_yield_strength_button = QPushButton("Wyznacz granicę plastyczności", self)
        self.plot_yield_strength_button.clicked.connect(self.estimate_yield)
        self.plot_layout.addWidget(self.plot_yield_strength_button)

        self.reset_button = QPushButton("Resetuj wykres", self)
        self.reset_button.clicked.connect(self.reset_plot)
        self.plot_layout.addWidget(self.reset_button)

        self.splitter.addWidget(self.segment_1)
        self.splitter.addWidget(self.plot_widget)


    def load_data(self, file_path):
        print(f"cls: CSVViewer, func: load_data [called] file_path={file_path}", end="  ")
        # Wykrywanie kodowania pliku
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
        encoding = result['encoding']
        print(f"encoding={encoding}")
        # Wczytywanie pliku z danymi próby
        trial_info = []
        data_start = 0
        with open(file_path, 'r', encoding=encoding) as file:
            lines = file.readlines()
            for i, line in enumerate(lines):
                if line.startswith("Godzina"):
                    data_start = i
                    break
                trial_info.append(line.strip())
        try:
            self.trial_info = pd.read_csv(file_path, sep=';', encoding=encoding, nrows=data_start)
            # Wyciągnięcie etykiety próbki
            sample_initial_width = float(self.trial_info.to_numpy()[0][0].replace(",", '.'))
            sample_initial_height = float(self.trial_info.to_numpy()[1][0].replace(",", '.'))
            sample_name = list(self.trial_info.columns)[1]
            self.sample_info = {'Sample name': sample_name,
                                'Initial Diameter': sample_initial_width,
                                'Initial Height': sample_initial_height}

            self.sample_info_layout.removeWidget(self.sample_info_lbl)
            self.sample_info_lbl.hide()
            self.sample_info_lbl.deleteLater()
            for key,value in self.sample_info.items():
                new_info_lbl = QLabel(f"{key}: {value}")
                new_info_lbl.setAlignment(Qt.AlignmentFlag.AlignHCenter)
                self.sample_info_layout.addWidget(new_info_lbl)
        except Exception as exc:
            print(f"ERROR during data info loading REASON: {exc}")
            return 0

        # Odczyt danych do DataFrame od miejsca gdzie zaczynają się dane
        try:
            self.df = pd.read_csv(file_path, sep=';', encoding=encoding, skiprows=data_start)
            self.df.columns = ["Time, s", "Strain, %", "Stress, MPa"]
            # Konwersja kolumn na typ float
            for col in self.df.columns:
                self.df[col] = self.df[col].apply(lambda x: str(x).replace(',', '.'))  # Zamiana przecinków na kropki
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                if col == "Strain, %":
                    self.df[col] = self.df[col].apply(lambda x: round(x/self.sample_info['Initial Height']*100, 1))

                if col == "Stress, MPa":
                    sample_area = math.pi * (self.sample_info['Initial Diameter']/2)**2
                    print(f"sample area {sample_area}")
                    self.df[col] = self.df[col].apply(lambda x: round(x/sample_area, 3))
            self.initial_df = self.df
            print("cls: CSVViewer, func: load_data [data loaded successfully]")
        except Exception as e:
            print(f"cls: CSVViewer, func: load_data [error] REASON: {e}")

        self.update_table()
        self.update_plot()

    def update_table(self):
        print(f"cls: CSVViewer, func: update_table [called]", end="  ")
        self.table.setRowCount(0)
        self.table.setColumnCount(0)

        if not self.df.empty:
            self.table.setColumnCount(len(self.df.columns))
            self.table.setHorizontalHeaderLabels(self.df.columns)
            for i, row in self.df.iterrows():
                self.table.insertRow(i)
                for j, value in enumerate(row):
                    self.table.setItem(i, j, QTableWidgetItem(str(value).replace(",", '.')))

            self.table.setMinimumWidth(self.table.columnWidth(1) * self.table.columnCount())

            tableView_columns_width = 0
            for i in range(len(self.df.columns)):
                tableView_columns_width += self.table.columnWidth(i)

            if 450 > tableView_columns_width > 0:
                self.table.setMinimumWidth(tableView_columns_width + 15)
            else:
                self.table.setMinimumWidth(450)

            print("[data displayed]")
        else:
            print("[dataframe is empty]")

    def update_plot(self, x_col_name=None, y_col_name=None):
        print(f"cls: CSVViewer, func: update_plot [called] x_col={x_col_name}, y_col={y_col_name}", end="  ")
        self.figure.clear()

        if x_col_name and y_col_name and not self.df.empty:
            self.figure.set_label(f"Compression test of {self.sample_info['Sample name']}")
            self.ax = self.figure.add_subplot(111)
            self.ax.plot(self.df[x_col_name], self.df[y_col_name])
            self.ax.set_xlabel(x_col_name)
            self.ax.set_ylabel(y_col_name)

            # Set x and y limits
            self.ax.set_xlim(left=0)
            self.ax.set_ylim(bottom=0)

            # Remove padding around the plot
            self.ax.margins(0)

            # Use tight layout to minimize the space around the plot
            self.figure.tight_layout()

            self.canvas.draw()
            self.current_x_axis, self.current_y_axis = x_col_name, y_col_name
            print("[plot updated]")
        else:
            print("[plot not updated - missing columns or empty dataframe]")

    def open_plot_settings(self):
        print(f"cls: CSVViewer, func: open_plot_settings [called]", end=" ")
        if self.df.empty:
            print("[dataframe is empty - settings dialog not opened]")
            return
        try:
            dialog = PlotSettingsDialog(self)
            dialog.pass_column_names(list(self.df.columns))
        except Exception as exc:
            print(f"Plot settings dialog could not be opened. REASON: {exc}")
        if dialog.exec():
            self.current_x_axis, self.current_y_axis = dialog.get_settings()
            self.update_plot(self.current_x_axis, self.current_y_axis)

    def open_plot_control(self):
        print(f"cls: CSVViewer, func: open_plot_control [called]", end=" ")
        dial_result = None
        try:
            dialog = PlotControlDialog(self)
            dial_result = dialog.exec()
            if dial_result:
                try:
                    print(f"[Dialog closed properly.]", end=" ")
                    x_min, x_max = float(dialog.plot_controls['x_min']), float(dialog.plot_controls['x_max'])
                    closest_index_min = (self.df[self.current_x_axis] - x_min).abs().idxmin()
                    closest_index_max = (self.df[self.current_x_axis] - x_max).abs().idxmin()
                    self.df = self.df.loc[closest_index_min:closest_index_max]

                    print(f"[Changes implemented.]")
                    self.update_plot(self.current_x_axis, self.current_y_axis)
                except Exception as exc:
                    print(f"ERROR, REASON: {exc}")
        except Exception as exc:
            print(f"Plot control dialog could not be opened. REASON: {exc}")

    def estimate_yield(self):
        print(f"cls: CSVViewer, func: estimate_yield [called]", end=" ")
        self.info_label.setText("Wybierz 2 punkty na wykresie do wyznaczenia prostej.")

        print(f"hvlines length: {len(self.hvlines)}")
        if len(self.hvlines) > 4:
            last_line = self.hvlines.pop(0)
            last_line.remove()
            last_line = self.hvlines.pop(0)
            last_line.remove()
        elif len(self.hvlines) < 4 :
            self.hline = self.ax.axhline(color='gray', linestyle='--')
            self.hvlines.append(self.hline)
            self.vline = self.ax.axvline(color='gray', linestyle='--')
            self.hvlines.append(self.vline)
            print(f"[Plot lines added]")
            self.motion_conn_id = self.figure.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
            self.press_conn_id = self.figure.canvas.mpl_connect('button_press_event', self.on_click_regression)
        elif len(self.hvlines) == 4:
            self.hline.remove()
            self.vline.remove()
            self.figure.canvas.mpl_disconnect(self.motion_conn_id)
            self.figure.canvas.mpl_disconnect(self.press_conn_id)
            self.add_regression_line()

    def add_regression_line(self):
        print(f"cls: CSVViewer, func:add_regression_line [Plotting reggression line...].")
        x_values = [self.regression_point_1[self.current_x_axis], self.regression_point_2[self.current_x_axis]]
        y_values = [self.regression_point_1[self.current_y_axis], self.regression_point_2[self.current_y_axis]]
        for i in range(len(self.hvlines)):
            try:
                line_to_removal = self.hvlines[i]
                print(line_to_removal, end=' - ')
                line_to_removal.remove()
                print(f"line removed.")
            except Exception as exc:
                print(f"ERROR -> {exc}")

        self.hvlines.clear()
        m = (y_values[1] - y_values[0])/(x_values[1] - x_values[0])
        b = y_values[0] - m * x_values[0]

        # Punkt przecięcia z osią X
        x_intercept = -b/m

        #nowy zakres X do przodu
        x_extrapolated = self.df.tail(1)[self.current_x_axis].values[0]*0.5

        x_values = np.array([x_intercept, x_extrapolated])
        y_values = m * x_values + b

        self.regression_line = self.ax.plot(x_values, y_values, 'r-', label='Regression Line', lw=0.25)
        self.canvas.draw()
        print(f"cls: CSVViewer, func:add_regression_line [Plotting Finished.].")
        self.add_yield_line(x_values, y_values)

    def add_yield_line(self, regression_x, regression_y):
        x_values, y_values = regression_x, regression_y
        x_offset = 0.2
        x1, x2 = x_values[0] + x_offset, x_values[-1] + x_offset
        y1, y2 = y_values[0], y_values[-1]

        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1

        x_min_row = self.get_closest_row(self.current_x_axis, x1).name

        data_y = self.df[self.current_y_axis].to_numpy()
        filtered_data_y = data_y[~np.isnan(data_y)]
        y_max_idx = np.argmax(filtered_data_y)
        y_max = filtered_data_y[y_max_idx]

        x_max_row = self.get_closest_row(self.current_y_axis, 0.8*y_max).name
        x_values = self.df.loc[x_min_row:x_max_row+1, self.current_x_axis]

        x_new_values = x_values.to_numpy()
        y_new_values = m * x_new_values + b
        self.ax.plot(x_new_values, y_new_values, 'g--', label='Yield line', lw=0.4)
        self.canvas.draw()
        intersect_x, intersect_y = self.find_intersection(x_new_values, y_new_values)
        print(f"Found intersect point : {intersect_x} - {intersect_y}")
        self.ax.plot([0, intersect_x],[intersect_y, intersect_y], 'b--', label="Yield Strenght line", lw=0.7)
        self.ax.annotate(f"{int(intersect_y)} MPa", (intersect_x, intersect_y), textcoords='offset points',
                         xytext=(-50,1), ha='center', fontsize=12, color='black')
        self.canvas.draw()

    def find_intersection(self, line_x_vals: np.array, line_y_vals: np.array):
        x_min_row = self.get_closest_row(self.current_x_axis, line_x_vals[0]).name
        x_max_row = self.get_closest_row(self.current_x_axis, line_x_vals[-1]).name

        curve_y_range = self.df.loc[x_min_row:x_max_row, self.current_y_axis]
        y_diff = np.abs(curve_y_range.to_numpy() - line_y_vals)
        minimal_diff_index = np.argmin(y_diff)

        x_intersect = line_x_vals[minimal_diff_index]
        y_intersect = line_y_vals[minimal_diff_index]

        return x_intersect, y_intersect

    def scale_data(self):
        print(f"cls: CSVViewer, func: scale_data [called]", end=" ")
        self.hline = self.ax.axhline(color='gray', linestyle='--')
        self.vline = self.ax.axvline(color='gray', linestyle='--')
        print(f"[Plot lines added]")
        self.motion_conn_id = self.figure.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.press_conn_id = self.figure.canvas.mpl_connect('button_press_event', self.on_click_scaling)

    def on_mouse_move(self, event):
        if not event.inaxes:
            return
        # Aktualizacja pozycji krzyżyka
        self.hline.set_ydata([event.ydata, event.ydata])
        self.vline.set_xdata([event.xdata, event.xdata])
        self.canvas.draw_idle()

    def on_click_regression(self, event):
        if not event.inaxes:
            return
        # Znalezienie najbliższego wiersza w DataFrame
        closest_row = self.get_closest_row(self.current_x_axis, event.xdata)
        print(f"Clicked at X={event.xdata}, Closest row: \n{closest_row.to_dict()}")
        if self.regression_point_1 is None:
            self.regression_point_1 = closest_row.to_dict()
        elif self.regression_point_2 is None:
            self.regression_point_2 = closest_row.to_dict()
        self.estimate_yield()

    def on_click_scaling(self, event):
        if not event.inaxes:
            return
        # Znalezienie najbliższego wiersza w DataFrame
        closest_row = self.get_closest_row(self.current_x_axis, event.xdata)
        print(f"Clicked at X={event.xdata}, Closest row: \n{closest_row.to_dict()}")
        self.zero_point_row = self.get_closest_row(self.current_x_axis, event.xdata)
        self.hline.remove()
        self.vline.remove()
        self.canvas.mpl_disconnect(self.motion_conn_id)
        self.canvas.mpl_disconnect(self.press_conn_id)
        self.hline = None
        self.vline = None

        self.df = self.df.loc[self.zero_point_row.name:]

        first_record = self.df.head(1)
        zero_strain = first_record[self.current_x_axis]

        self.df[self.current_x_axis] = self.df[self.current_x_axis].apply(lambda x: x-zero_strain)

        self.update_plot(self.current_x_axis, self.current_y_axis)

    def get_closest_row(self, column_name, value) -> pd.Series:
        closest_index = (self.df[column_name] - value).abs().idxmin()
        return self.df.loc[closest_index]

    def reset_plot(self):
        self.regression_point_1, self.regression_point_2 = None, None
        self.df = self.initial_df
        self.update_plot(self.current_x_axis, self.current_y_axis)


class DiagramDrawerApp(QMainWindow):
    def __init__(self):
        super(DiagramDrawerApp, self).__init__()
        self.setWindowTitle("File Dialog Example")
        self.resize(1200, 900)

        print(f"Creating class CSViewer...", end=' - ')
        self.csv_viewer = CSVViewer()
        print(f"Success.")
        self.setCentralWidget(self.csv_viewer)

        self.menu = self.menuBar().addMenu("File")
        self.open_action = self.menu.addAction("Open")
        self.open_action.triggered.connect(self.open_file_dialog)

    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "",
                                                   "CSV Files (*.csv);;All Files (*)")
        print(f"Filepath selected: {file_path}")
        if file_path:
            self.csv_viewer.load_data(file_path)
            self.csv_viewer.open_plot_settings()

        return file_path


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = DiagramDrawerApp()
    print(f"starting display...")
    window.show()
    sys.exit(app.exec())
