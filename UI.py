"""Setp and run GUI to plot MSI images."""
from PyQt5 import QtWidgets, uic, QtCore, QtGui
import sys
from dataclasses import dataclass
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pickle
import pandas as pd
import numpy as np
import re
import os

import matplotlib
# Ensure using PyQt5 backend
matplotlib.use('Qt5Agg')

ui_file = r'MSI_plotting.ui'


def normal_distribution(x, mu, sigma):
    """Kernel for normal distribution centered around mu."""
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-.5 * ((x - mu) / sigma) ** 2)


def rect_function(x, center, width):
    """Rectangular function centered around center with height = 1 / width."""
    return ((x >= (center - width / 2)) & (x <= (center + width / 2))).astype(float) / width


def check_file_integrity(
        file: str, is_file: bool = True, suffixes: list[str] = None
) -> bool:
    """Check if a given file exists and optionally is of right type."""
    if os.path.exists(file):
        if is_file != os.path.isfile(file):
            print(f'{file} is not the right type (folder instead of file or vise versa)')
            return False
        elif is_file and (suffixes is not None):
            if (suffix := os.path.splitext(file)[1]) not in suffixes:
                print(f'{file} should type should be one of {suffixes}, not {suffix}')
                return False
            else:
                print(f'{file} is okay')
                return True
        else:
            print(f'{file} is okay')
            return True
    elif file != '':
        print(f'{file} does not exist.')
    return False


def get_data(file_path: str):
    """Read data from txt file and return pixels and spectra."""
    spectra_mzs = []
    spectra_intensities = []
    spectra_SNR = []
    pixel_names = []
    with open(re.findall(r'(?:file:///)?(.+)', file_path)[0]) as f:
        # first line is number of pixels
        for i, line in enumerate(f):
            # skip first line (only contains information about the number of pixels)
            if i != 0:
                # values in line are separated by semicolons
                pixel_entries = line.replace(',', '.').split(';')
                pixel_name, n_mzs = pixel_entries[:2]
                # making use of start:stop:step notation
                mzs = pixel_entries[2::3]
                intensities = pixel_entries[3::3]
                signal_to_noise_ratios = pixel_entries[4::3]

                pixel_names.append(pixel_name)
                spectra_mzs.append(np.array(mzs, dtype=float))
                spectra_intensities.append(np.array(intensities, dtype=float))
                spectra_SNR.append(np.array(signal_to_noise_ratios, dtype=float))
    return np.array(pixel_names), spectra_mzs, spectra_intensities, spectra_SNR


def create_FT_for_comps(
        mz: str | float,
        pixel_names: np.ndarray[str],
        spectra_mzs: list[np.ndarray[float]],
        data: list[np.ndarray[float]],
        norm_spectra: bool = False,
        kernel_mode: str = 'gauss',
        renormalize_window_fct: bool = True,
        width_filter: float = 0.003,
        **kwargs
) -> pd.DataFrame:
    """
    Given an mz and data, create dataframe object.

    Parameters
    ----------
    mz : str | float
        The mass to plot.
    pixel_names : list[str]
        list of pixel names.
    spectra_mzs : list[float]
        list of masses in each spectum.
    data : list[float]
        list of intensities or SNRs for each pixel.
    norm_spectra : bool, optional
        If true, each spectrum will be scaled to its median. The default is False.
    kernel_mode : str, optional
        The type of weights to use around mz. The default is 'gauss'.
    width_filter : float, optional
        Width of the filter function. For The default is 0.003.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    FT : TYPE
        DESCRIPTION.

    """
    mz = float(mz)
    # if kernel_mode is gauss, width defines 3 * sigma
    # otherwise width is the width of the window
    if kernel_mode == 'gauss':
        width_filter /= 3
    N_pixels = len(pixel_names)

    img_x = np.empty(N_pixels, dtype=int)
    img_y = np.empty(N_pixels, dtype=int)

    array = np.zeros(N_pixels, dtype=float)
    # iterate over pixels (=lines in txt file)
    for idx_pixel, (pixel, mzs, data_point) in enumerate(zip(
            pixel_names, spectra_mzs, data
    )):
        # set x and y values
        img_x[idx_pixel] = int(re.findall('X(.*)Y', pixel)[0])  # x coordinate
        img_y[idx_pixel] = int(re.findall('Y(.*)', pixel)[0])  # y coordinate

        if (is_notempty := (len(data_point) > 0)) and norm_spectra:
            m = np.median(data_point[data_point > 0])
            if m > 0:
                data_point /= m
            else:
                data_point = np.zeros_like(data_point),
        elif not is_notempty:
            array[idx_pixel] = 0
            continue

        if kernel_mode == 'max':
            window_fct = rect_function(mzs, mz, width_filter) * width_filter
            array[idx_pixel] = np.max(data_point * window_fct)
        elif kernel_mode in ('gauss', 'rect'):
            # normal distribution around mz
            if kernel_mode == 'gauss':
                window_fct = normal_distribution(mzs, mz, width_filter)
            else:
                window_fct = rect_function(mzs, mz, width_filter)

            if renormalize_window_fct and ((A := np.trapz(window_fct, x=mzs)) > 0):
                window_fct /= A
            array[idx_pixel] = np.trapz(data_point * window_fct, x=mzs)

    FT = pd.DataFrame(data=array, columns=[mz])
    FT['x'] = img_x
    FT['y'] = img_y

    return FT


def handle_labels(
        fig: plt.figure,
        ax: plt.axes,
        img_shape: tuple[int],
        distance_pixels: float | None = None,
        label_unit: str | None = None,
        N_labels: int = 5,
        depth_offset: float = 0,
        **kwargs
):
    if distance_pixels not in (None, ''):
        str_to_flt_unit = {'pixel units': 1 / distance_pixels,
                           'mm': 1e3,
                           'cm': 1e2,
                           'm': 1,
                           f'{chr(181)}m': 1e6
                           }
        unit = str_to_flt_unit[label_unit]

        ax.set_ylabel(f'depth ({label_unit})')
        pixel_to_depth = distance_pixels * unit
    else:
        ax.set_ylabel(r'depth (pixel units)')
        pixel_to_depth = 1
    y_tick_positions = np.arange(
        0,
        img_shape[0] * (N_labels) / (N_labels - 1),
        img_shape[0] / (N_labels - 1))
    y_tick_labels = y_tick_positions * pixel_to_depth + depth_offset

    significant_digits = 2
    y_tick_precision = int(significant_digits - np.log10(np.max(y_tick_labels)))
    y_tick_labels = np.round(y_tick_labels, y_tick_precision)
    if y_tick_precision <= 0:
        y_tick_labels = y_tick_labels.astype(int)

    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(y_tick_labels)
    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False
    )
    return fig, ax


def plt_comp_in_df(
    mz: str | float,
    ft: pd.DataFrame,
    plt_SNR: bool,
    save_path: str | None = None,
    flip: bool = False,
    resolution_saves=300,
    title=None,
    titles=None,
    dpi=None,
    hold=False,
    figsize: tuple[float] | None = None,
    **kwargs
):
    """Plot the specified mz in the dataframe."""
    mz = float(mz)
    img_mz = ft.pivot(
        index='y', columns='x', values=mz
    ).to_numpy().astype(float)

    if flip:
        img_mz = img_mz.T

    # clip values above vmax
    vmax = ft[mz].quantile(.95)

    fig = plt.Figure(figsize=figsize, layout='constrained', dpi=dpi)
    ax = fig.add_subplot(111)
    im = ax.imshow(img_mz,
                   aspect='equal',
                   interpolation='none',
                   vmax=vmax)

    if title is not None:
        pass
    elif titles is not None:
        title = titles
    else:
        title = f'{mz}'

    fig, ax = handle_labels(fig, ax, img_mz.shape, **kwargs)
    ax.set_title(title)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="20%", pad=0.05)
    # SNR ratios
    if plt_SNR:
        ticks = [0, vmax / 3, 2 * vmax / 3, vmax]
        ticklabels = [
            '0',
            f'{np.around(vmax/3,1)}',
            f'{np.around(2*vmax/3,1)}',
            f'>{np.around(vmax,1)}'
        ]
        cbar_label = 'SNR'
    else:
        ticks = [0, vmax]
        ticklabels = ['0', '{:.0e}'.format(vmax)]
        cbar_label = 'Intensity'
    cbar = fig.colorbar(im, cax=cax, ticks=ticks)
    cbar.ax.set_yticklabels(ticklabels)
    cbar.ax.set_ylabel(cbar_label, rotation=270)
    # fig.tight_layout()
    if save_path not in (None, '', ' '):
        image_name = f"{str(mz).replace('.', 'dot')}.png"
        image_file = os.path.join(save_path, image_name)
        print(f'saving image to {save_path} as {image_name}')
        fig.savefig(image_file, dpi=resolution_saves)
    if not hold:
        fig.show()
    else:
        return fig, ax


def plt_compound(
        mz: float | str | list[float | str],
        plt_SNR: bool = False,
        **kwargs):
    if isinstance(mz, list) and (len(mz) > 1):
        return plt_compounds_colored(mz, plt_SNR=plt_SNR, **kwargs)
    else:
        if isinstance(mz, list):
            mz = mz[0]
        ft = create_FT_for_comps(mz=mz, plt_SNR=plt_SNR, **kwargs)
        return plt_comp_in_df(mz=mz, ft=ft, plt_SNR=plt_SNR, **kwargs)


def plt_compounds_colored(
        mz1, mz2=None, mz3=None,
        title=None,
        flip=False,
        save_path=None,
        hold=False,
        figsize=None,
        dpi=None,
        **kwargs
):

    def prepare_slice(ft: pd.DataFrame) -> np.ndarray[np.uint8]:
        c = ft.pivot(index='y', columns='x').to_numpy()
        c_clip = np.nanquantile(c, .95)
        c[c > c_clip] = c_clip
        c[c < 0] = 0
        if c_clip > 0:
            c *= 1 / c_clip
        c[np.isnan(c)] = 0
        return (c * 255).astype(np.uint8)

    def as_masses() -> tuple[str]:
        title1 = str(round(float(mz1), 4))
        title2 = str(round(float(mz2), 4))
        if not is_dual:
            title3 = str(round(float(mz3), 4))
        else:
            title3 = ''
        return title1, title2, title3

    def set_titles(title: str | list[str] | None) -> str:
        # no title passed
        if title in (None, '', ' '):
            title1, title2, title3 = as_masses()
        # title is str (separated by &\n)
        elif isinstance(title, str):
            if ' &\n' in title:
                s = ' &\n'
            elif ';' in title:
                s = ';'
            # title cannot be split
            else:
                title1, title2, title3 = as_masses()
                return title, title1, title2, title3
            titles = title.split(s)
            if len(titles) == 2:
                title1, title2 = titles
                title3 = ''
            elif len(titles) == 3:
                title1, title2, title3 = titles
            else:
                title1, title2, title3 = as_masses()
                return title, title1, title2, title3
        elif len(title) == 2:
            title1, title2 = title
            title3 = ''
        elif len(title) == 3:
            title1, title2, title3 = title
        else:
            title1, title2, title3 = as_masses()
            return title, title1, title2, title3

        if title3 != '':
            title = ' &\n'.join([title1, title2, title3])
        else:
            title = ' &\n'.join([title1, title2])
        return title, title1, title2, title3

    # unpack masses
    # not a string but iterable
    if (not isinstance(mz1, str)) and isinstance(mz1, Iterable):
        # two masses in iterable
        if len(mz1) == 2:
            mz1, mz2 = mz1
        # three masses in iterable
        elif len(mz1) == 3:
            mz1, mz2, mz3 = mz1
        else:
            print('masses passed are in wrong format')
            return

    is_dual = False
    if mz3 in (None, '', ' '):
        is_dual = True
        mz3 = None

    FT1 = create_FT_for_comps(float(mz1), **kwargs)
    FT2 = create_FT_for_comps(float(mz2), **kwargs)
    r = prepare_slice(FT1)
    g = prepare_slice(FT2)
    if not is_dual:
        FT3 = create_FT_for_comps(float(mz3), **kwargs)
        b = prepare_slice(FT3)
    else:
        mz3 = None
        b = np.zeros_like(r)

    fig = plt.Figure(figsize=figsize, layout='constrained', dpi=dpi)
    ax = fig.add_subplot(111)

    image = np.dstack((r, g, b))

    if flip:
        image = image.swapaxes(0, 1)

    ax.imshow(image, interpolation='None')

    title, title1, title2, title3 = set_titles(title)

    if not is_dual:
        handles = [
            Patch(facecolor='red', label=title1),
            Patch(facecolor='green', label=title2),
            Patch(facecolor='blue', label=title3)
        ]
    else:
        handles = [
            Patch(facecolor='red', label=title1),
            Patch(facecolor='green', label=title2)
        ]

    ax.legend(
        handles=handles,
        bbox_to_anchor=(0.5, -0.05),
        # mode='expand',
        ncols=2 + (1 - is_dual),
        loc='upper center'
    )

    fig, ax = handle_labels(fig, ax, image.shape, **kwargs)
    ax.set_title(title)

    # plt.gcf().set_size_inches(4.2, 4.5)
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    if not hold:
        plt.show()
    return fig, ax


@dataclass
class Spectra:
    pixel_names: np.ndarray[str]
    spectra_mzs: list[np.ndarray[float]]
    spectra_intensities: list[np.ndarray[float]]
    spectra_SNR: list[np.ndarray[float]]

    def get_data(self, SNR: bool):
        if SNR:
            return self.spectra_SNR
        return self.spectra_intensities


@dataclass
class Options:
    kernel_mode: str
    width_filter: int
    plt_SNR: bool
    norm_spectra: bool
    N_labels: int
    depth_offset: float
    distance_pixels: float
    flip: bool
    label_unit: str
    autosave: bool
    resolution_saves: int


class MassFile:
    def __init__(self, file_path: str, column_masses: str, column_names: str, column_indizes: str):
        self.file_path = file_path
        self.column_masses = column_masses
        self.column_names = column_names
        self.column_indizes = column_indizes

        self.is_okay = True
        self.issue = None

        self.try_finding_cols()
        if self.is_okay:
            self.cleanup()

        if self.issue is not None:
            print(self.issue)

        self.idx = 0
        self.N_unique_idxs = len(np.unique(self.df[self.column_indizes]))

    def try_finding_cols(self):
        if (suffix := os.path.splitext(self.file_path)[1]) in ('.csv', '.txt'):
            self.df = pd.read_csv(self.file_path)
        elif suffix == '.xlsx':
            self.df = pd.read_excel(self.file_path)
        # strip whitespaces
        self.df = self.df.map(lambda x: x.strip() if isinstance(x, str) else x)
        self.df.columns = self.df.columns.str.strip()
        print('read file as')
        print(self.df.head(5))

        # try to infere missing columns
        if (self.column_masses == '') or (self.column_masses not in self.df.columns):
            try_cols = {'mz', 'm/z', 'mzs', 'mass', 'masses', 'm'}
            for col in self.df.columns:
                if col.lower().strip() in try_cols:
                    self.column_masses = col
                    break
            # in case no match was found
            if self.column_masses not in self.df.columns:
                self.issue = 'could not find a column for masses'
                self.is_okay = False

                return

        if (self.column_names == '') or (self.column_names not in self.df.columns):
            try_cols = {'name', 'names', 'title', 'titles', 'label', 'labels', 'formula'}
            for col in self.df.columns:
                if col.lower().strip() in try_cols:
                    self.column_names = col
                    break
            # in case no match was found
            if self.column_names not in self.df.columns:
                print('could not find a column for names, using masses')
                self.column_names = 'names'
                try:
                    self.df[self.column_names] = np.around(self.get_masses().astype(float), 4).astype(str)
                except:
                    self.issue = 'masses contain invalid values'
                    self.is_okay = False
                    return

        if (self.column_indizes == '') or (self.column_indizes not in self.df.columns):
            try_cols = {'index', 'indizes', 'idx', 'idxs', 'i', 'n', 'num', 'number'}
            for col in self.df.columns:
                if col.lower().strip() in try_cols:
                    self.column_indizes = col
                    break
            # in case no match was found
            if self.column_indizes not in self.df.columns:
                print('could not find a column for indizes, adding one')
                self.column_indizes = 'idx'
                self.df[self.column_indizes] = np.arange(self.df.shape[0])

    def get_masses(self):
        return self.df.loc[:, self.column_masses]

    def get_names(self):
        return self.df.loc[:, self.column_names]

    def get_indizes(self):
        return self.df.loc[:, self.column_indizes]

    def get_entry_at_current_idx(self):
        # unique indizes in dataframe
        idxs = np.unique(self.df[self.column_indizes])
        # get nth unique index
        idx_u = idxs[self.idx]

        # find rows matching nth unique index
        m = self.df[self.column_indizes] == idx_u
        sec = self.df.loc[m, :]
        # get masses and anmes for that index
        masses = sec[self.column_masses].tolist()[:3]
        names = sec[self.column_names].tolist()[:3]
        return masses, names

    def next_idx(self):
        self.idx += 1
        self.idx %= self.N_unique_idxs
        return self.get_entry_at_current_idx()

    def prev_idx(self):
        self.idx -= 1
        self.idx %= self.N_unique_idxs
        return self.get_entry_at_current_idx()

    def cleanup(self):
        # throw out bad indizes
        mask_valid = self.df[self.column_indizes] >= 0
        self.df = self.df.loc[mask_valid, :]
        # check if columns contain valid entries
        try:
            self.df[self.column_masses] = [float(m) for m in self.get_masses()]
        except:
            self.issue = 'masses contain invalid value(s) that can not be interpreted as a number'
            self.is_okay = False
        try:
            self.df[self.column_names] = [str(n) for n in self.get_names()]
        except:
            self.issue = 'names contain invalid value(s) that can not be interpreted as string'
            self.is_okay = False

        # try:
        #     self.df[self.column_indizes] = [int(i) for i in self.get_indizes()]
        #     if np.any(self.df[self.column_indizes] < 0):
        #         self.is_okay = False
        #         self.issue = 'indizes contain value(s) smaller than 0'
        # except:
        #     self.issue = 'indizes contain invalid value(s) that can not be interpreted as int'
        #     self.is_okay = False

        self.df[self.column_indizes] = [int(i) for i in self.get_indizes()]
        if np.any(self.df[self.column_indizes] < 0):
            self.is_okay = False
            self.issue = 'indizes contain value(s) smaller than 0'

        try:
            self.df = self.df.sort_values(by=self.column_indizes).reset_index(drop=True)
        except:
            pass


class EmittingStream(QtCore.QObject):
    textWritten = QtCore.pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))

    def flush(self):
        pass  # Implement the flush method as a no-op


class MplCanvas(FigureCanvas):

    def __init__(self):
        fig = plt.Figure()
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

    def replace_figure_and_axes(self, new_figure, new_axes):
        # Remove the current axes from the current figure
        self.figure.delaxes(self.axes)

        # Assign the new axes and figure
        self.figure = new_figure
        self.axes = new_axes

        # Add the new axes to the new figure
        self.figure.add_axes(self.axes)


class UI(QtWidgets.QMainWindow):
    def __init__(self):
        super(UI, self).__init__()
        uic.loadUi(ui_file, self)
        self.initiate_plt_area()
        self.show()
        # console output in textView
        sys.stdout = EmittingStream(textWritten=self.normalOutputWritten)
        # load default options
        self.update_options()

        # link buttons to functions
        self.link_widgets()

        self.has_valid_mass_file = False

    def closeEvent(self, event):
        # Restore sys.stdout when the GUI is closed
        sys.stdout = sys.__stdout__
        event.accept()

    def normalOutputWritten(self, text):
        """Append text to the QTextEdit."""
        # Maybe QTextEdit.append() works as well, but this is how I do it:
        cursor = self.textEdit_console.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.textEdit_console.setTextCursor(cursor)
        self.textEdit_console.ensureCursorVisible()

    def link_widgets(self):
        self.btn_read_data.clicked.connect(self.read_data)
        self.btn_read_masses.clicked.connect(self.read_mass_file)
        self.btn_browse_data.clicked.connect(self.get_txt_file_from_dialog)
        self.btn_browse_masses.clicked.connect(self.get_mass_file_from_dialog)
        self.btn_browse_img_folder.clicked.connect(self.get_img_dir_from_dialog)
        self.btn_plot.clicked.connect(self.plt_comp)
        self.btn_save_plot.clicked.connect(self.save_plot)
        self.btn_load.clicked.connect(self.load_settings)
        self.btn_save.clicked.connect(self.save_settings)

        self.btn_next.clicked.connect(self.plt_next)
        self.btn_prev.clicked.connect(self.plt_prev)
        self.stop = False
        self.btn_all_mass.clicked.connect(self.plt_all)
        self.btn_stop.clicked.connect(self.set_stop)

    def get_txt_file_from_dialog(self):
        txt_file = QtWidgets.QFileDialog.getOpenFileName(self, 'File to read', 'c:\\', '*.txt')[0]
        le = self.findChild(QtWidgets.QLineEdit, 'lineEdit_file_spectra')
        le.setText(txt_file)

    def get_img_dir_from_dialog(self):
        img_dir = QtWidgets.QFileDialog.getExistingDirectory(self, 'Directory for images', 'c:\\')
        le = self.findChild(QtWidgets.QLineEdit, 'lineEdit_dir_imgs')
        le.setText(img_dir)

    def get_mass_file_from_dialog(self):
        file = QtWidgets.QFileDialog.getOpenFileName(self, 'File for masses', 'c:\\', '(*.txt *.csv *.xlsx)')[0]
        le = self.findChild(QtWidgets.QLineEdit, 'lineEdit_mass_list')
        le.setText(file)

    def read_data(self):
        """Read data in specified path and store in Spectra class."""
        # get the entry in lineEdit
        le = self.findChild(QtWidgets.QLineEdit, 'lineEdit_file_spectra')
        txt_file = le.text()
        if txt_file in (None, '', ' '):
            return

        print('loading data...')
        pixel_names, spectra_mzs, spectra_intensities, spectra_SNR = get_data(txt_file)
        self.spectra = Spectra(
            pixel_names=pixel_names,
            spectra_mzs=spectra_mzs,
            spectra_intensities=spectra_intensities,
            spectra_SNR=spectra_SNR
        )
        print('finished loading')

        mzs_all = [float(mz) for mzs in self.spectra.spectra_mzs for mz in mzs]
        SNR_all = [float(snr) for SNRs in self.spectra.spectra_SNR for snr in SNRs]
        N_pixels = str(len(self.spectra.pixel_names))

        mz_min = str(round(min(mzs_all), 4))
        mz_max = str(round(max(mzs_all), 4))
        snr = str(round(min(SNR_all), 1))
        self.tableWidget.setItem(0, 0, QtWidgets.QTableWidgetItem(mz_min))
        self.tableWidget.setItem(1, 0, QtWidgets.QTableWidgetItem(mz_max))
        self.tableWidget.setItem(2, 0, QtWidgets.QTableWidgetItem(N_pixels))
        self.tableWidget.setItem(3, 0, QtWidgets.QTableWidgetItem(snr))

    def read_data_default(self):
        """Read data in specified path and store in Spectra class."""
        txt_file = r'D:/My Drive/Master Thesis/Data/490-495/GDGT/490-495_GDGT_mw1310-1330_dots.txt'

        print('loading data...')
        pixel_names, spectra_mzs, spectra_intensities, spectra_SNR = get_data(txt_file)
        self.spectra = Spectra(
            pixel_names=pixel_names,
            spectra_mzs=spectra_mzs,
            spectra_intensities=spectra_intensities,
            spectra_SNR=spectra_SNR
        )
        print('finished loading')

        mzs_all = [float(mz) for mzs in self.spectra.spectra_mzs for mz in mzs]
        SNR_all = [float(snr) for SNRs in self.spectra.spectra_SNR for snr in SNRs]

        mz_min = str(round(min(mzs_all)))
        mz_max = str(round(max(mzs_all)))
        snr = str(round(min(SNR_all)))
        self.tableWidget.setItem(0, 0, QtWidgets.QTableWidgetItem(mz_min))
        self.tableWidget.setItem(1, 0, QtWidgets.QTableWidgetItem(mz_max))
        self.tableWidget.setItem(3, 0, QtWidgets.QTableWidgetItem(snr))

    def update_le_mass_title(self, mass: list[float | str], name: list[str]):
        lem = self.findChild(QtWidgets.QLineEdit, 'lineEdit_mz_val')
        let = self.findChild(QtWidgets.QLineEdit, 'lineEdit_title')
        if (not isinstance(mass, str)) and isinstance(mass, Iterable):
            mass = ';'.join([str(m) for m in mass])
            name = ' &\n'.join(name)
        lem.setText(mass)
        let.setText(name)

    def read_mass_file(self):
        masses_file = self.findChild(QtWidgets.QLineEdit, 'lineEdit_mass_list').text()
        # omit file:/// if it is in the path
        if masses_file in (None, '', ' '):
            return
        masses_file = re.findall(r'(?:file:///)?(.+)', masses_file)[0]
        if not check_file_integrity(masses_file, suffixes=['.txt', '.csv', '.xlsx']):
            print('invalid mass list file')
            return
        # get columns for masses, names and index
        le_masses = self.findChild(QtWidgets.QLineEdit, 'lineEdit_col_mz')
        le_names = self.findChild(QtWidgets.QLineEdit, 'lineEdit_col_names')
        le_indizes = self.findChild(QtWidgets.QLineEdit, 'lineEdit_col_idx')

        column_masses = le_masses.text()
        column_names = le_names.text()
        column_indizes = le_indizes.text()

        mass_file = MassFile(masses_file, column_masses, column_names, column_indizes)
        if not mass_file.is_okay:
            return
        else:
            self.has_valid_mass_file = True

        self.mass_file: MassFile = mass_file

        # set les
        le_masses.setText(self.mass_file.column_masses)
        le_names.setText(self.mass_file.column_names)
        le_indizes.setText(self.mass_file.column_indizes)

        # set mass as first entry
        mass, name = self.mass_file.get_entry_at_current_idx()
        self.update_le_mass_title(mass, name)

        mzs_all = self.mass_file.get_masses()
        N_col = self.mass_file.df.shape[0]

        mz_min = str(np.round(np.min(mzs_all), 4))
        mz_max = str(np.round(np.max(mzs_all), 4))
        N_col = str(N_col)
        self.tableWidget.setItem(0, 1, QtWidgets.QTableWidgetItem(mz_min))
        self.tableWidget.setItem(1, 1, QtWidgets.QTableWidgetItem(mz_max))
        self.tableWidget.setItem(2, 1, QtWidgets.QTableWidgetItem(N_col))

    def plt_next(self):
        if not self.has_valid_mass_file:
            return
        mass, name = self.mass_file.next_idx()
        self.update_le_mass_title(mass, name)
        self.plt_comp()

    def plt_prev(self):
        if not self.has_valid_mass_file:
            return
        mass, name = self.mass_file.prev_idx()
        self.update_le_mass_title(mass, name)
        self.plt_comp()

    def set_stop(self):
        if not self.has_valid_mass_file:
            return
        print('stopping plotting')
        self.stop = True

    def plt_all(self):
        if not self.has_valid_mass_file:
            return
        print('plotting all compounds in mass file')
        self.mass_file.idx = -1
        for i in range(self.mass_file.N_unique_idxs):
            if self.stop:
                break
            QtWidgets.QApplication.processEvents()
            self.plt_next()
        self.stop = False

    def update_options(self):
        try:
            resolution_saves = self.findChild(QtWidgets.QLineEdit, 'lineEdit_resolution').text()
            resolution_saves = int(resolution_saves)
        except:
            print('resolution must be whole number')
            return
        try:
            N_labels = self.findChild(QtWidgets.QLineEdit, 'lineEdit_n_labels').text()
            N_labels = int(N_labels)
        except:
            print('Number labels must be whole number')
            return
        try:
            depth_offset = self.findChild(QtWidgets.QLineEdit, 'lineEdit_do').text()
            depth_offset = float(depth_offset)
        except:
            print('depth offset must be number')
            return
        try:
            distance_pixels = self.findChild(QtWidgets.QLineEdit, 'lineEdit_dp').text()
            if distance_pixels != '':
                distance_pixels = float(distance_pixels) * 1e-6
        except:
            print('distance pixels must be number or empty, will be ignored otherwise')
        try:
            width_filter = self.findChild(QtWidgets.QLineEdit, 'lineEdit_wf').text()
            width_filter = float(width_filter) / 1000
        except:
            print('filter width must be number')
            return
        try:
            resolution_saves = int(self.lineEdit_resolution.text())
        except:
            resolution_saves = 600
            print('resolution must be whole number bigger than 0, using default value of 600')
        try:
            SMALL_SIZE = self.horizontalScrollBar.value() / 10
            MEDIUM_SIZE = SMALL_SIZE * 3 / 2
            BIGGER_SIZE = SMALL_SIZE * 5 / 3

            plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
            plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
            plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
            plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
            plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
            plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
            plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
            print('updated small fs to', SMALL_SIZE)
        except:
            pass

        flip = self.findChild(QtWidgets.QCheckBox, 'checkBox_flip_image').isChecked()
        autosave = self.findChild(QtWidgets.QCheckBox, 'checkBox_autosave').isChecked()
        norm_spectra = self.findChild(QtWidgets.QCheckBox, 'checkBox_norm_spectra').isChecked()

        snr = self.findChild(QtWidgets.QComboBox, 'comboBox_plt_SNR').currentText()
        if snr == 'signal to noise ratio':
            snr = True
        elif snr == 'intensities':
            snr = False

        kernel_mode = self.findChild(QtWidgets.QComboBox, 'comboBox_km').currentText()

        unit_labels = self.findChild(QtWidgets.QComboBox, 'comboBox_unit').currentText()

        self.opts = Options(
            kernel_mode=kernel_mode,
            width_filter=width_filter,
            plt_SNR=snr,
            norm_spectra=norm_spectra,
            N_labels=N_labels,
            depth_offset=depth_offset,
            distance_pixels=distance_pixels,
            flip=flip,
            label_unit=unit_labels,
            autosave=autosave,
            resolution_saves=resolution_saves
        )

    def initiate_plt_area(self):
        placeholder = self.findChild(QtWidgets.QWidget, 'plt_area')

        # Get the existing layout of the widget or create a new one if it doesn't have a layout
        layout = QtWidgets.QVBoxLayout()
        placeholder.setLayout(layout)

        self.canvas = MplCanvas()

        # Add the FigureCanvas to the layout
        layout.addWidget(self.canvas)

        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        # Set stretch factor to 1 to make it expand to fill the available space
        layout.setStretchFactor(self.canvas, 1)

        self.canvas.show()

    def updata_plt_area(self):
        self.canvas.replace_figure_and_axes(self.fig, self.ax)
        self.canvas.draw()

    @QtCore.pyqtSlot()
    def plt_comp(self):
        self.update_options()
        # get mz val
        mzs: list[str] = self.findChild(QtWidgets.QLineEdit, 'lineEdit_mz_val').text().split(';')
        snr: bool = self.opts.plt_SNR
        print('plotting', mzs)

        # convert strs to float
        try:
            mzs: list[float] = [float(mz) for mz in mzs]
        except:
            print(f'provided mzs ({mzs}) cannot be interpreted as number')
            return

        # read title from lineEdit
        t = self.findChild(QtWidgets.QLineEdit, 'lineEdit_title').text()
        if (t != '') and ((len(t.split(';')) == len(mzs)) or (len(t.split('&')) == len(mzs))):
            title = t
        else:
            title = None

        self.fig, self.ax = plt_compound(
            mzs,
            pixel_names=self.spectra.pixel_names,
            spectra_mzs=self.spectra.spectra_mzs,
            data=self.spectra.get_data(snr),
            figsize=self.canvas.figure.get_size_inches(),
            dpi=self.canvas.figure.get_dpi(),
            title=title,
            hold=True,
            **self.opts.__dict__
        )

        self.updata_plt_area()
        self.lineEdit_resolution.setText(str(self.canvas.figure.get_dpi()))

        # update name
        if title is None:
            file_name = 'and'.join([str(mz) for mz in mzs]).replace('.', 'd')
        else:
            file_name = title.replace(' &\n', 'and').replace('.', 'd').replace(':', 'dd')

        self.findChild(QtWidgets.QLineEdit, 'lineEdit_image_name')\
            .setText(file_name)

        # check autosave
        if self.opts.autosave:
            self.save_plot()

    def save_plot(self):
        # get folder
        folder_imgs = self.findChild(QtWidgets.QLineEdit, 'lineEdit_dir_imgs').text()
        # get name
        name = self.findChild(QtWidgets.QLineEdit, 'lineEdit_image_name').text() + '.png'
        try:
            dpi = float(self.lineEdit_resolution.text())
        except:
            dpi = 600
        try:
            self.fig.savefig(os.path.join(folder_imgs, name), dpi=dpi)
            print(f'saved plot in {folder_imgs} as {name}')
        except AttributeError:
            print('Create a figure first!')

    def save_settings(self):
        fields = [
            'lineEdit_file_spectra', 'lineEdit_mass_list', 'lineEdit_col_mz',
            'lineEdit_col_names', 'lineEdit_col_idx', 'lineEdit_mz_val',
            'lineEdit_title', 'checkBox_flip_image', 'lineEdit_wf', 'lineEdit_resolution'
            'comboBox_plt_SNR', 'checkBox_norm_spectra', 'comboBox_km',
            'lineEdit_n_labels', 'comboBox_unit', 'lineEdit_do', 'lineEdit_dp',
            'lineEdit_dir_imgs', 'checkBox_autosave',
            'lineEdit_image_name', 'horizontalScrollBar'
        ]

        entries = []
        for field in fields:
            widget_type = field.split('_')[0]
            if widget_type == 'lineEdit':
                e = self.findChild(QtWidgets.QLineEdit, field).text()
            elif widget_type == 'comboBox':
                e = self.findChild(QtWidgets.QComboBox, field).currentText()
            elif widget_type == 'checkBox':
                e = self.findChild(QtWidgets.QCheckBox, field).isChecked()
            elif widget_type == 'horizontalScrollBar':
                e = self.findChild(QtWidgets.QScrollBar, field).value()
            else:
                raise NotImplementedError
            entries.append(e)
        d = dict(zip(fields, entries))
        with open('gui_settings.pickle', 'wb') as handle:
            pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('saved settings')

    def load_settings(self):
        print('loading settings')
        with open('gui_settings.pickle', 'rb') as handle:
            d = pickle.load(handle)

        for field, entry in d.items():
            try:
                widget_type = field.split('_')[0]
                if widget_type == 'lineEdit':
                    e = self.findChild(QtWidgets.QLineEdit, field)
                    e.setText(entry)
                elif widget_type == 'comboBox':
                    e = self.findChild(QtWidgets.QComboBox, field)
                    e.setCurrentText(entry)
                elif widget_type == 'checkBox':
                    e = self.findChild(QtWidgets.QCheckBox, field)
                    e.setChecked(entry)
            except:
                pass

        self.read_data()
        self.read_mass_file()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = UI()
    sys.exit(app.exec_())
