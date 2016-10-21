"""
Interactive plotting tools.
"""

__all__ = ['Figure', 'Plot']

import numpy as np
import matplotlib.pyplot as plt

from IPython.display import display, clear_output, HTML, Javascript
from IPython.core.ultratb import VerboseTB
import IPython.utils.traitlets as traitlets
from collections import OrderedDict

import types
import inspect
import os
import sys
import imp
from cStringIO import StringIO
from base64 import b64encode

from .helpers import fix_args
from . import widgets

def set_limits(self, min, max):
    ''' set min and max simultaneously '''
    if min < self.max:
        self.min = float(min)
        self.max = float(max)
    else:
        self.max = float(max)
        self.min = float(min)
widgets.BoundedFloatText.set_limits = set_limits


class Figure(widgets.DOMWidget):
    """
    An IPython widget showing a matplotlib `Figure`, with zooming and cursors.
    
    In the notebook, `Axes` within the `Figure` are zoomed by drawing a zoom
    rectancle with the mouse. The zoom is reset by double-clicking the `Axes`.
    Zooming can be limited to the x or y axis by holding the control or shift
    keys while drawing the rectangle.
    
    Rulers are created by clicking and dragging the top or left borders into
    the `Axes`. Cursors are created by dragging the little square at the
    intersection of the top and left borders into the `Axes`. Rulers and
    cursors are removed by dragging them out of the `Axes`.
    
    A context menu provides additional options.
    
    Notes
    -----
    `Figure` currently works best with Firefox and has a few issues with
    webkit-based browsers such as Chrome and Safari.
    
    Parameters
    ----------
    fig : `matplotlib.Figure`
        The displayed figure.
        
    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> plt.close(fig)
    >>> xs = np.linspace(0, 5*np.pi)
    >>> ax.plot(xs, np.sin(xs))
    >>> uqtools.Figure(fig)
    """
    
    #_view_name = traitlets.Unicode('FigureView', sync=True)
    #_view_name = traitlets.Unicode('ZoomFigureView', sync=True)
    _view_name = traitlets.Unicode('ZoomCursorFigureView', sync=True)
    _format = traitlets.Unicode('png', sync=True)
    _b64image = traitlets.Unicode(sync=True)
    #width = Integer(sync=True)
    #height = Integer(sync=True)
    limits = traitlets.Dict(sync=True)
    cursors = traitlets.List(traitlets.List(traitlets.List), sync=True)
    axes = traitlets.List(traitlets.Dict, sync=True)
    
    _style = HTML(
        '''<style type="text/css">
        .Figure {
            border: solid 1px #e0e0e0;
        }
        
        .Axes {}
        
        .Cursors {
            position: relative;
            width: 100%;
            height: 100%;
        }
        .Cursor {
            position: absolute;
            top: 0px;
            left: 0px;
            width: 100%;
            height: 100%;
            //border: solid 1px;
        }
        .Cursor .hRuler, .Cursor .vRuler {
            position: absolute;
            background-color: black;
            background-clip: content-box;
            border-width: 1px;
            border-color: transparent;
        }
        .Cursor .hRuler {
            top: -1px;
            left: 0px;
            width: 100%;
            height: 3px;
            border-style: solid none;
            cursor: row-resize;
        }
        .Cursor .vRuler {
            top: 0px;
            left: -1px;
            width: 3px;
            height: 100%;
            border-style: none solid;
            cursor: col-resize;
        }
        .Cursor .iRuler {
            position: absolute;
            top: -1px;
            left: -1px;
            width: 3px;
            height: 3px;
            background-color: white;
            border: solid 1px black;
            cursor: crosshair;
        }
        .Cursor .hLabel, .Cursor .vLabel {
            position: absolute;
            color: blue;
            background-color: rgba(255,255,255,0.85);
        }
        </style>''')
    
    @property
    def fig(self):
        return self._fig
    
    @fig.setter
    def fig(self, fig):
        self._fig = fig
        self._zoom_history = []
        self.update()
        
    def update(self):
        # rasterize image
        png_data = StringIO()
        self.fig.canvas.print_png(png_data)
        self._b64image = b64encode(png_data.getvalue())
        _, height = self.fig.canvas.get_width_height()
        # save axes properties in model
        axes = []
        for idx, ax in enumerate(self.fig.get_axes()):
            u_min, u_max = ax.get_xlim()
            v_min, v_max = ax.get_ylim()
            transform = ax.transData.transform
            bbox = transform(np.array([(u_min, v_min), (u_max, v_min), 
                                       (u_max, v_max), (u_min, v_max)]))
            polar = bool(np.all(np.isclose(bbox[2], bbox[3])))
            x_min, y_min = bbox[0]
            x_max, y_max = bbox[2]
            ax_dict = dict(index=idx,
                           u_min=u_min, v_min=v_min, 
                           u_max=u_max, v_max=v_max,
                           x_min=x_min, y_min=height-y_min,
                           x_max=x_max, y_max=height-y_max,
                           polar=polar,
                           zoomable=ax.get_navigate() and ax.can_zoom(),
                           navigable=ax.get_navigate())
            axes.append(ax_dict)
        self.axes = axes

        # update zoom history
        zoom_state = self._zoom_state()
        if zoom_state not in self._zoom_history:
            self._zoom_history.append(zoom_state)
            self._zoom_index = -1
        else:
            self._zoom_index = (self._zoom_history.index(zoom_state) - 
                                len(self._zoom_history))
        
    def __init__(self, fig=None, **kwargs):
        self._zoom_history = []
        self._zoom_index = -1
        if fig is not None:
            kwargs['fig'] = fig
        super(Figure, self).__init__(**kwargs)
        self._zoom_handlers = widgets.CallbackDispatcher()
        self.on_zoom(self.zoom)
        self.on_msg(self._handle_messages)
        
    def compile(self):
        """Push style sheets and JavaScript to the browser."""
        # load and display js
        plotpy_fn = inspect.getfile(inspect.currentframe())
        plotpy_path = os.path.dirname(os.path.abspath(plotpy_fn))
        js_fn = os.path.join(plotpy_path, 'FigureWidget.js')
        js = Javascript(file(js_fn).read())
        display(js)
        # display css
        display(self._style)
        
    def _ipython_display_(self):
        # push CSS and JS to the browser
        self.compile()
        # display self
        super(Figure, self)._ipython_display_()
        
    def _handle_messages(self, _, content):
        if content.get('event', None) == 'zoom':
            # zoom to rectangle drawn by user
            axis = content['axis']
            xlim = (content['u_min'], content['u_max'])
            ylim = (content['v_min'], content['v_max'])
            # execute zoom handlers
            self._zoom_handlers(axis, xlim, ylim)
        if content.get('event', None) == 'zoom_reset':
            # reset zoom of an axis
            axis = content['axis']
            limits = self._zoom_history[0][axis]
            self._zoom_handlers(axis, *limits)
        if content.get('event', None) == 'print':
            display(self.fig)
        if content.get('event', None) == 'clear':
            clear_output()
        if content.get('event', None) == 'close':
            self.close()
        
    def on_zoom(self, callback, remove=False):
        """Register a callback executed when `Axes` are zoomed.

        The callback will be called with three arguments,
        the axis index, new xlim and new ylim.

        Parameters
        ----------
        remove : `bool` (optional)
            Set to True to remove the callback from the list of callbacks.
        """
        self._zoom_handlers.register_callback(callback, remove=remove)
        
    def zoom(self, axis, xlim=None, ylim=None, update=True, **kwargs):
        """ 
        Set `Axes` limits.
        
        Parameters
        ----------
        axis : `int`
            `Axes` index
        xlim : `tuple of float`, optional
            Horizontal axis limits.
        ylim : `tuple of float`, optional
            Vertical axis limits.
        update : `bool`, default True
            If False, do not update the figure display.
        """
        # set limits on figure
        ax = self.fig.get_axes()[axis]
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        # redraw
        if update:
            self.update()
        
    def _zoom_state(self):
        ''' return a list of tuples of the x and y limits of all axes '''
        return [((ax['u_min'], ax['u_max']), (ax['v_min'], ax['v_max'])) 
                for ax in self.axes]

    def _zoom_to(self, idx):
        """Set zoom to _zoom_history item at idx."""
        for axis, limits in enumerate(self._zoom_history[idx]):
            self.zoom(axis, *limits, update=False)
        self.update()
        
    def zoom_reset(self):
        """Reset zoom to its initial value when the fig was set."""
        self._zoom_to(0)
        
    def zoom_prev(self):
        """Zoom to previous item in the zoom history."""
        if self._zoom_index > -len(self._zoom_history):
            self._zoom_to(self._zoom_index-1)
        
    def zoom_next(self):
        """Zoom to next item in the zoom history."""
        if self._zoom_index < -1:
            self._zoom_to(self._zoom_index+1)



class AxisWidget(widgets.Box):
    """
    An axis selection and limit setting widget.
    
    Combines a drop-down, min/max float inputs and an autoscale flag.

    Parameters
    ----------
    description : `str`
        Descriptive text.
    options : `{str: int} dict`
        Axis label to index map.
    
    Attributes
    ----------
    axis : `Integer`
        Selected axis
    min, max : `Float
        Axis limits.
    scaling : `Integer`
        Autoscale mode, one of SCALING_MANUAL, SCALING_AUTO, SCALING_FULL.
    """
    axis = traitlets.Any()#Integer(allow_none=True)
    min = traitlets.Float()
    max = traitlets.Float()
    scaling = traitlets.Integer(allow_none=True)
    disabled = traitlets.Bool()

    SCALING_MANUAL = 0
    SCALING_AUTO = 1
    SCALING_FULL = 2
    
    def __init__(self, description, options, **kwargs):
        axis = kwargs.pop('axis', options.values()[0])
        scaling = kwargs.pop('scaling', self.SCALING_AUTO)
        options = OrderedDict(options)
        super(AxisWidget, self).__init__(axis=axis, scaling=scaling, **kwargs)
        self._w_select = widgets.Dropdown(description=description, 
                                          options=options)
        traitlets.link((self, 'axis'), (self._w_select, 'value'))
        self._w_min = widgets.FloatText(description='min')
        traitlets.link((self._w_min, 'value'), (self, 'min'))
        self._w_max = widgets.FloatText(description='max')
        traitlets.link((self._w_max, 'value'), (self, 'max'))
        self._w_auto = widgets.ToggleButtons(description='scaling')
        self._w_auto.options = OrderedDict((('manual', self.SCALING_MANUAL),
                                            ('auto', self.SCALING_AUTO),
                                            ('full', self.SCALING_FULL)))
        traitlets.link((self, 'scaling'), (self._w_auto, 'value'))
        traitlets.link((self, 'disabled'), (self._w_auto, 'disabled'),
                       (self._w_min, 'disabled'), (self._w_max, 'disabled'))
        self.children = (self._w_select, self._w_min, self._w_max, self._w_auto)
        self.on_trait_change(self._on_limit_change, ('min', 'max'))
    
    set_limits = set_limits
    
    def _on_limit_change(self):
        try:
            # IPython 2.3.1 has no official mechanism to determine if
            # a trait change was initiated by the frontend or backend.
            # This relies on functionality that will change in later
            # versions.
            if ((self._w_min._property_lock[0] == 'value') or
                (self._w_max._property_lock[0] == 'value')):
                self.scaling = self.SCALING_MANUAL
        except:
            pass

    @staticmethod
    def reverse_enumerate(iterable, start=0):
        ''' enumerate yielding (value, index) pairs '''
        idx=start
        for item in iterable:
            yield item, idx
            idx += 1
             

class FunctionWidget(widgets.FlexBox):
    """
    A function selection widget, a select box combined with a code editor.
    
    Functions can be selected in the select box or defined in the code editor
    and activated by clicking compile.    
    
    Parameters
    ----------
    module : Python `module`, optional
        The function select box is populated with all functions in `module`.
        Defaults to `__main__`. Pass None to suppress the default.
    
    Attributes
    ----------
    function : `callable`
        Curently selected function.
        
    Notes
    -----
    Functions `default`, `real`, `imag`, `abs`, `arg`, `arg_xref`, `arg_zref`,
    `dB_from_P`, `dB_from_V` are always available.
    
    """
    
    function = traitlets.Any()
    
    def __init__(self, **kwargs):
        super(FunctionWidget, self).__init__(orientation='horizontal')
        self.functions = imp.new_module('functions')
        self.sources = imp.new_module('sources')
        # add a few default functions
        self.compile('def abs(zs):\n  return np.abs(zs)')
        self.compile('def arg(zs):\n  return np.angle(zs)')
        self.compile('def arg_xref(zs):\n  return np.angle(zs/zs[[0], :])')
        self.compile('def arg_zref(zs):\n  return np.angle(zs/zs[:, [0]])')
        self.compile('def default(zs):\n  if np.iscomplexobj(zs):\n' + 
                     '    return np.abs(zs)\n  else:\n    return zs')
        self.compile('def dB_from_P(zs):\n  return 10*np.log10(np.abs(zs))')
        self.compile('def dB_from_V(zs):\n  return 20*np.log10(np.abs(zs))')
        self.compile('def imag(zs):\n  return np.imag(zs)')
        self.compile('def real(zs):\n  return np.real(zs)')
        self.default = self.functions.default
        # load functions from module
        if 'module' in kwargs:
            module = kwargs.get('module')
            if module is not None:
                self.load_module(module)
        else:
            import __main__
            self.load_module(__main__)
        # create ui
        self._w_select = widgets.Select()
        self._w_select.width = '150px'
        self._w_select.height = '150px'
        self._w_source = widgets.Textarea()
        self._w_source.height = '140px'
        self._w_source.font_family = 'monospace'
        self._w_compile = widgets.Button(description='compile')
        self._w_compile.margin_top = '5px'
        self._w_compile.on_click(self._on_compile)
        self._w_output = widgets.HTML()
        self._w_output.margin_top = '5px'
        self._w_output.width = '300px'
        self._w_output.font_family = 'monospace'
        self._w_compile_box = widgets.Box()
        self._w_compile_box.children = [self._w_compile, self._w_output]
        self.children = (self._w_select, self._w_source, self._w_compile_box)
        traitlets.link((self, 'function'), (self._w_select, 'value'))
        # install event handlers
        self._w_select.on_trait_change(self._on_select, 'value')
        # populate select box
        self.update()
            
    def _on_select(self, _, function):
        """Show source code when a function is selected."""
        if function is not None:
            self._w_source.value = getattr(self.sources, function.__name__)
            self._w_output.visible = False
        
    def _on_compile(self, _):
        """Compile source code and update function."""
        source = self._w_source.value
        try:
            function = self.compile(source)
            self._w_output.visible = False
        except Exception as err:
            self._w_output.value = self.format_exception(err)
            self._w_output.visible = True
            return 
        # update select widget
        self.update(function)
        
    def format_exception(self, err):
        """Pretty-print common compiler exceptions."""
        parts = []
        parts.append('<span class="ansired">{0}</span>'.
                     format(err.__class__.__name__))
        if err.message or len(err.args):
            parts.append(':')
        if err.message:
            parts.append(err.message)
        if err.__class__ == SyntaxError:
            parts.append(err.args[0])
            parts.append('(line {1} column {2})'.format(*err.args[1]))
        else:
            if len(err.args) > 1:
                parts.append(str(err.args[1:]))
        return ' '.join(parts)
    
    def update(self, function=None):
        """Update select box, set current selection to function."""
        functions = OrderedDict([(name, func) for name, func 
                                 in sorted(self.functions.__dict__.iteritems()) 
                                 if not name.startswith('__')])
        self._w_select.options = functions
        if function is not None:
            self._w_select.value = function
        elif self.default is not None:
            self._w_select.value = self.default
    
    def compile(self, source):
        """
        Compile source code
        
        The source code is stored in `self.sources`, the compiled function is 
        stored in `self.functions`.
        
        Parameters
        ----------
        source : `str`
            Source code.
            
        Returns
        -------
        compiled function
        """
        code = compile(source, '<string>', 'single')
        if len(code.co_names) != 1:
            raise ValueError('source code must define exactly one name')
        name, = code.co_names
        exec code in globals(), self.functions.__dict__
        setattr(self.sources, name, source)
        return getattr(self.functions, name)
    
    def load_module(self, module):
        """
        Add all functions in module to the list of functions.
        
        Parameters
        ----------
        module : any object that has a `__dict__`
            Module to inspect.
        """
        for key in module.__dict__.keys():
            if key.startswith('__'):
                continue
            func = getattr(module, key)
            if not callable(func) or not isinstance(func, types.FunctionType):
                continue
            # store functions and code in internal modules
            setattr(self.functions, key, func)
            try:
                source = inspect.getsource(func)
                setattr(self.sources, key, source)
            except:
                setattr(self.sources, key, '# source code unavailable')
        if 'default' in module.__dict__:
            self.default = getattr(module, 'default')
            
            
class FloatTextSliderWidget(widgets.FlexBox):
    """
    A slider with an associated float input box.
    The components update each other.

    Parameters
    ----------
    description : `str`
        Descriptive text.
    values : `tuple of float`
        Valid slider values.
    
    Attributes
    ----------
    index : `Integer`
        Slider position.
    values : `tuple of Float`
        When `index` is changed, `value` is set to `values[index]`.
    value : `Float`
        Text box value (not guaranteed to be an element of values)
    """
    index = traitlets.Integer()
    value = traitlets.Float()
    values = traitlets.Tuple()
    disabled = traitlets.Bool()
    
    def __init__(self, description, values=[0], **kwargs):
        super(FloatTextSliderWidget, self).__init__(orientation='horizontal',
                                                    **kwargs)
        # create widgets    
        self._w_slider = widgets.IntSlider(description=description,
                                           min=0, value=0, readout=False)
        traitlets.link((self._w_slider, 'value'), (self, 'index'))
        self._w_text = widgets.BoundedFloatText(value=values[0])
        traitlets.link((self._w_text, 'value'), (self, 'value'))
        self.children = (self._w_slider, self._w_text)
        # register event handlers
        self._w_slider.on_trait_change(self._on_index_change, 'value')
        self._w_text.on_trait_change(self._on_value_change, 'value')
        self.on_trait_change(self._on_values_change, 'values')
        traitlets.link((self._w_text, 'disabled'), (self._w_slider, 'disabled'),
                       (self, 'disabled'))
        # store values and trigger range update
        self.values = values
        self.index = 0
    
    def _on_values_change(self, _, values):
        """Update ranges on values change."""
        self._w_slider.max = len(values)-1
        self._w_text.set_limits(min(values), max(values))
        self._on_index_change()
        
    def _on_index_change(self):
        """Set value on index change."""
        value = self.values[self.index]
        self._w_text.value = value
            
    def _on_value_change(self):
        """Set index on value change."""
        values = np.array(self.values)
        self.index = np.argmin(np.abs(values-self.value))


class Plot(object):
    """
    An interactive plotting widget.
    
    Parameters
    ----------
    frame : `DataFrame`
        Data to be plotted.
    module : Python `module`, optional
        If given, load data functions from `module` instead of `__main__`.
    """
    
    AXIS_NONE = -1
    
    def __init__(self, frame, **kwargs):
        # check inputs
        cs, ds = frame.to_csds()
        self._check_inputs(cs, ds)
        # separate keys and values, support str and Parameter dicts
        self.labels = [label.name if hasattr(label, 'name') else label
                       for label in cs.keys()+ds.keys()]
        self.data = cs.values()+ds.values()
        # initial display: select longest two axes, select first item on remaining axes
        self.shape = ds.values()[0].shape
        self.ndim = len(self.shape) # data dimensions
        # show longest two axes first
        self.axes = list(np.argsort(self.shape)[-2:][::-1]) + [self.ndim,]
        # switch to 1d plotting if second longest axis has only one point
        if (self.ndim >= 2) and (self.shape[self.axes[1]] == 1):
            self.axes.pop(1)
        self.indices = [0]*self.ndim
        # initialize widgets
        self._ui(**kwargs)
        
    @staticmethod
    def _check_inputs(cs, ds):
        if not len(ds):
            raise ValueError('ds can not be empty')
        # silently cast all inputs to ndarray (this modifies the input dicts)
        for key, value in cs.iteritems():
            cs[key] = np.array(value)
        for key, value in ds.iteritems():
            ds[key] = np.array(value)
        # check shapes
        shape = ds.values()[0].shape
        if len(cs) != len(shape):
            raise ValueError('the number of dimensions of all arrays '+
                             'must be equal to the number of coordinates.')
        for arr in cs.values()+ds.values():
            if arr.shape != shape:
                raise ValueError('all coordinate and data matrices '+
                                 'must have the same shape.')
        
    def _ui(self, **kwargs):
        ''' create widget ui '''
        # axes panel
        w_axes = []
        values = OrderedDict([(label, axis) 
                              for axis, label in enumerate(self.labels[:self.ndim])
                              if self.shape[axis] > 1])
        if self.ndim > 0:
            w_axes.append(AxisWidget('x axis', axis=self.axes[0], options=values))
        if (len(self.axes) > 2):
            values['None'] = self.AXIS_NONE
            w_axes.append(AxisWidget('y axis', axis=self.axes[1], options=values))
        values = dict(AxisWidget.reverse_enumerate(self.labels[self.ndim:], self.ndim))
        w_axes.append(AxisWidget('z axis', axis=self.axes[-1], options=values))
        self.w_axes = widgets.FlexBox(orientation='horizontal', children=w_axes)
        for axis, w_axis in enumerate(w_axes):
            w_axis.on_trait_change(fix_args(self._on_axis_change, plot_axis=axis), 'axis')
            w_axis.on_trait_change(self._on_limit_change, ('min', 'max'))
            w_axis.on_trait_change(self._on_scaling_change, 'scaling')
        # coordinate sliders
        sliders = []
        for axis, coordinate in enumerate(self.labels[:self.ndim]):
            slider = FloatTextSliderWidget(description=coordinate)
            slider.on_trait_change(fix_args(self._on_slider_change, data_axis=axis), 'index')
            sliders.append(slider)
        self.w_sliders = widgets.Box()
        self.w_sliders.children = sliders
        # functions on data
        self.w_functions = FunctionWidget(**kwargs)
        self.w_functions.on_trait_change(self._on_function_change, 'function')
        # cursors
        self.w_cursors = widgets.HTML()
        self.w_cursors.align = 'center'
        self.w_cursors.padding = '10px'
        # controls panel (axes, functions, sliders)
        self.w_controls = widgets.Tab()
        self.w_controls.children = [self.w_functions, self.w_axes, 
                                    self.w_sliders, self.w_cursors]
        # plot panel
        self.w_plot = Figure()
        self.w_plot.on_zoom(self._on_zoom)
        self.w_plot.on_trait_change(self._on_cursors_change, 'cursors')
        # application window
        self.w_app = widgets.Box()
        self.w_app.children = [self.w_controls, self.w_plot]
    
    def _ipython_display_(self):
        self.update()
        self.w_plot.compile()
        self.w_app._ipython_display_()
        self.w_controls.set_title(0, 'Data functions')
        self.w_controls.set_title(1, 'Axis selection')
        self.w_controls.set_title(2, 'Data browser')
        self.w_controls.set_title(3, 'Cursors')

    def update(self):
        self._update_sliders()
        self._update_ranges()
        self._update_plot()
    
    def _update_plot(self):
        ''' update plot in widget ui '''
        #png_data = StringIO()
        #self.plot().canvas.print_png(png_data)
        #self.w_plot.value = png_data.getvalue()
        self.w_plot.fig = self.plot()

    def _update_sliders(self):
        ''' update slider ranges and states '''
        for idx, slider in enumerate(self.w_sliders.children):
            # update value ranges of all sliders
            slice_ = list(self.indices)
            slice_[idx] = slice(None)
            slider.values = tuple(self.data[idx][slice_])
            # disable sliders for the current plot axes
            slider.disabled = idx in self.axes

    def _update_ranges(self):
        ''' update x/y/zlim boxes '''
        for w_axis, axis in zip(self.w_axes.children, self.axes):
            # don't update limits of disabled axes
            if axis == self.AXIS_NONE:
                continue
            # retrieve data
            if w_axis.scaling == AxisWidget.SCALING_AUTO:
                if axis < self.ndim:
                    xs = self.data[axis][self.slice]
                else:
                    # apply function to slice
                    xs = self.data_slice[-1]
            elif w_axis.scaling == AxisWidget.SCALING_FULL:
                if axis < self.ndim:
                    xs = self.data[axis]
                else:
                    # apply function to complete data set
                    # TODO: slow
                    xss = [self.data[ax] for ax in self.axes 
                           if ax != self.AXIS_NONE]
                    xs = self.function(*xss)
            else:
                # manual scaling
                continue
            # update limits
            w_axis.on_trait_change(self._on_limit_change, ('min', 'max'), True)
            w_axis.set_limits(np.min(xs), np.max(xs))
            w_axis.on_trait_change(self._on_limit_change, ('min', 'max'), False)
    
    #
    # Figure callbacks
    #
    @property
    def cursors(self):
        if (self.w_plot.cursors is not None) and len(self.w_plot.cursors):
            return self.w_plot.cursors[0]
        else:
            return []
        
    @cursors.setter
    def cursors(self, cursors):
        self.w_plot.cursors = [cursors]
    
    def _on_zoom(self, _, xlim, ylim):
        ''' update limit text boxes when a zoom event occurs '''
        w_xaxis, w_yaxis = self.w_axes.children[:2]
        for w_axis, lim in [(w_xaxis, xlim), (w_yaxis, ylim)]:
            # update scaling button
            w_axis.on_trait_change(self._on_scaling_change, 'scaling', True)
            w_axis.scaling = AxisWidget.SCALING_MANUAL
            w_axis.on_trait_change(self._on_scaling_change, 'scaling', False)
            # update axis values
            w_axis.on_trait_change(self._on_limit_change, ('min', 'max'), True)
            w_axis.set_limits(lim[0], lim[1])
            w_axis.on_trait_change(self._on_limit_change, ('min', 'max'), False)
        
    def _on_cursors_change(self, _, __, cursors):
        ''' update cursors table when a cursor is created/moved '''
        # html short-hands
        def tag(tag, content, style=None):
            if style is None:
                return '<{0}>{1}</{0}>'.format(tag, content)
            else:
                return '<{0} style="{2}">{1}</{0}>'.format(tag, content, style)
        tht = lambda content: tag('th', content, 'border-bottom: solid 1px;')
        thl = lambda content: tag('th', content, 'border-right: solid 1px;')
        thtl = lambda content: tag('th', content, 'border-bottom: solid 1px; border-right: solid 1px;')
        tdl = lambda content: tag('td', content, 'padding: 5px; border-right: solid 1px;')
        td = lambda content: tag('td', content, 'padding: 5px;')
        tr = lambda content: tag('tr', content)
        table = lambda content: tag('table', content)
        
        if (cursors != None) and len(cursors) and len(cursors[0]):
            rows = []
            rows.append(tr(thl('#') + 
                           tht('x') + thtl('y') + 
                           tht(u'&Delta;x') + tht(u'&Delta;y')))
            ref = cursors[0][0]
            for idx, pos in enumerate(cursors[0]):
                delta = [(a - r) if idx and (a is not None) and (r is not None) else None
                         for a, r in zip(pos, ref)]
                rows.append(tr(thl(idx+1) + 
                               td(pos[0]) + tdl(pos[1]) +
                               td(delta[0]) + td(delta[1])))
            self.w_cursors.value = table('\n'.join(rows))
        else:
            self.w_cursors.value = ''
        
        
    # 
    # Axis selection tab callbacks
    #
    def _on_axis_change(self, plot_axis, _, old_axis, new_axis):
        ''' value of axis selection dropdown changed handler '''
        if new_axis in self.axes:
            # first axis must not be disabled
            if old_axis == self.AXIS_NONE:
                # TODO: decline change. not easy since event handlers have
                # no return values and the value is locked during event
                # handling
                return
            # conflicting axis selection: swap axes
            swap_axis = self.axes.index(new_axis)
            self.axes[plot_axis] = new_axis
            # this triggers a call to on_set_axis that does the updates
            self.w_axes.children[swap_axis].axis = old_axis
        else:
            # normal update
            self.axes[plot_axis] = new_axis
            if (new_axis is self.AXIS_NONE) or (old_axis is self.AXIS_NONE):
                # enable/disable axis
                if new_axis == self.AXIS_NONE:
                    self.axes[plot_axis] = self.AXIS_NONE
                self.w_axes.children[plot_axis].disabled = \
                    (new_axis == self.AXIS_NONE)
                self._update_sliders()
            # update plot
            self._update_ranges()
            if (plot_axis != self.AXIS_NONE) and (plot_axis != len(self.axes)-1):
                self._update_sliders()
            self._update_plot()
        
    def _on_limit_change(self, plot_axis, old, new):
        ''' value of axis min/max input box changed handler '''
        w_xaxis, w_yaxis = self.w_axes.children[:2]
        self.w_plot.zoom(0, 
                         (w_xaxis.min, w_xaxis.max), 
                         (w_yaxis.min, w_yaxis.max))
        #self.update_plot()
    
    def _on_scaling_change(self):
        ''' axis autoscale state changed handler '''
        self._update_ranges()
        self._update_plot()

    #
    # Function tab callbacks
    #
    def _on_function_change(self):
        ''' data function changed handler '''
        self._update_ranges()
        self._update_plot()
    
    #
    # Slider tab callbacks
    #
    def _on_slider_change(self, data_axis, _, index):
        ''' excess coordinate slider changed handler '''
        # set current slider value
        self.indices[data_axis] = index
        # update other sliders and plot
        self.update()        
    
    #
    # Plotting-related code
    #
    @property
    def slice(self):
        """Return nd slice for the current selection."""
        slice_ = list(self.indices)
        for axis in self.axes[:-1]:
            if axis != self.AXIS_NONE:
                slice_[axis] = slice(None)
        return slice_    
    
    @property
    def data_slice(self):
        """Return slices of the data matrices for all active axes."""
        axes = [ax for ax in self.axes if ax != self.AXIS_NONE]
        # extract input data
        xss = [self.data[axis][self.slice] for axis in axes]
        # apply user function
        xss[-1] = self.function(*xss)
        # return
        return xss
    
    @staticmethod
    def pcolor_matrices(xs, ys):
        """
        Generate corner points of quadrilaterals that have the input points
        in their centers.
        
        Parameters
        ----------
        xs, ys: `ndarray`
            (N, M) coordinate matrices
        
        Returns
        -------
        xs, ys: `ndarray`
            (N+1, M+1) coordinate matrices
        """
        shape_out = [l+1 for l in xs.shape]
        xs_out = np.empty(shape_out)
        ys_out = np.empty(shape_out)
        # inner grid points are centered between data points, 
        # edge points are chosen such that the data points are
        # centered in the edge quadrilaterals
        for arr_out, arr in ((xs_out, xs), (ys_out, ys)):
            arr_out[1:-1,1:-1] = (arr[:-1,:-1]+arr[:-1,1:]+arr[1:,:-1]+arr[1:,1:])/4.
            arr_out[-1,1:-1] = arr_out[-2,1:-1]+(arr[-1,:-1]-arr[-2,:-1])
            arr_out[1:,-1] = arr_out[1:,-2]+(arr[:,-1]-arr[:,-2])
            arr_out[0,1:] = arr_out[1,1:]-(arr[1,:]-arr[0,:])
            arr_out[1:,0] = arr_out[1:,1]-(arr[:,1]-arr[:,0])
            arr_out[0,0] = arr_out[0,1]-(arr[0,1]-arr[0,0])
        return xs_out, ys_out
    
    def function(self, *xss):
        """Run selected function on data."""
        function = self.w_functions.function
        # pass the required number of function arguments
        argspec = inspect.getargspec(function)
        nargs = len(argspec.args)
        if argspec.defaults is not None:
            nargs -= len(argspec.defaults)
        try:
            if nargs == 0:
                raise ValueError('Data function must accept at least one ' + 
                                 'non-default argument.')
            if nargs == 1:
                zs = self.w_functions.function(xss[-1])
            elif (nargs == 2) and (len(xss) > 1):
                zs = self.w_functions.function(xss[0], xss[-1])
            elif (nargs == 3) and (len(xss) > 2):
                zs = self.w_functions.function(xss[0], xss[1], xss[2])
            else:
                raise ValueError('Data function requires more arguments than' +
                                 'available.')
        except:
            tb = VerboseTB()
            print tb.text(*sys.exc_info(), tb_offset=1)
            # return a matrix of NaNs
            zs = np.full_like(xss[0], np.nan)
        return zs
    
    @property
    def function_name(self):
        return self.w_functions.function.__name__
    
    def plot(self):
        """Return a figure containing a 1d or 2d plot of the selection."""
        xss = self.data_slice
        
        fig = plt.figure(figsize=(10,6))
        plt.close(fig)
        ax = fig.add_subplot(111)
        zlabel = '{1}({0})'.format(self.labels[self.axes[-1]], 
                                   self.function_name)
        if len(xss) == 3:
            xs, ys, zs = xss
            xs, ys = Plot.pcolor_matrices(xs, ys)
            pl = ax.pcolormesh(xs, ys, zs)
            cb = fig.colorbar(pl, ax=ax)
            w_axes = self.w_axes.children
            ax.set_xlim(w_axes[0].min, w_axes[0].max)
            ax.set_ylim(w_axes[1].min, w_axes[1].max)
            pl.set_clim(w_axes[-1].min, w_axes[-1].max)
            ax.set_xlabel(self.labels[self.axes[0]])
            ax.set_ylabel(self.labels[self.axes[1]])
            cb.set_label(zlabel)
        elif len(xss) == 2:
            xs, zs = xss
            pl = ax.plot(xs, zs)
            w_axes = self.w_axes.children
            ax.set_xlim(w_axes[0].min, w_axes[0].max)
            ax.set_ylim(w_axes[-1].min, w_axes[-1].max)
            ax.set_xlabel(self.labels[self.axes[0]])
            ax.set_ylabel(zlabel)
        return fig