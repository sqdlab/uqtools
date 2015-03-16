'''Interactive plotting tool based on IPython.widgets

Authors:
 * Markus Jerger, 2015
'''

import numpy as np
import matplotlib.pyplot as plt

from IPython.html import widgets
from IPython.display import display, clear_output, HTML, Javascript
from IPython.core.ultratb import VerboseTB
import IPython.utils.traitlets as traitlets
from collections import OrderedDict

import inspect
import os
import sys
import imp
from cStringIO import StringIO
from base64 import b64encode

from .helpers import fix_args


class FigureWidget(widgets.DOMWidget):
    #_view_name = traitlets.Unicode('FigureView', sync=True)
    #_view_name = traitlets.Unicode('ZoomFogireView', sync=True)
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
            height: 1px;
            border-style: solid none;
            cursor: row-resize;
        }
        .Cursor .vRuler {
            top: 0px;
            left: -1px;
            width: 1px;
            height: 100%;
            border-style: none solid;
            cursor: col-resize;
        }
        .Cursor .iRuler {
            position: absolute;
            top: -2px;
            left: -2px;
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
        self.fig.canvas.print_figure(png_data, format='png')
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
        super(FigureWidget, self).__init__(**kwargs)
        self._zoom_handlers = widgets.CallbackDispatcher()
        self.on_zoom(self.zoom)
        self.on_msg(self._handle_messages)
        
    def compile(self):
        '''
        push CSS and JS to the browser
        '''
        # load and display js
        plotpy_fn = inspect.getfile(inspect.currentframe())
        plotpy_path = os.path.dirname(os.path.abspath(plotpy_fn))
        js_fn = os.path.join(plotpy_path, 'widgets', 'FigureWidget.js')
        js = Javascript(file(js_fn).read())
        display(js)
        # display css
        display(self._style)
        
    def _ipython_display_(self):
        # push CSS and JS to the browser
        self.compile()
        # display self
        super(FigureWidget, self)._ipython_display_()
        
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
        """Register a callback to execute when Axes are zoomed.

        The callback will be called with three arguments,
        the axis index, new xlim and new ylim.

        Parameters
        ----------
        remove : bool (optional)
            Set to true to remove the callback from the list of callbacks."""
        self._zoom_handlers.register_callback(callback, remove=remove)
        
    def zoom(self, axis, xlim=None, ylim=None, update=True, **kwargs):
        ''' 
        Set axis limits.
        
        Input:
            axis (int) - axis index
            u_min, u_max (float) - horizontal axis limits
            v_min, v_max (float) - vertical axis limits
        '''
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
        ''' set zoom to _zoom_history item at idx '''
        for axis, limits in enumerate(self._zoom_history[idx]):
            self.zoom(axis, *limits, update=False)
        self.update()
        
    def zoom_reset(self):
        ''' reset zoom to its initial value when the fig was set '''
        self._zoom_to(0)
        
    def zoom_prev(self):
        ''' zoom to previous item in the zoom history '''
        if self._zoom_index > -len(self._zoom_history):
            self._zoom_to(self._zoom_index-1)
        
    def zoom_next(self):
        ''' zoom to next item in the zoom history '''
        if self._zoom_index < -1:
            self._zoom_to(self._zoom_index+1)



class AxisWidget(widgets.ContainerWidget):
    '''
    An axis selecting and limit setting widget.
    Combines a drop-down, min/max float inputs and an autoscale flag.

    Traits:
        axis (Integer) - selected axis
        min, max (Float) - limit box
        scaling (Integer) - autoscale mode (SCALING_ constants) 
    '''
    axis = traitlets.Integer()
    min = traitlets.Float()
    max = traitlets.Float()
    scaling = traitlets.Integer()
    disabled = traitlets.Bool()

    SCALING_MANUAL = 0
    SCALING_AUTO = 1
    SCALING_FULL = 2
    
    def __init__(self, description, values, **kwargs):
        '''
        Input:
            description (str) - descriptive text
            values (str:int dict) - axis labels
        '''
        values = OrderedDict(values)
        self._w_select = widgets.DropdownWidget(description=description,
                                                values=values)
        traitlets.link((self._w_select, 'value'), (self, 'axis'))
        self._w_min = widgets.FloatTextWidget(description='min')
        traitlets.link((self._w_min, 'value'), (self, 'min'))
        self._w_max = widgets.FloatTextWidget(description='max')
        traitlets.link((self._w_max, 'value'), (self, 'max'))
        self._w_auto = widgets.ToggleButtonsWidget(description='scaling')
        self._w_auto.values = OrderedDict((('manual', self.SCALING_MANUAL),
                                          ('auto', self.SCALING_AUTO),
                                          ('full', self.SCALING_FULL)))
        traitlets.link((self._w_auto, 'value'), (self, 'scaling'))
        traitlets.link((self, 'disabled'), (self._w_auto, 'disabled'),
                       (self._w_min, 'disabled'), (self._w_max, 'disabled'))
        axis = kwargs.pop('axis', values.values()[0])
        scaling = kwargs.pop('scaling', self.SCALING_AUTO)
        super(AxisWidget, self).__init__(axis=axis, scaling=scaling, **kwargs)
        self.children = (self._w_select, self._w_min, self._w_max, self._w_auto)
        self.on_trait_change(self._on_limit_change, ('min', 'max'))
        
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
            
            

class FunctionWidget(widgets.ContainerWidget):
    '''
    A function selector widget.
    A select box combined with a code editor.
    
    Traits:
        function - selected function
    '''
    function = traitlets.Any()
    
    def __init__(self, module=None):
        '''
        Input:
            module - functions from module are added to the select box 
        '''
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
        if module is not None:
            self.load_module(module)
        # create ui
        self._w_select = widgets.SelectWidget()
        self._w_select.set_css({'width':'150px', 'height':'150px'})
        self._w_source = widgets.TextareaWidget()
        self._w_source.set_css({'height':'140px', 'font-family':'monospace'})
        self._w_compile = widgets.ButtonWidget(description='compile')
        self._w_compile.set_css({'margin-top':'5px'})
        self._w_compile.on_click(self._on_compile)
        self._w_output = widgets.HTMLWidget()
        self._w_output.set_css({'margin-top':'5px', 'width':'300px', 
                                'font-family':'monospace'})
        self._w_compile_box = widgets.ContainerWidget()
        self._w_compile_box.children = [self._w_compile, self._w_output]
        self.children = (self._w_select, self._w_source, self._w_compile_box)
        traitlets.link((self, 'function'), (self._w_select, 'value'))
        # install event handlers
        self._w_select.on_trait_change(self._on_select, 'value')
        self.on_displayed(self._on_displayed)
        # populate select box
        self.update()
    
    def _on_displayed(self, _):
        ''' set correct CSS classes '''
        self.remove_class('vbox')
        self.add_class('hbox')
        
    def _on_select(self, _, function):
        ''' show source code when a function is selected '''
        self._w_source.value = getattr(self.sources, function.__name__)
        self._w_output.visible = False
        
    def _on_compile(self, _):
        # compile code
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
        ''' pretty-print common compiler exceptions '''
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
        ''' update select box, set current selection to function '''
        functions = OrderedDict([(name, func) for name, func 
                                 in sorted(self.functions.__dict__.iteritems()) 
                                 if not name.startswith('__')])
        self._w_select.values = functions
        if function is not None:
            self._w_select.value = function
        elif self.default is not None:
            self._w_select.value = self.default
    
    def compile(self, source):
        '''
        compile source code
        
        the source code is stored in self.sources, the compiled function is 
        stored in self.functions.
        
        Input:
            source (str) - source code
        Returns:
            compiled function object
        '''
        code = compile(source, '<string>', 'single')
        if len(code.co_names) != 1:
            raise ValueError('source code must define exactly one name')
        name, = code.co_names
        exec code in globals(), self.functions.__dict__
        setattr(self.sources, name, source)
        return getattr(self.functions, name)
    
    def load_module(self, module):
        ''' 
        examine all callables in a module and add them to the list
        of functions.
        
        Input:
            module - any object that has a __dict__
        Remarks:
        '''
        for key in module.__dict__.keys():
            if key.startswith('__'):
                continue
            func = getattr(module, key)
            if not callable(func):
                continue
            # store functions and code in internal modules
            setattr(self.functions, func.__name__, func)
            source = inspect.cleandoc(inspect.getsource(func))
            setattr(self.sources, func.__name__, source)
        if 'default' in module.__dict__:
            self.default = getattr(module, 'default')
            
            
                        
class FloatTextSliderWidget(widgets.ContainerWidget):
    '''
    A slider with associated float input box.
    The components update each other.

    Traits:
        index - slider position
        value - text box value (not guaranteed to be an element of values)
    '''
    index = traitlets.Integer()
    value = traitlets.Float()
    values = traitlets.Tuple()
    disabled = traitlets.Bool()
    
    def __init__(self, description, values=[0], **kwargs):
        '''
        Input:
            description - descriptive text
            values (tuple) - valid float values
        '''
        super(FloatTextSliderWidget, self).__init__(**kwargs)
        # create widgets    
        self._w_slider = widgets.IntSliderWidget(description=description,
                                                 min=0, value=0, readout=False)
        traitlets.link((self._w_slider, 'value'), (self, 'index'))
        self._w_text = widgets.BoundedFloatTextWidget(value=values[0])
        traitlets.link((self._w_text, 'value'), (self, 'value'))
        self.children = (self._w_slider, self._w_text)
        # register event handlers
        self._w_slider.on_trait_change(self._on_index_change, 'value')
        self._w_text.on_trait_change(self._on_value_change, 'value')
        self.on_trait_change(self._on_values_change, 'values')
        traitlets.link((self._w_text, 'disabled'), (self._w_slider, 'disabled'),
                       (self, 'disabled'))
        self.on_displayed(self._on_displayed)
        # store values and trigger range update
        self.values = values
        self.index = 0
    
    def _on_displayed(self, _):
        ''' apply styles '''
        self.remove_class('vbox')
        self.add_class('hbox')        

    def _on_values_change(self, _, values):
        ''' update ranges on values change '''
        self._w_slider.max = len(values)-1
        self._w_text.min = min(values)
        self._w_text.max = max(values)
        self._on_index_change()
        
    def _on_index_change(self):
        ''' set value on index change '''
        value = self.values[self.index]
        self._w_text.value = value
            
    def _on_value_change(self):
        ''' set index on value change '''
        values = np.array(self.values)
        self.index = np.argmin(np.abs(values-self.value))



class Plot(object):
    '''
    An interactive plotting widget.
    '''
    AXIS_NONE = -1
    
    def __init__(self, cs, ds):
        '''
        Input:
            cs - coordinate dictionary
            ds - data dictionary
        '''
        # check inputs
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
        self._ui()
        
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
        
    def _ui(self):
        ''' create widget ui '''
        # axes panel
        w_axes = []
        values = OrderedDict([(label, axis) 
                              for axis, label in enumerate(self.labels[:self.ndim])
                              if self.shape[axis] > 1])
        if self.ndim > 0:
            w_axes.append(AxisWidget('x axis', axis=self.axes[0], values=values))
        if (len(self.axes) > 2):
            values['None'] = self.AXIS_NONE
            w_axes.append(AxisWidget('y axis', axis=self.axes[1], values=values))
        values = dict(AxisWidget.reverse_enumerate(self.labels[self.ndim:], self.ndim))
        w_axes.append(AxisWidget('z axis', axis=self.axes[-1], values=values))
        self.w_axes = widgets.ContainerWidget(children=w_axes)
        for axis, w_axis in enumerate(w_axes):
            w_axis.on_trait_change(fix_args(self.on_axis_change, plot_axis=axis), 'axis')
            w_axis.on_trait_change(self.on_limit_change, ('min', 'max'))
            w_axis.on_trait_change(self.on_scaling_change, 'scaling')
        # coordinate sliders
        sliders = []
        for axis, coordinate in enumerate(self.labels[:self.ndim]):
            slider = FloatTextSliderWidget(description=coordinate)
            slider.on_trait_change(fix_args(self.on_slider_change, data_axis=axis), 'index')
            sliders.append(slider)
        self.w_sliders = widgets.ContainerWidget()
        self.w_sliders.children = sliders
        # functions on data
        self.w_functions = FunctionWidget()
        self.w_functions.on_trait_change(self.on_function_change, 'function')
        # cursors
        self.w_cursors = widgets.HTMLWidget()
        self.w_cursors.set_css({'align':'center', 'padding': '10px'})
        # controls panel (axes, functions, sliders)
        self.w_controls = widgets.TabWidget()
        self.w_controls.children = [self.w_functions, self.w_axes, 
                                    self.w_sliders, self.w_cursors]
        # plot panel
        self.w_plot = FigureWidget()
        self.w_plot.on_zoom(self.on_zoom)
        self.w_plot.on_trait_change(self.on_cursors_change, 'cursors')
        # application window
        self.w_app = widgets.ContainerWidget()
        self.w_app.children = [self.w_controls, self.w_plot]
    
    def _ipython_display_(self):
        self.update()
        self.w_plot.compile()
        self.w_app._ipython_display_()
        self.w_axes.remove_class('vbox')
        self.w_axes.add_class('hbox')
        self.w_controls.set_title(0, 'Data functions')
        self.w_controls.set_title(1, 'Axis selection')
        self.w_controls.set_title(2, 'Data browser')
        self.w_controls.set_title(3, 'Cursors')

    def update(self):
        self.update_sliders()
        self.update_ranges()
        self.update_plot()
    
    def update_plot(self):
        ''' update plot in widget ui '''
        #png_data = StringIO()
        #self.plot().canvas.print_png(png_data)
        #self.w_plot.value = png_data.getvalue()
        self.w_plot.fig = self.plot()

    def update_sliders(self):
        ''' update slider ranges and states '''
        for idx, slider in enumerate(self.w_sliders.children):
            # update value ranges of all sliders
            slice_ = list(self.indices)
            slice_[idx] = slice(None)
            slider.values = tuple(self.data[idx][slice_])
            # disable sliders for the current plot axes
            slider.disabled = idx in self.axes

    def update_ranges(self):
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
            w_axis.on_trait_change(self.on_limit_change, ('min', 'max'), True)
            w_axis.min = np.min(xs)
            w_axis.max = np.max(xs)
            w_axis.on_trait_change(self.on_limit_change, ('min', 'max'), False)
    
    #
    # FigureWidget callbacks
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
    
    def on_zoom(self, _, xlim, ylim):
        ''' update limit text boxes when a zoom event occurs '''
        w_xaxis, w_yaxis = self.w_axes.children[:2]
        for w_axis, lim in [(w_xaxis, xlim), (w_yaxis, ylim)]:
            # update scaling button
            w_axis.on_trait_change(self.on_scaling_change, 'scaling', True)
            w_axis.scaling = AxisWidget.SCALING_MANUAL
            w_axis.on_trait_change(self.on_scaling_change, 'scaling', False)
            # update axis values
            w_axis.on_trait_change(self.on_limit_change, ('min', 'max'), True)
            w_axis.min = lim[0]
            w_axis.max = lim[1]
            w_axis.on_trait_change(self.on_limit_change, ('min', 'max'), False)
        
    def on_cursors_change(self, _, __, cursors):
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
    def on_axis_change(self, plot_axis, _, old_axis, new_axis):
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
                self.update_sliders()
            # update plot
            self.update_ranges()
            if (plot_axis != self.AXIS_NONE) and (plot_axis != len(self.axes)-1):
                self.update_sliders()
            self.update_plot()
        
    def on_limit_change(self, plot_axis, old, new):
        ''' value of axis min/max input box changed handler '''
        w_xaxis, w_yaxis = self.w_axes.children[:2]
        self.w_plot.zoom(0, 
                         (w_xaxis.min, w_xaxis.max), 
                         (w_yaxis.min, w_yaxis.max))
        #self.update_plot()
    
    def on_scaling_change(self):
        ''' axis autoscale state changed handler '''
        self.update_ranges()
        self.update_plot()

    #
    # Function tab callbacks
    #
    def on_function_change(self):
        ''' data function changed handler '''
        self.update_ranges()
        self.update_plot()
    
    #
    # Slider tab callbacks
    #
    def on_slider_change(self, data_axis, _, index):
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
        ''' return nd slice for the current selection '''
        slice_ = list(self.indices)
        for axis in self.axes[:-1]:
            if axis != self.AXIS_NONE:
                slice_[axis] = slice(None)
        return slice_    
    
    @property
    def data_slice(self):
        ''' return slices of the data matrices for all active axes '''
        axes = [ax for ax in self.axes if ax != self.AXIS_NONE]
        # extract input data
        xss = [self.data[axis][self.slice] for axis in axes]
        # apply user function
        xss[-1] = self.function(*xss)
        # return
        return xss
    
    @staticmethod
    def pcolor_matrices(xs, ys):
        '''
        Generate corner points of quadrilaterals that have the input points
        in their centers.
        
        Input:
            xs, ys: (N,M) coordinate matrices
        Output:
            xs, ys: (N+1,M+1) coordinate matrices
        '''
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
        ''' run selected function on data '''
        function = self.w_functions.function
        # pass the required number of function arguments
        argspec = inspect.getargspec(function)
        try:
            if len(argspec.args) == 0:
                raise ValueError('Data function must accept at least one ' + 
                                 'argument.')
            if len(argspec.args) == 1:
                zs = self.w_functions.function(xss[-1])
            elif (len(argspec.args) == 2) and (len(xss) > 1):
                zs = self.w_functions.function(xss[0], xss[-1])
            elif (len(argspec.args) == 3) and (len(xss) > 2):
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
        ''' return a figure containing a 1d or 2d plot of the selection '''
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