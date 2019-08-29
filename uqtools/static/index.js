define(["jupyter-js-widgets"], function(__WEBPACK_EXTERNAL_MODULE_2__) { return /******/ (function(modules) { // webpackBootstrap
/******/ 	// The module cache
/******/ 	var installedModules = {};
/******/
/******/ 	// The require function
/******/ 	function __webpack_require__(moduleId) {
/******/
/******/ 		// Check if module is in cache
/******/ 		if(installedModules[moduleId])
/******/ 			return installedModules[moduleId].exports;
/******/
/******/ 		// Create a new module (and put it into the cache)
/******/ 		var module = installedModules[moduleId] = {
/******/ 			exports: {},
/******/ 			id: moduleId,
/******/ 			loaded: false
/******/ 		};
/******/
/******/ 		// Execute the module function
/******/ 		modules[moduleId].call(module.exports, module, module.exports, __webpack_require__);
/******/
/******/ 		// Flag the module as loaded
/******/ 		module.loaded = true;
/******/
/******/ 		// Return the exports of the module
/******/ 		return module.exports;
/******/ 	}
/******/
/******/
/******/ 	// expose the modules object (__webpack_modules__)
/******/ 	__webpack_require__.m = modules;
/******/
/******/ 	// expose the module cache
/******/ 	__webpack_require__.c = installedModules;
/******/
/******/ 	// __webpack_public_path__
/******/ 	__webpack_require__.p = "";
/******/
/******/ 	// Load entry module and return exports
/******/ 	return __webpack_require__(0);
/******/ })
/************************************************************************/
/******/ ([
/* 0 */
/***/ (function(module, exports, __webpack_require__) {

	// Entry point for the notebook bundle containing custom model definitions.
	//
	// Setup notebook base URL
	//
	// Some static assets may be required by the custom widget javascript. The base
	// url for the notebook is not known at build time and is therefore computed
	// dynamically.
	__webpack_require__.p = document.querySelector('body').getAttribute('data-base-url') + 'nbextensions/uqtools/';
	
	// Export widget models and views, and the npm package version number.
	module.exports = __webpack_require__(1);
	module.exports['version'] = __webpack_require__(5).version;


/***/ }),
/* 1 */
/***/ (function(module, exports, __webpack_require__) {

	var widgets = __webpack_require__(2);
	var _ = __webpack_require__(3);
	
	function Axes(view, ax) {
	    // copy attributes: index, limits, navigable, zoomable, polar
	    $.extend(this, ax);
	    // calculate bbox
	    if (this.polar) {
	        this.rxy = this.x_max - this.x_min;
	        this.ruv = this.v_max - this.v_min;
	        this.x_min = this.x_min - this.rxy - 1;
	        this.x_max = this.x_max - 2; 
	        this.y_min = this.y_min + this.rxy - 1;
	        this.y_max = this.y_min - 2*this.rxy - 1;
	    } else {
	        this.xscale = this.x_max - this.x_min;
	        this.yscale = this.y_min - this.y_max;
	        this.uscale = this.u_max - this.u_min;
	        this.vscale = this.v_max - this.v_min;
	    }
	    // create in DOM
	    this.$el = $('<div />')
	        .addClass('Axes')
	        .css('position', 'absolute')
	        .css('left', this.x_min)
	        .css('top', this.y_max)
	        .css('width', this.x_max - this.x_min)
	        .css('height', this.y_min - this.y_max)
	        .css('border-radius', this.polar ? '100%' : 'none')
	        .appendTo(view.$el);
	    
	    this.destroy = function() {
	        this.$el.remove();
	    }
	    
	    this.mouseXY = function(event, clip) {
	        /**
	         * transform MouseEvent clientX/clientY to subplot X/Y
	         */
	        var rect = this.$el[0].getBoundingClientRect();
	        var x = event.clientX - rect.left,
	            y = event.clientY - rect.top;
	        if((clip != undefined) && (!clip)) {
	            return [x, y];
	        } else {
	            return [Math.max(0, Math.min(rect.width, x)),
	                    Math.max(0, Math.min(rect.height, y))];
	        }
	    }
	    
	    this.mouseXIn = function(event) {
	        var rect = this.$el[0].getBoundingClientRect();
	        return (event.clientX >= rect.left) && (event.clientX <= rect.right);
	    }
	    
	    this.mouseYIn = function(event) {
	        var rect = this.$el[0].getBoundingClientRect();
	        return (event.clientY >= rect.top) && (event.clientY <= rect.bottom);
	    }
	
	    this.mouseIn = function(event) {
	        var rect = this.$el[0].getBoundingClientRect();
	        return ((event.clientX >= rect.left) && (event.clientX <= rect.right) &&
	                (event.clientY >= rect.top) && (event.clientY <= rect.bottom));
	    }
	    
	    this.transform = function(u, v) {
	        /**
	         * tranform user coordinates into pixel coordinates relative to 
	         * the top left corner of ax
	         */
	        if (this.polar) {
	            return [this.rxy + v*this.rxy/this.ruv * Math.cos(u),
	                    this.rxy - v*this.rxy/this.ruv * Math.sin(u)];
	        } else {
	            return [(u - this.u_min) / this.uscale * this.xscale,
	                    (1 - (v - this.v_min) / this.vscale) * this.yscale];
	        }
	    }
	    
	    this.transformInverse = function(x, y) {
	        /**
	         * transform pixel x, y relative to the top left corner of
	         * ax into user coordinates
	         */
	        if (this.polar) {
	            var	xt = x - this.rxy,
	                yt = y - this.rxy,
	                u = (Math.atan2(-yt, xt) + 2*Math.PI) % (2*Math.PI),
	                v = Math.sqrt(xt*xt + yt*yt) / this.rxy*this.ruv;
	            return [u, v];
	        } else {
	            return [this.u_min + x / this.xscale * this.uscale,
	                    this.v_max - y / this.yscale * this.vscale]; 
	        }
	    }
	}
	
	
	var FigureView = widgets.DOMWidgetView.extend({
	    render: function() {
	        /**
	         * Create plot image
	         */
	        //console.log('ConsoleView render');
	        this.$el.addClass('Figure');
	        // force position
	        this.$el
	            .css('position', 'relative')
	            .on('remove', this, this.figureClose);
	        this.$image = $('<img />').appendTo(this.$el);
	        // create context menu
	        this.$menu = $('<menu />')
	            .attr('type', 'context')
	            .attr('id', (Math.random()*(1<<64)).toString(16).substr(2))
	            .appendTo(this.$el);
	        $('<menuitem />')
	            .attr('label', 'Print to notebook')
	            .on('click', this, this.figurePrint)
	            .appendTo(this.$menu);
	        $('<menuitem />')
	            .attr('label', 'Clear cell output')
	            .on('click', this, this.figureClear)
	            .appendTo(this.$menu);
	        this.$el.attr('contextmenu', this.$menu[0].id);
	        // create axes
	        this.axes = [];
	        this.figureUpdate();
	    },
	    
	    figureUpdate: function() {
	        /**
	         * Update plot image, create subplot divs
	         */
	        // show image
	        var image_src = ('data:image/' + this.model.get('_format') + 
	                         ';base64,' + this.model.get('_b64image'));
	        this.$image.attr('src', image_src);
	        // delete all subplots
	        for(var idx=0; idx<this.axes.length; idx++) {
	            this.axes[idx].destroy();
	        }
	        this.axes = [];
	        // re-create subplots
	        var axes = this.model.get('axes');
	        for(var idx=0; idx<axes.length; idx++) {
	            this.axes[idx] = new Axes(this, axes[idx]);
	        }
	    },
	    
	    figurePrint: function(event) {
	        // send print event to python
	        event.data.send({'event': 'print'});
	    },
	    
	    figureClear: function(event) {
	        // sent clear event to python
	        event.data.send({'event': 'clear'});
	    },
	    
	    figureClose: function(event) {
	        // send close event to python
	        event.data.send({'event': 'close'});
	    },
	    
	    update: function() {
	        //console.log('FigureView update');
	        FigureView.__super__.update.apply(this);
	        this.figureUpdate();
	    },
	
	});
	
	
	var ZoomFigureView = FigureView.extend({
	    render: function() {
	        ZoomFigureView.__super__.render.apply(this);
	        this.zoomUpdate();
	    },
	
	    zoomUpdate: function() {
	        /**
	         * install event handlers for zoomable axes
	         */
	        for(var idx=0; idx<this.axes.length; idx++) {
	            var ax = this.axes[idx];
	            if (ax.zoomable) {
	                var data = {view: this, ax: ax};
	                ax.$el
	                    .css('cursor', 'crosshair')
	                    .on('mousedown', data, this.zoomStart)
	            }
	        }
	    },
	    
	    update: function() {
	        ZoomFigureView.__super__.update.apply(this);
	        this.zoomUpdate();
	    },
	
	    zoomStart: function(event) {
	        /**
	         * Start dragging a zoom window
	         */
	        //console.log('zoomStart');
	        var view = event.data.view, 
	            ax = event.data.ax;
	        // only run when the main mouse button is pressed
	        if (event.button != 0) {
	            return;
	        }
	        // prevent text selection
	        event.preventDefault();
	        window.getSelection().removeAllRanges();
	        // handle double click
	        if ((event.timeStamp != undefined) && (event.timeStamp != 0)) {
	            // run a zoomReset if time between successive mouseDown events 
	            // was less than 250ms
	            if ((view.timeStamp != undefined) && 
	                (event.timeStamp - view.timeStamp < 250)) {
	                view.zoomReset(event);
	                return;
	            }
	            view.timeStamp = event.timeStamp;
	        }
	        // create zoom overlay
	        var rect = event.target.getBoundingClientRect();
	        var canvas = $('<canvas />')
	            .css('position', 'absolute')
	            .css('left', 0)
	            .css('top', 0)
	            .css('width', '100%')
	            .css('height', '100%')
	            .attr('width', rect.width)
	            .attr('height', rect.height)
	            .appendTo(event.target);
	        // save starting point
	        var zoom = {start: null, end: null};
	        zoom.start = ax.mouseXY(event);
	        var keys = {alt: false, shift: false, ctrl: false};
	        // install event handlers
	        var data = $.extend({}, event.data, {canvas: canvas, zoom: zoom, keys: keys});
	        $(document).on('mousemove', data, view.zoomMove);
	        $(document).one('mouseup', data, view.zoomFinish);
	        $(document).on('keydown keyup', data, view.zoomKeyDown);
	    },
	
	    zoomMove: function(event) {
	        /**
	         * Resize zoom window according to mouse position
	         */
	        var view = event.data.view,
	            ax = event.data.ax,
	            zoom = event.data.zoom;
	        zoom.end = ax.mouseXY(event);
	        view.zoomDraw(event);
	    },
	    
	    zoomDraw: function(event) {
	        /*
	         * Draw zoom window
	         */
	        var ax = event.data.ax,
	            zoom = event.data.zoom,
	            keys = event.data.keys,
	            canvas = event.data.canvas;
	        // draw zoom rectangle
	        var rect = canvas[0].getBoundingClientRect();
	        var ctx = canvas[0].getContext('2d');
	        ctx.fillStyle = "rgba(0,0,0,0.1)";
	        ctx.clearRect(0, 0, rect.width, rect.height);
	        ctx.fillRect(0, 0, rect.width, rect.height);
	        var x = Math.min(zoom.start[0], zoom.end[0]),
	            y = Math.min(zoom.start[1], zoom.end[1]),
	            w = Math.abs(zoom.start[0] - zoom.end[0]),
	            h = Math.abs(zoom.start[1] - zoom.end[1]);
	        if (keys.ctrl) {
	            y = 0;
	            h = rect.height;
	        }
	        if (keys.shift) {
	            x = 0;
	            w = rect.width;
	        }
	        ctx.clearRect(x, y, w, h);
	        ctx.strokeRect(x, y, w, h);
	    },
	    
	    zoomKeyDown: function(event) {
	        var view = event.data.view,
	            keys = event.data.keys;
	        if (event.which == 27) {
	            view.zoomAbort(event);
	            event.stopPropagation();
	        }
	        // redraw when any of the modifier keys is depressed
	        if ((event.shiftKey != keys.shift) || 
	            (event.ctrlKey != keys.ctrl) || 
	            (event.altKey != keys.alt)) {
	            keys.shift = event.shiftKey;
	            keys.alt = event.altKey;
	            keys.ctrl = event.ctrlKey;
	            view.zoomDraw(event);
	        } 
	    },
	    
	    zoomAbort: function(event) {
	        //console.log('zoomAbort');
	        var view = event.data.view;
	        // disable event handlers
	        $(document).off('mousemove', view.zoomMove);
	        $(document).off('mouseup', view.zoomFinish);
	        $(document).off('keydown keyup', view.zoomKeyDown);
	        // remove zoom box
	        event.data.canvas.remove();
	    },
	
	    zoomFinish: function(event) {
	        //console.log('zoomFinish');
	        var view = event.data.view,
	            ax = event.data.ax,
	            keys = event.data.keys,
	            zoom = event.data.zoom;
	        // clear zoom box and event handlers
	        view.zoomAbort(event);
	        // check for non-zero zoom area
	        zoom.end = ax.mouseXY(event);
	        if ((zoom.start[0] != zoom.end[0]) && 
	            (zoom.start[1] != zoom.end[1])) {
	            // transform zoom rectangle into user coordinates
	            var x_min = Math.min(zoom.start[0], zoom.end[0]),
	                x_max = Math.max(zoom.start[0], zoom.end[0]),
	                y_min = Math.max(zoom.start[1], zoom.end[1]),
	                y_max = Math.min(zoom.start[1], zoom.end[1]);
	            var uv_min = ax.transformInverse(x_min, y_min),
	                uv_max = ax.transformInverse(x_max, y_max);
	            if (keys.ctrl) {
	                uv_min[1] = ax.v_min;
	                uv_max[1] = ax.v_max;
	            }
	            if (keys.shift) {
	                uv_min[0] = ax.u_min;
	                uv_max[0] = ax.u_max;
	            }
	            // update model
	            view.send({event: 'zoom', 
	                       axis: ax.index, 
	                       u_min: uv_min[0], u_max: uv_max[0], 
	                       v_min: uv_min[1], v_max: uv_max[1]});
	        }
	    },
	    
	    
	    zoomReset: function(event) {
	        //console.log('zoomReset');
	        var view = event.data.view, ax = event.data.ax;
	        view.send({event: 'zoom_reset', axis: ax.index});
	    },
	
	});
	
	
	var Cursor = Backbone.Model.extend({
	    defaults: {
	        'template': false,
	        'direction': 3,
	        'position': [null, null],
	    },
	    
	    validate: function(attr, options) {
	        /**
	         * Check if coordinates are within the axes.
	         */
	        var u = attr.position[0], v = attr.position[1];
	        if ((u != null) && ((u < attr.ax.u_min) || (u > attr.ax.u_max)) ||
	            (v != null) && ((v < attr.ax.v_min) || (v > attr.ax.v_max))) {
	            return '!';
	        }
	    return null;
	    }
	});
	
	var CursorCollection = Backbone.Collection.extend({
	    model: Cursor,
	    
	    initialize: function(models, options) {
	        /**
	         * Store ax.
	         */
	        this.ax = options.ax;
	    }        
	});
	
	var CursorView = Backbone.View.extend({
	    // TODO: ax must be provided to the constructor, how to check?
	    
	    events: {
	        'mousedown .hRuler': 'grab',
	        'mousedown .vRuler': 'grab',
	        'mousedown .iRuler': 'grab',
	    },
	    
	    render: function() {
	        /**
	         * Create a new cursor in the DOM
	         *
	         * Input:
	         *   ax (object) - associated Axes
	         *   direction (int) - 1: vertical, 2: horizontal, 3: both
	         *   labels (bool) - show labels
	         * Return:
	         *   jQuery selector of the cursor group
	         */
	        var ax = this.model.get('ax'),
	            template = this.model.get('template'),
	            direction = this.model.get('direction');
	
	        // create DOM elements
	        this.$el
	            .addClass('Cursor')
	            .attr('hidden', !this.model.isValid());
	        if (ax.polar) {
	            this.$el
	                .css('border-radius', '100%');
	        }
	        if (direction & 1) {
	            $('<div />')
	                .addClass('hRuler')
	                .css('position', 'absolute')
	                .appendTo(this.$el);
	            if(!template) {
	                $('<div />')
	                    .addClass('hLabel')
	                    .css('position', 'absolute')
	                    .css('left', Math.abs(ax.x_max - ax.x_min) + 4)
	                    .appendTo(this.$el);
	            }
	        }
	        if (!ax.polar && (direction & 2)) {
	            $('<div />')
	                .addClass('vRuler')
	                .css('position', 'absolute')
	                .appendTo(this.$el);
	            if(!template) {
	                $('<div />')
	                    .addClass('vLabel')
	                    .css('position', 'absolute')
	                    .css('top', Math.abs(ax.y_max - ax.y_min) + 1)
	                    .appendTo(this.$el);
	            }
	        }
	        if (direction == 3) {
	            $('<div />')
	                .addClass('iRuler')
	                .css('position', 'absolute')
	                .appendTo(this.$el);
	        }
	        this.update();
	    },
	    
	    update: function() {
	        /**
	         * Move cursor to the position stored in the model.
	         */
	        var ax = this.model.get('ax'),
	            direction = this.model.get('direction');
	        this.position = _.clone(this.model.get('position'));
	        if ((this.position[0] != null) || (this.position[1] != null)) {
	            var xy = ax.transform(this.position[0], this.position[1]);
	            this.move(direction, xy[0], xy[1]);
	        }
	    },
	    
	    formatNumber: function(u, u_min, u_max, x_min, x_max) {
	        /**
	         * Format u such that its number of significant digits is at least
	         * equal to log10 of the number of pixels between x_min and x_max.
	         */
	        var precision = Math.ceil(Math.LOG10E * Math.log(Math.abs(x_max - x_min))),
	            u_msd = Math.LOG10E * Math.log(Math.abs(u)),
	            s_msd = Math.LOG10E * Math.log(Math.max(Math.abs(u_max), Math.abs(u_min))),
	            d_msd = Math.LOG10E * Math.log(Math.abs(u_max - u_min)),
	            digits = Math.max(1, Math.ceil(precision + u_msd - d_msd));
	        if ((Math.abs(s_msd) > 3) || (Math.abs(u_msd) > 3)) {
	            return u.toExponential(digits - 1);
	        } else {
	            return u.toPrecision(digits);
	        }
	        
	    },
	
	    move: function(direction, x, y) {
	        /**
	         * Move a cursor and update its labels.
	         *
	         * Input:
	         *   direction (int) - 1: vertical, 2: horizontal, 3: both
	         *   x, y (float) - new position in screen coordinates
	         */
	        var ax = this.model.get('ax'),
	            uv = ax.transformInverse(x, y),
	            x_invalid = (uv[0] < ax.u_min) || (uv[0] > ax.u_max),
	            y_invalid = (uv[1] < ax.v_min) || (uv[1] > ax.v_max);
	        x = Math.max(0, Math.min(ax.x_max-ax.x_min, x));
	        if (ax.polar) {
	            y = (ax.y_min - ax.y_max)/2.
	        } else {
	            y = Math.max(0, Math.min(ax.y_min-ax.y_max, y));
	        }
	        
	        if (ax.polar) {
	            this.$el.css('transform', 'rotate(' + (-uv[0]).toString() + 'rad)');
	        }
	        if (direction & 1) {
	            this.position[1] = uv[1];
	            this.$el.children('.hRuler').css('top', y - 1);
	            this.$el.children('.iRuler').css('top', y - 1);
	            var hlabel = this.$el.children('.hLabel');
	            if (y_invalid) {
	                hlabel.text('');
	            } else {
	                hlabel.text(this.formatNumber(uv[1], ax.v_min, ax.v_max, ax.y_min, ax.y_max));
	                hlabel.css('top', y - hlabel.height()/2);
	            }
	        }
	        if (!ax.polar && (direction & 2)) {
	            this.position[0] = uv[0];
	            this.$el.children('.vRuler').css('left', x - 1);
	            this.$el.children('.iRuler').css('left', x - 1);
	            var vlabel = this.$el.children('.vLabel');
	            if (x_invalid) {
	                vlabel.text('');
	            } else {
	                vlabel.text(this.formatNumber(uv[0], ax.u_min, ax.u_max, ax.x_min, ax.x_max));
	                vlabel.css('left', x - vlabel.width()/2);
	            }
	        }            
	    },
	
	    grab: function(event, arg2, arg3) {
	        /**
	         * Start dragging a cursor or ruler.
	         *  
	         * Installs mousemove/mouseup event handlers to follow
	         * mouse cursor movements.
	         * Dragging is limited to the x or y axis if the lines
	         * of the cursor instead of the intersection are dragged.
	         * If the template cursor is dragged, a new cursor/ruler
	         * is created before dragging.
	         */
	        var self = this;
	        // prevent text selection
	        event.preventDefault();
	        window.getSelection().removeAllRanges();
	        // prevent zoom start
	        event.stopPropagation()            
	        // determine which cursor element is being dragged
	        var direction = 3;
	        if ($(event.target).hasClass('hRuler')) {
	            direction = 1;
	        } else if ($(event.target).hasClass('vRuler')) {
	            direction = 2;
	        }
	        if (self.model.get('template')) {
	            self.trigger('duplicate', direction, event);
	        } else {
	            // install event handlers
	            var data = {cursor: self, direction: direction};
	            $(document).on('mousemove', data, self.drag);
	            $(document).one('mouseup', data, self.drop);
	        }
	    },
	    
	    drag: function(event) {
	        /**
	         * Respond to mousemove events.
	         *
	         * Updates cursor position and coordinate labels.
	         */
	        var self = event.data.cursor,
	            direction = event.data.direction,
	            ax = self.model.get('ax');
	        var pos = ax.mouseXY(event, false);
	        self.move(direction, pos[0], pos[1]);
	    },
	    
	    drop: function(event) {
	        /**
	         * Respond to mouseup events.
	         *
	         * Remove event handlers. 
	         * Delete the cursor if it is out of range.
	         */
	        var self = event.data.cursor;
	        // remove event handlers
	        $(document).off('mousemove', self.drag);
	        // save new cursor position
	        self.model.set('position', self.position);
	        // destroy model if the new position is invalid
	        if(!self.model.isValid()) {
	            self.model.destroy();
	        }
	    }
	
	});
	
	var CursorCollectionView = Backbone.View.extend({
	    
	    initialize: function(options) {
	        this.views = [];
	        this.listenTo(this.model, 'add', this.add);
	        this.listenTo(this.model, 'remove', this.remove);
	        this.listenTo(this.model, 'reset', this.reset);
	    },
	    
	    render: function() {
	        //console.log('render');
	        this.$el.addClass('Cursors');
	        // create template cursor
	        if (this.model.ax.navigable) {
	            var tCursor = new Cursor({ax: this.model.ax, template: true}),
	                tView = new CursorView({model: tCursor});
	            tView.render();
	            this.listenTo(tView, 'duplicate', this.templateGrab);
	            this.$el.append(tView.el);
	        }
	        // create user cursors
	        this.update();
	    },
	    
	    update: function() {
	        //console.log('update');
	        // re-create user cursors
	        //this.reset();
	        this.model.each(this.add, this);
	    },
	    
	    templateGrab: function(direction, event) {
	        this.model.add(new Cursor({ax: this.model.ax, direction: direction}));
	        // ...add callback runs ...
	        // grab new view
	        this.views[this.views.length-1].grab(event);
	    },
	    
	    reset: function() {
	        //console.log('reset');
	        // delete all cursors
	        this.views.forEach(function(view) {
	            view.remove();
	        });
	    },
	    
	    add: function(cursor) {
	        /**
	         * Create a cursor view
	         */
	        //console.log('add');
	        var view = new CursorView({ax: this.ax, model: cursor});
	        this.views.push(view);
	        this.$el.append(view.el);
	        view.render();
	    },
	    
	    remove: function(cursor, model, options) {
	        /**
	         * Remove a cursor view
	         */
	        //console.log('remove');
	        var idx = options.index;
	        if (this.views.length > idx) {
	            this.views[idx].remove();
	            this.views.splice(idx, 1);
	        }
	    }
	});
	
	var ZoomCursorFigureView = ZoomFigureView.extend({
	    render: function() {
	        // render ZoomFigureView
	        ZoomCursorFigureView.__super__.render.apply(this);
	        this.cursorsUpdate();
	    },
	    
	    cursorsUpdate: function() {
	        // delete existing cursors
	        if (this.axes_cursors == undefined) {
	            this.axes_cursors = [];
	        }
	        this.axes_cursors.forEach(function(cursors) {
	            cursors.destroy();
	        });
	        // create new cursors
	        var nb_axes_cursors = this.model.get('cursors');
	        for (var idx=0; idx<this.axes.length; idx++) {
	            // create new set of cursors
	            var ax = this.axes[idx],
	                nb_cursors = nb_axes_cursors[idx];
	            if(!ax.navigable) continue;
	            var cursors = new CursorCollection([], {ax: ax});
	            if (nb_cursors != undefined) {
	                nb_cursors.forEach(function(nb_cursor) {
	                    var direction = (((nb_cursor[0] != null) ? 2 : 0) + 
	                                     ((nb_cursor[1] != null) ? 1 : 0));
	                    var cursor = new Cursor({ax: ax, direction: direction, 
	                                             position: nb_cursor});
	                    //if (cursor.isValid()) {
	                        cursors.add(cursor);
	                    //} else {
	                    //    cursor.destroy();
	                    //}
	                });
	            }
	            // create a view for the cursors
	            var view = new CursorCollectionView({model: cursors});
	            this.axes[idx].$el.append(view.el);
	            view.render();
	            // fire change if number of cursors changed (due to validation)
	            //if ((nb_cursors != undefined) && (nb_cursors.length != cursors.length)) {
	            //    this.cursorChange({collection: cursors});
	            //}
	            // listen to cursor events
	            //this.listenTo(cursors, 'add', this.change);
	            this.listenTo(cursors, 'remove', this.cursorChange);
	            this.listenTo(cursors, 'change', this.cursorChange);
	        }
	    },
	    
	    update: function() {
	        //console.log('PlotCursorView update');
	        ZoomCursorFigureView.__super__.update.apply(this);
	        this.cursorsUpdate();
	    },
	    
	    cursorChange: function(changed, options) {
	        //console.log('PlotCursorView change');
	        if (changed.collection == undefined) return;
	        // calculate new cursors for axis
	        var cursors = [];
	        changed.collection.forEach(function(cursor) {
	            cursors.push(cursor.get('position')); 
	        });
	        // update model
	        var cursorss = _.clone(this.model.get('cursors'));
	        for(var idx=0; idx<this.axes.length; idx++) {
	            if (cursorss[idx] == undefined)
	                cursorss[idx] = [];
	        }
	        cursorss[changed.collection.ax.index] = cursors;
	        this.model.set('cursors', cursorss);
	        this.touch();
	    }
	});
	
	
	module.exports = {
	    FigureView: FigureView,
	    ZoomFigureView: ZoomFigureView,
	    ZoomCursorFigureView: ZoomCursorFigureView
	};

/***/ }),
/* 2 */
/***/ (function(module, exports) {

	module.exports = __WEBPACK_EXTERNAL_MODULE_2__;

/***/ }),
/* 3 */
/***/ (function(module, exports, __webpack_require__) {

	var __WEBPACK_AMD_DEFINE_ARRAY__, __WEBPACK_AMD_DEFINE_RESULT__;/* WEBPACK VAR INJECTION */(function(global, module) {//     Underscore.js 1.9.1
	//     http://underscorejs.org
	//     (c) 2009-2018 Jeremy Ashkenas, DocumentCloud and Investigative Reporters & Editors
	//     Underscore may be freely distributed under the MIT license.
	
	(function() {
	
	  // Baseline setup
	  // --------------
	
	  // Establish the root object, `window` (`self`) in the browser, `global`
	  // on the server, or `this` in some virtual machines. We use `self`
	  // instead of `window` for `WebWorker` support.
	  var root = typeof self == 'object' && self.self === self && self ||
	            typeof global == 'object' && global.global === global && global ||
	            this ||
	            {};
	
	  // Save the previous value of the `_` variable.
	  var previousUnderscore = root._;
	
	  // Save bytes in the minified (but not gzipped) version:
	  var ArrayProto = Array.prototype, ObjProto = Object.prototype;
	  var SymbolProto = typeof Symbol !== 'undefined' ? Symbol.prototype : null;
	
	  // Create quick reference variables for speed access to core prototypes.
	  var push = ArrayProto.push,
	      slice = ArrayProto.slice,
	      toString = ObjProto.toString,
	      hasOwnProperty = ObjProto.hasOwnProperty;
	
	  // All **ECMAScript 5** native function implementations that we hope to use
	  // are declared here.
	  var nativeIsArray = Array.isArray,
	      nativeKeys = Object.keys,
	      nativeCreate = Object.create;
	
	  // Naked function reference for surrogate-prototype-swapping.
	  var Ctor = function(){};
	
	  // Create a safe reference to the Underscore object for use below.
	  var _ = function(obj) {
	    if (obj instanceof _) return obj;
	    if (!(this instanceof _)) return new _(obj);
	    this._wrapped = obj;
	  };
	
	  // Export the Underscore object for **Node.js**, with
	  // backwards-compatibility for their old module API. If we're in
	  // the browser, add `_` as a global object.
	  // (`nodeType` is checked to ensure that `module`
	  // and `exports` are not HTML elements.)
	  if (typeof exports != 'undefined' && !exports.nodeType) {
	    if (typeof module != 'undefined' && !module.nodeType && module.exports) {
	      exports = module.exports = _;
	    }
	    exports._ = _;
	  } else {
	    root._ = _;
	  }
	
	  // Current version.
	  _.VERSION = '1.9.1';
	
	  // Internal function that returns an efficient (for current engines) version
	  // of the passed-in callback, to be repeatedly applied in other Underscore
	  // functions.
	  var optimizeCb = function(func, context, argCount) {
	    if (context === void 0) return func;
	    switch (argCount == null ? 3 : argCount) {
	      case 1: return function(value) {
	        return func.call(context, value);
	      };
	      // The 2-argument case is omitted because we’re not using it.
	      case 3: return function(value, index, collection) {
	        return func.call(context, value, index, collection);
	      };
	      case 4: return function(accumulator, value, index, collection) {
	        return func.call(context, accumulator, value, index, collection);
	      };
	    }
	    return function() {
	      return func.apply(context, arguments);
	    };
	  };
	
	  var builtinIteratee;
	
	  // An internal function to generate callbacks that can be applied to each
	  // element in a collection, returning the desired result — either `identity`,
	  // an arbitrary callback, a property matcher, or a property accessor.
	  var cb = function(value, context, argCount) {
	    if (_.iteratee !== builtinIteratee) return _.iteratee(value, context);
	    if (value == null) return _.identity;
	    if (_.isFunction(value)) return optimizeCb(value, context, argCount);
	    if (_.isObject(value) && !_.isArray(value)) return _.matcher(value);
	    return _.property(value);
	  };
	
	  // External wrapper for our callback generator. Users may customize
	  // `_.iteratee` if they want additional predicate/iteratee shorthand styles.
	  // This abstraction hides the internal-only argCount argument.
	  _.iteratee = builtinIteratee = function(value, context) {
	    return cb(value, context, Infinity);
	  };
	
	  // Some functions take a variable number of arguments, or a few expected
	  // arguments at the beginning and then a variable number of values to operate
	  // on. This helper accumulates all remaining arguments past the function’s
	  // argument length (or an explicit `startIndex`), into an array that becomes
	  // the last argument. Similar to ES6’s "rest parameter".
	  var restArguments = function(func, startIndex) {
	    startIndex = startIndex == null ? func.length - 1 : +startIndex;
	    return function() {
	      var length = Math.max(arguments.length - startIndex, 0),
	          rest = Array(length),
	          index = 0;
	      for (; index < length; index++) {
	        rest[index] = arguments[index + startIndex];
	      }
	      switch (startIndex) {
	        case 0: return func.call(this, rest);
	        case 1: return func.call(this, arguments[0], rest);
	        case 2: return func.call(this, arguments[0], arguments[1], rest);
	      }
	      var args = Array(startIndex + 1);
	      for (index = 0; index < startIndex; index++) {
	        args[index] = arguments[index];
	      }
	      args[startIndex] = rest;
	      return func.apply(this, args);
	    };
	  };
	
	  // An internal function for creating a new object that inherits from another.
	  var baseCreate = function(prototype) {
	    if (!_.isObject(prototype)) return {};
	    if (nativeCreate) return nativeCreate(prototype);
	    Ctor.prototype = prototype;
	    var result = new Ctor;
	    Ctor.prototype = null;
	    return result;
	  };
	
	  var shallowProperty = function(key) {
	    return function(obj) {
	      return obj == null ? void 0 : obj[key];
	    };
	  };
	
	  var has = function(obj, path) {
	    return obj != null && hasOwnProperty.call(obj, path);
	  }
	
	  var deepGet = function(obj, path) {
	    var length = path.length;
	    for (var i = 0; i < length; i++) {
	      if (obj == null) return void 0;
	      obj = obj[path[i]];
	    }
	    return length ? obj : void 0;
	  };
	
	  // Helper for collection methods to determine whether a collection
	  // should be iterated as an array or as an object.
	  // Related: http://people.mozilla.org/~jorendorff/es6-draft.html#sec-tolength
	  // Avoids a very nasty iOS 8 JIT bug on ARM-64. #2094
	  var MAX_ARRAY_INDEX = Math.pow(2, 53) - 1;
	  var getLength = shallowProperty('length');
	  var isArrayLike = function(collection) {
	    var length = getLength(collection);
	    return typeof length == 'number' && length >= 0 && length <= MAX_ARRAY_INDEX;
	  };
	
	  // Collection Functions
	  // --------------------
	
	  // The cornerstone, an `each` implementation, aka `forEach`.
	  // Handles raw objects in addition to array-likes. Treats all
	  // sparse array-likes as if they were dense.
	  _.each = _.forEach = function(obj, iteratee, context) {
	    iteratee = optimizeCb(iteratee, context);
	    var i, length;
	    if (isArrayLike(obj)) {
	      for (i = 0, length = obj.length; i < length; i++) {
	        iteratee(obj[i], i, obj);
	      }
	    } else {
	      var keys = _.keys(obj);
	      for (i = 0, length = keys.length; i < length; i++) {
	        iteratee(obj[keys[i]], keys[i], obj);
	      }
	    }
	    return obj;
	  };
	
	  // Return the results of applying the iteratee to each element.
	  _.map = _.collect = function(obj, iteratee, context) {
	    iteratee = cb(iteratee, context);
	    var keys = !isArrayLike(obj) && _.keys(obj),
	        length = (keys || obj).length,
	        results = Array(length);
	    for (var index = 0; index < length; index++) {
	      var currentKey = keys ? keys[index] : index;
	      results[index] = iteratee(obj[currentKey], currentKey, obj);
	    }
	    return results;
	  };
	
	  // Create a reducing function iterating left or right.
	  var createReduce = function(dir) {
	    // Wrap code that reassigns argument variables in a separate function than
	    // the one that accesses `arguments.length` to avoid a perf hit. (#1991)
	    var reducer = function(obj, iteratee, memo, initial) {
	      var keys = !isArrayLike(obj) && _.keys(obj),
	          length = (keys || obj).length,
	          index = dir > 0 ? 0 : length - 1;
	      if (!initial) {
	        memo = obj[keys ? keys[index] : index];
	        index += dir;
	      }
	      for (; index >= 0 && index < length; index += dir) {
	        var currentKey = keys ? keys[index] : index;
	        memo = iteratee(memo, obj[currentKey], currentKey, obj);
	      }
	      return memo;
	    };
	
	    return function(obj, iteratee, memo, context) {
	      var initial = arguments.length >= 3;
	      return reducer(obj, optimizeCb(iteratee, context, 4), memo, initial);
	    };
	  };
	
	  // **Reduce** builds up a single result from a list of values, aka `inject`,
	  // or `foldl`.
	  _.reduce = _.foldl = _.inject = createReduce(1);
	
	  // The right-associative version of reduce, also known as `foldr`.
	  _.reduceRight = _.foldr = createReduce(-1);
	
	  // Return the first value which passes a truth test. Aliased as `detect`.
	  _.find = _.detect = function(obj, predicate, context) {
	    var keyFinder = isArrayLike(obj) ? _.findIndex : _.findKey;
	    var key = keyFinder(obj, predicate, context);
	    if (key !== void 0 && key !== -1) return obj[key];
	  };
	
	  // Return all the elements that pass a truth test.
	  // Aliased as `select`.
	  _.filter = _.select = function(obj, predicate, context) {
	    var results = [];
	    predicate = cb(predicate, context);
	    _.each(obj, function(value, index, list) {
	      if (predicate(value, index, list)) results.push(value);
	    });
	    return results;
	  };
	
	  // Return all the elements for which a truth test fails.
	  _.reject = function(obj, predicate, context) {
	    return _.filter(obj, _.negate(cb(predicate)), context);
	  };
	
	  // Determine whether all of the elements match a truth test.
	  // Aliased as `all`.
	  _.every = _.all = function(obj, predicate, context) {
	    predicate = cb(predicate, context);
	    var keys = !isArrayLike(obj) && _.keys(obj),
	        length = (keys || obj).length;
	    for (var index = 0; index < length; index++) {
	      var currentKey = keys ? keys[index] : index;
	      if (!predicate(obj[currentKey], currentKey, obj)) return false;
	    }
	    return true;
	  };
	
	  // Determine if at least one element in the object matches a truth test.
	  // Aliased as `any`.
	  _.some = _.any = function(obj, predicate, context) {
	    predicate = cb(predicate, context);
	    var keys = !isArrayLike(obj) && _.keys(obj),
	        length = (keys || obj).length;
	    for (var index = 0; index < length; index++) {
	      var currentKey = keys ? keys[index] : index;
	      if (predicate(obj[currentKey], currentKey, obj)) return true;
	    }
	    return false;
	  };
	
	  // Determine if the array or object contains a given item (using `===`).
	  // Aliased as `includes` and `include`.
	  _.contains = _.includes = _.include = function(obj, item, fromIndex, guard) {
	    if (!isArrayLike(obj)) obj = _.values(obj);
	    if (typeof fromIndex != 'number' || guard) fromIndex = 0;
	    return _.indexOf(obj, item, fromIndex) >= 0;
	  };
	
	  // Invoke a method (with arguments) on every item in a collection.
	  _.invoke = restArguments(function(obj, path, args) {
	    var contextPath, func;
	    if (_.isFunction(path)) {
	      func = path;
	    } else if (_.isArray(path)) {
	      contextPath = path.slice(0, -1);
	      path = path[path.length - 1];
	    }
	    return _.map(obj, function(context) {
	      var method = func;
	      if (!method) {
	        if (contextPath && contextPath.length) {
	          context = deepGet(context, contextPath);
	        }
	        if (context == null) return void 0;
	        method = context[path];
	      }
	      return method == null ? method : method.apply(context, args);
	    });
	  });
	
	  // Convenience version of a common use case of `map`: fetching a property.
	  _.pluck = function(obj, key) {
	    return _.map(obj, _.property(key));
	  };
	
	  // Convenience version of a common use case of `filter`: selecting only objects
	  // containing specific `key:value` pairs.
	  _.where = function(obj, attrs) {
	    return _.filter(obj, _.matcher(attrs));
	  };
	
	  // Convenience version of a common use case of `find`: getting the first object
	  // containing specific `key:value` pairs.
	  _.findWhere = function(obj, attrs) {
	    return _.find(obj, _.matcher(attrs));
	  };
	
	  // Return the maximum element (or element-based computation).
	  _.max = function(obj, iteratee, context) {
	    var result = -Infinity, lastComputed = -Infinity,
	        value, computed;
	    if (iteratee == null || typeof iteratee == 'number' && typeof obj[0] != 'object' && obj != null) {
	      obj = isArrayLike(obj) ? obj : _.values(obj);
	      for (var i = 0, length = obj.length; i < length; i++) {
	        value = obj[i];
	        if (value != null && value > result) {
	          result = value;
	        }
	      }
	    } else {
	      iteratee = cb(iteratee, context);
	      _.each(obj, function(v, index, list) {
	        computed = iteratee(v, index, list);
	        if (computed > lastComputed || computed === -Infinity && result === -Infinity) {
	          result = v;
	          lastComputed = computed;
	        }
	      });
	    }
	    return result;
	  };
	
	  // Return the minimum element (or element-based computation).
	  _.min = function(obj, iteratee, context) {
	    var result = Infinity, lastComputed = Infinity,
	        value, computed;
	    if (iteratee == null || typeof iteratee == 'number' && typeof obj[0] != 'object' && obj != null) {
	      obj = isArrayLike(obj) ? obj : _.values(obj);
	      for (var i = 0, length = obj.length; i < length; i++) {
	        value = obj[i];
	        if (value != null && value < result) {
	          result = value;
	        }
	      }
	    } else {
	      iteratee = cb(iteratee, context);
	      _.each(obj, function(v, index, list) {
	        computed = iteratee(v, index, list);
	        if (computed < lastComputed || computed === Infinity && result === Infinity) {
	          result = v;
	          lastComputed = computed;
	        }
	      });
	    }
	    return result;
	  };
	
	  // Shuffle a collection.
	  _.shuffle = function(obj) {
	    return _.sample(obj, Infinity);
	  };
	
	  // Sample **n** random values from a collection using the modern version of the
	  // [Fisher-Yates shuffle](http://en.wikipedia.org/wiki/Fisher–Yates_shuffle).
	  // If **n** is not specified, returns a single random element.
	  // The internal `guard` argument allows it to work with `map`.
	  _.sample = function(obj, n, guard) {
	    if (n == null || guard) {
	      if (!isArrayLike(obj)) obj = _.values(obj);
	      return obj[_.random(obj.length - 1)];
	    }
	    var sample = isArrayLike(obj) ? _.clone(obj) : _.values(obj);
	    var length = getLength(sample);
	    n = Math.max(Math.min(n, length), 0);
	    var last = length - 1;
	    for (var index = 0; index < n; index++) {
	      var rand = _.random(index, last);
	      var temp = sample[index];
	      sample[index] = sample[rand];
	      sample[rand] = temp;
	    }
	    return sample.slice(0, n);
	  };
	
	  // Sort the object's values by a criterion produced by an iteratee.
	  _.sortBy = function(obj, iteratee, context) {
	    var index = 0;
	    iteratee = cb(iteratee, context);
	    return _.pluck(_.map(obj, function(value, key, list) {
	      return {
	        value: value,
	        index: index++,
	        criteria: iteratee(value, key, list)
	      };
	    }).sort(function(left, right) {
	      var a = left.criteria;
	      var b = right.criteria;
	      if (a !== b) {
	        if (a > b || a === void 0) return 1;
	        if (a < b || b === void 0) return -1;
	      }
	      return left.index - right.index;
	    }), 'value');
	  };
	
	  // An internal function used for aggregate "group by" operations.
	  var group = function(behavior, partition) {
	    return function(obj, iteratee, context) {
	      var result = partition ? [[], []] : {};
	      iteratee = cb(iteratee, context);
	      _.each(obj, function(value, index) {
	        var key = iteratee(value, index, obj);
	        behavior(result, value, key);
	      });
	      return result;
	    };
	  };
	
	  // Groups the object's values by a criterion. Pass either a string attribute
	  // to group by, or a function that returns the criterion.
	  _.groupBy = group(function(result, value, key) {
	    if (has(result, key)) result[key].push(value); else result[key] = [value];
	  });
	
	  // Indexes the object's values by a criterion, similar to `groupBy`, but for
	  // when you know that your index values will be unique.
	  _.indexBy = group(function(result, value, key) {
	    result[key] = value;
	  });
	
	  // Counts instances of an object that group by a certain criterion. Pass
	  // either a string attribute to count by, or a function that returns the
	  // criterion.
	  _.countBy = group(function(result, value, key) {
	    if (has(result, key)) result[key]++; else result[key] = 1;
	  });
	
	  var reStrSymbol = /[^\ud800-\udfff]|[\ud800-\udbff][\udc00-\udfff]|[\ud800-\udfff]/g;
	  // Safely create a real, live array from anything iterable.
	  _.toArray = function(obj) {
	    if (!obj) return [];
	    if (_.isArray(obj)) return slice.call(obj);
	    if (_.isString(obj)) {
	      // Keep surrogate pair characters together
	      return obj.match(reStrSymbol);
	    }
	    if (isArrayLike(obj)) return _.map(obj, _.identity);
	    return _.values(obj);
	  };
	
	  // Return the number of elements in an object.
	  _.size = function(obj) {
	    if (obj == null) return 0;
	    return isArrayLike(obj) ? obj.length : _.keys(obj).length;
	  };
	
	  // Split a collection into two arrays: one whose elements all satisfy the given
	  // predicate, and one whose elements all do not satisfy the predicate.
	  _.partition = group(function(result, value, pass) {
	    result[pass ? 0 : 1].push(value);
	  }, true);
	
	  // Array Functions
	  // ---------------
	
	  // Get the first element of an array. Passing **n** will return the first N
	  // values in the array. Aliased as `head` and `take`. The **guard** check
	  // allows it to work with `_.map`.
	  _.first = _.head = _.take = function(array, n, guard) {
	    if (array == null || array.length < 1) return n == null ? void 0 : [];
	    if (n == null || guard) return array[0];
	    return _.initial(array, array.length - n);
	  };
	
	  // Returns everything but the last entry of the array. Especially useful on
	  // the arguments object. Passing **n** will return all the values in
	  // the array, excluding the last N.
	  _.initial = function(array, n, guard) {
	    return slice.call(array, 0, Math.max(0, array.length - (n == null || guard ? 1 : n)));
	  };
	
	  // Get the last element of an array. Passing **n** will return the last N
	  // values in the array.
	  _.last = function(array, n, guard) {
	    if (array == null || array.length < 1) return n == null ? void 0 : [];
	    if (n == null || guard) return array[array.length - 1];
	    return _.rest(array, Math.max(0, array.length - n));
	  };
	
	  // Returns everything but the first entry of the array. Aliased as `tail` and `drop`.
	  // Especially useful on the arguments object. Passing an **n** will return
	  // the rest N values in the array.
	  _.rest = _.tail = _.drop = function(array, n, guard) {
	    return slice.call(array, n == null || guard ? 1 : n);
	  };
	
	  // Trim out all falsy values from an array.
	  _.compact = function(array) {
	    return _.filter(array, Boolean);
	  };
	
	  // Internal implementation of a recursive `flatten` function.
	  var flatten = function(input, shallow, strict, output) {
	    output = output || [];
	    var idx = output.length;
	    for (var i = 0, length = getLength(input); i < length; i++) {
	      var value = input[i];
	      if (isArrayLike(value) && (_.isArray(value) || _.isArguments(value))) {
	        // Flatten current level of array or arguments object.
	        if (shallow) {
	          var j = 0, len = value.length;
	          while (j < len) output[idx++] = value[j++];
	        } else {
	          flatten(value, shallow, strict, output);
	          idx = output.length;
	        }
	      } else if (!strict) {
	        output[idx++] = value;
	      }
	    }
	    return output;
	  };
	
	  // Flatten out an array, either recursively (by default), or just one level.
	  _.flatten = function(array, shallow) {
	    return flatten(array, shallow, false);
	  };
	
	  // Return a version of the array that does not contain the specified value(s).
	  _.without = restArguments(function(array, otherArrays) {
	    return _.difference(array, otherArrays);
	  });
	
	  // Produce a duplicate-free version of the array. If the array has already
	  // been sorted, you have the option of using a faster algorithm.
	  // The faster algorithm will not work with an iteratee if the iteratee
	  // is not a one-to-one function, so providing an iteratee will disable
	  // the faster algorithm.
	  // Aliased as `unique`.
	  _.uniq = _.unique = function(array, isSorted, iteratee, context) {
	    if (!_.isBoolean(isSorted)) {
	      context = iteratee;
	      iteratee = isSorted;
	      isSorted = false;
	    }
	    if (iteratee != null) iteratee = cb(iteratee, context);
	    var result = [];
	    var seen = [];
	    for (var i = 0, length = getLength(array); i < length; i++) {
	      var value = array[i],
	          computed = iteratee ? iteratee(value, i, array) : value;
	      if (isSorted && !iteratee) {
	        if (!i || seen !== computed) result.push(value);
	        seen = computed;
	      } else if (iteratee) {
	        if (!_.contains(seen, computed)) {
	          seen.push(computed);
	          result.push(value);
	        }
	      } else if (!_.contains(result, value)) {
	        result.push(value);
	      }
	    }
	    return result;
	  };
	
	  // Produce an array that contains the union: each distinct element from all of
	  // the passed-in arrays.
	  _.union = restArguments(function(arrays) {
	    return _.uniq(flatten(arrays, true, true));
	  });
	
	  // Produce an array that contains every item shared between all the
	  // passed-in arrays.
	  _.intersection = function(array) {
	    var result = [];
	    var argsLength = arguments.length;
	    for (var i = 0, length = getLength(array); i < length; i++) {
	      var item = array[i];
	      if (_.contains(result, item)) continue;
	      var j;
	      for (j = 1; j < argsLength; j++) {
	        if (!_.contains(arguments[j], item)) break;
	      }
	      if (j === argsLength) result.push(item);
	    }
	    return result;
	  };
	
	  // Take the difference between one array and a number of other arrays.
	  // Only the elements present in just the first array will remain.
	  _.difference = restArguments(function(array, rest) {
	    rest = flatten(rest, true, true);
	    return _.filter(array, function(value){
	      return !_.contains(rest, value);
	    });
	  });
	
	  // Complement of _.zip. Unzip accepts an array of arrays and groups
	  // each array's elements on shared indices.
	  _.unzip = function(array) {
	    var length = array && _.max(array, getLength).length || 0;
	    var result = Array(length);
	
	    for (var index = 0; index < length; index++) {
	      result[index] = _.pluck(array, index);
	    }
	    return result;
	  };
	
	  // Zip together multiple lists into a single array -- elements that share
	  // an index go together.
	  _.zip = restArguments(_.unzip);
	
	  // Converts lists into objects. Pass either a single array of `[key, value]`
	  // pairs, or two parallel arrays of the same length -- one of keys, and one of
	  // the corresponding values. Passing by pairs is the reverse of _.pairs.
	  _.object = function(list, values) {
	    var result = {};
	    for (var i = 0, length = getLength(list); i < length; i++) {
	      if (values) {
	        result[list[i]] = values[i];
	      } else {
	        result[list[i][0]] = list[i][1];
	      }
	    }
	    return result;
	  };
	
	  // Generator function to create the findIndex and findLastIndex functions.
	  var createPredicateIndexFinder = function(dir) {
	    return function(array, predicate, context) {
	      predicate = cb(predicate, context);
	      var length = getLength(array);
	      var index = dir > 0 ? 0 : length - 1;
	      for (; index >= 0 && index < length; index += dir) {
	        if (predicate(array[index], index, array)) return index;
	      }
	      return -1;
	    };
	  };
	
	  // Returns the first index on an array-like that passes a predicate test.
	  _.findIndex = createPredicateIndexFinder(1);
	  _.findLastIndex = createPredicateIndexFinder(-1);
	
	  // Use a comparator function to figure out the smallest index at which
	  // an object should be inserted so as to maintain order. Uses binary search.
	  _.sortedIndex = function(array, obj, iteratee, context) {
	    iteratee = cb(iteratee, context, 1);
	    var value = iteratee(obj);
	    var low = 0, high = getLength(array);
	    while (low < high) {
	      var mid = Math.floor((low + high) / 2);
	      if (iteratee(array[mid]) < value) low = mid + 1; else high = mid;
	    }
	    return low;
	  };
	
	  // Generator function to create the indexOf and lastIndexOf functions.
	  var createIndexFinder = function(dir, predicateFind, sortedIndex) {
	    return function(array, item, idx) {
	      var i = 0, length = getLength(array);
	      if (typeof idx == 'number') {
	        if (dir > 0) {
	          i = idx >= 0 ? idx : Math.max(idx + length, i);
	        } else {
	          length = idx >= 0 ? Math.min(idx + 1, length) : idx + length + 1;
	        }
	      } else if (sortedIndex && idx && length) {
	        idx = sortedIndex(array, item);
	        return array[idx] === item ? idx : -1;
	      }
	      if (item !== item) {
	        idx = predicateFind(slice.call(array, i, length), _.isNaN);
	        return idx >= 0 ? idx + i : -1;
	      }
	      for (idx = dir > 0 ? i : length - 1; idx >= 0 && idx < length; idx += dir) {
	        if (array[idx] === item) return idx;
	      }
	      return -1;
	    };
	  };
	
	  // Return the position of the first occurrence of an item in an array,
	  // or -1 if the item is not included in the array.
	  // If the array is large and already in sort order, pass `true`
	  // for **isSorted** to use binary search.
	  _.indexOf = createIndexFinder(1, _.findIndex, _.sortedIndex);
	  _.lastIndexOf = createIndexFinder(-1, _.findLastIndex);
	
	  // Generate an integer Array containing an arithmetic progression. A port of
	  // the native Python `range()` function. See
	  // [the Python documentation](http://docs.python.org/library/functions.html#range).
	  _.range = function(start, stop, step) {
	    if (stop == null) {
	      stop = start || 0;
	      start = 0;
	    }
	    if (!step) {
	      step = stop < start ? -1 : 1;
	    }
	
	    var length = Math.max(Math.ceil((stop - start) / step), 0);
	    var range = Array(length);
	
	    for (var idx = 0; idx < length; idx++, start += step) {
	      range[idx] = start;
	    }
	
	    return range;
	  };
	
	  // Chunk a single array into multiple arrays, each containing `count` or fewer
	  // items.
	  _.chunk = function(array, count) {
	    if (count == null || count < 1) return [];
	    var result = [];
	    var i = 0, length = array.length;
	    while (i < length) {
	      result.push(slice.call(array, i, i += count));
	    }
	    return result;
	  };
	
	  // Function (ahem) Functions
	  // ------------------
	
	  // Determines whether to execute a function as a constructor
	  // or a normal function with the provided arguments.
	  var executeBound = function(sourceFunc, boundFunc, context, callingContext, args) {
	    if (!(callingContext instanceof boundFunc)) return sourceFunc.apply(context, args);
	    var self = baseCreate(sourceFunc.prototype);
	    var result = sourceFunc.apply(self, args);
	    if (_.isObject(result)) return result;
	    return self;
	  };
	
	  // Create a function bound to a given object (assigning `this`, and arguments,
	  // optionally). Delegates to **ECMAScript 5**'s native `Function.bind` if
	  // available.
	  _.bind = restArguments(function(func, context, args) {
	    if (!_.isFunction(func)) throw new TypeError('Bind must be called on a function');
	    var bound = restArguments(function(callArgs) {
	      return executeBound(func, bound, context, this, args.concat(callArgs));
	    });
	    return bound;
	  });
	
	  // Partially apply a function by creating a version that has had some of its
	  // arguments pre-filled, without changing its dynamic `this` context. _ acts
	  // as a placeholder by default, allowing any combination of arguments to be
	  // pre-filled. Set `_.partial.placeholder` for a custom placeholder argument.
	  _.partial = restArguments(function(func, boundArgs) {
	    var placeholder = _.partial.placeholder;
	    var bound = function() {
	      var position = 0, length = boundArgs.length;
	      var args = Array(length);
	      for (var i = 0; i < length; i++) {
	        args[i] = boundArgs[i] === placeholder ? arguments[position++] : boundArgs[i];
	      }
	      while (position < arguments.length) args.push(arguments[position++]);
	      return executeBound(func, bound, this, this, args);
	    };
	    return bound;
	  });
	
	  _.partial.placeholder = _;
	
	  // Bind a number of an object's methods to that object. Remaining arguments
	  // are the method names to be bound. Useful for ensuring that all callbacks
	  // defined on an object belong to it.
	  _.bindAll = restArguments(function(obj, keys) {
	    keys = flatten(keys, false, false);
	    var index = keys.length;
	    if (index < 1) throw new Error('bindAll must be passed function names');
	    while (index--) {
	      var key = keys[index];
	      obj[key] = _.bind(obj[key], obj);
	    }
	  });
	
	  // Memoize an expensive function by storing its results.
	  _.memoize = function(func, hasher) {
	    var memoize = function(key) {
	      var cache = memoize.cache;
	      var address = '' + (hasher ? hasher.apply(this, arguments) : key);
	      if (!has(cache, address)) cache[address] = func.apply(this, arguments);
	      return cache[address];
	    };
	    memoize.cache = {};
	    return memoize;
	  };
	
	  // Delays a function for the given number of milliseconds, and then calls
	  // it with the arguments supplied.
	  _.delay = restArguments(function(func, wait, args) {
	    return setTimeout(function() {
	      return func.apply(null, args);
	    }, wait);
	  });
	
	  // Defers a function, scheduling it to run after the current call stack has
	  // cleared.
	  _.defer = _.partial(_.delay, _, 1);
	
	  // Returns a function, that, when invoked, will only be triggered at most once
	  // during a given window of time. Normally, the throttled function will run
	  // as much as it can, without ever going more than once per `wait` duration;
	  // but if you'd like to disable the execution on the leading edge, pass
	  // `{leading: false}`. To disable execution on the trailing edge, ditto.
	  _.throttle = function(func, wait, options) {
	    var timeout, context, args, result;
	    var previous = 0;
	    if (!options) options = {};
	
	    var later = function() {
	      previous = options.leading === false ? 0 : _.now();
	      timeout = null;
	      result = func.apply(context, args);
	      if (!timeout) context = args = null;
	    };
	
	    var throttled = function() {
	      var now = _.now();
	      if (!previous && options.leading === false) previous = now;
	      var remaining = wait - (now - previous);
	      context = this;
	      args = arguments;
	      if (remaining <= 0 || remaining > wait) {
	        if (timeout) {
	          clearTimeout(timeout);
	          timeout = null;
	        }
	        previous = now;
	        result = func.apply(context, args);
	        if (!timeout) context = args = null;
	      } else if (!timeout && options.trailing !== false) {
	        timeout = setTimeout(later, remaining);
	      }
	      return result;
	    };
	
	    throttled.cancel = function() {
	      clearTimeout(timeout);
	      previous = 0;
	      timeout = context = args = null;
	    };
	
	    return throttled;
	  };
	
	  // Returns a function, that, as long as it continues to be invoked, will not
	  // be triggered. The function will be called after it stops being called for
	  // N milliseconds. If `immediate` is passed, trigger the function on the
	  // leading edge, instead of the trailing.
	  _.debounce = function(func, wait, immediate) {
	    var timeout, result;
	
	    var later = function(context, args) {
	      timeout = null;
	      if (args) result = func.apply(context, args);
	    };
	
	    var debounced = restArguments(function(args) {
	      if (timeout) clearTimeout(timeout);
	      if (immediate) {
	        var callNow = !timeout;
	        timeout = setTimeout(later, wait);
	        if (callNow) result = func.apply(this, args);
	      } else {
	        timeout = _.delay(later, wait, this, args);
	      }
	
	      return result;
	    });
	
	    debounced.cancel = function() {
	      clearTimeout(timeout);
	      timeout = null;
	    };
	
	    return debounced;
	  };
	
	  // Returns the first function passed as an argument to the second,
	  // allowing you to adjust arguments, run code before and after, and
	  // conditionally execute the original function.
	  _.wrap = function(func, wrapper) {
	    return _.partial(wrapper, func);
	  };
	
	  // Returns a negated version of the passed-in predicate.
	  _.negate = function(predicate) {
	    return function() {
	      return !predicate.apply(this, arguments);
	    };
	  };
	
	  // Returns a function that is the composition of a list of functions, each
	  // consuming the return value of the function that follows.
	  _.compose = function() {
	    var args = arguments;
	    var start = args.length - 1;
	    return function() {
	      var i = start;
	      var result = args[start].apply(this, arguments);
	      while (i--) result = args[i].call(this, result);
	      return result;
	    };
	  };
	
	  // Returns a function that will only be executed on and after the Nth call.
	  _.after = function(times, func) {
	    return function() {
	      if (--times < 1) {
	        return func.apply(this, arguments);
	      }
	    };
	  };
	
	  // Returns a function that will only be executed up to (but not including) the Nth call.
	  _.before = function(times, func) {
	    var memo;
	    return function() {
	      if (--times > 0) {
	        memo = func.apply(this, arguments);
	      }
	      if (times <= 1) func = null;
	      return memo;
	    };
	  };
	
	  // Returns a function that will be executed at most one time, no matter how
	  // often you call it. Useful for lazy initialization.
	  _.once = _.partial(_.before, 2);
	
	  _.restArguments = restArguments;
	
	  // Object Functions
	  // ----------------
	
	  // Keys in IE < 9 that won't be iterated by `for key in ...` and thus missed.
	  var hasEnumBug = !{toString: null}.propertyIsEnumerable('toString');
	  var nonEnumerableProps = ['valueOf', 'isPrototypeOf', 'toString',
	    'propertyIsEnumerable', 'hasOwnProperty', 'toLocaleString'];
	
	  var collectNonEnumProps = function(obj, keys) {
	    var nonEnumIdx = nonEnumerableProps.length;
	    var constructor = obj.constructor;
	    var proto = _.isFunction(constructor) && constructor.prototype || ObjProto;
	
	    // Constructor is a special case.
	    var prop = 'constructor';
	    if (has(obj, prop) && !_.contains(keys, prop)) keys.push(prop);
	
	    while (nonEnumIdx--) {
	      prop = nonEnumerableProps[nonEnumIdx];
	      if (prop in obj && obj[prop] !== proto[prop] && !_.contains(keys, prop)) {
	        keys.push(prop);
	      }
	    }
	  };
	
	  // Retrieve the names of an object's own properties.
	  // Delegates to **ECMAScript 5**'s native `Object.keys`.
	  _.keys = function(obj) {
	    if (!_.isObject(obj)) return [];
	    if (nativeKeys) return nativeKeys(obj);
	    var keys = [];
	    for (var key in obj) if (has(obj, key)) keys.push(key);
	    // Ahem, IE < 9.
	    if (hasEnumBug) collectNonEnumProps(obj, keys);
	    return keys;
	  };
	
	  // Retrieve all the property names of an object.
	  _.allKeys = function(obj) {
	    if (!_.isObject(obj)) return [];
	    var keys = [];
	    for (var key in obj) keys.push(key);
	    // Ahem, IE < 9.
	    if (hasEnumBug) collectNonEnumProps(obj, keys);
	    return keys;
	  };
	
	  // Retrieve the values of an object's properties.
	  _.values = function(obj) {
	    var keys = _.keys(obj);
	    var length = keys.length;
	    var values = Array(length);
	    for (var i = 0; i < length; i++) {
	      values[i] = obj[keys[i]];
	    }
	    return values;
	  };
	
	  // Returns the results of applying the iteratee to each element of the object.
	  // In contrast to _.map it returns an object.
	  _.mapObject = function(obj, iteratee, context) {
	    iteratee = cb(iteratee, context);
	    var keys = _.keys(obj),
	        length = keys.length,
	        results = {};
	    for (var index = 0; index < length; index++) {
	      var currentKey = keys[index];
	      results[currentKey] = iteratee(obj[currentKey], currentKey, obj);
	    }
	    return results;
	  };
	
	  // Convert an object into a list of `[key, value]` pairs.
	  // The opposite of _.object.
	  _.pairs = function(obj) {
	    var keys = _.keys(obj);
	    var length = keys.length;
	    var pairs = Array(length);
	    for (var i = 0; i < length; i++) {
	      pairs[i] = [keys[i], obj[keys[i]]];
	    }
	    return pairs;
	  };
	
	  // Invert the keys and values of an object. The values must be serializable.
	  _.invert = function(obj) {
	    var result = {};
	    var keys = _.keys(obj);
	    for (var i = 0, length = keys.length; i < length; i++) {
	      result[obj[keys[i]]] = keys[i];
	    }
	    return result;
	  };
	
	  // Return a sorted list of the function names available on the object.
	  // Aliased as `methods`.
	  _.functions = _.methods = function(obj) {
	    var names = [];
	    for (var key in obj) {
	      if (_.isFunction(obj[key])) names.push(key);
	    }
	    return names.sort();
	  };
	
	  // An internal function for creating assigner functions.
	  var createAssigner = function(keysFunc, defaults) {
	    return function(obj) {
	      var length = arguments.length;
	      if (defaults) obj = Object(obj);
	      if (length < 2 || obj == null) return obj;
	      for (var index = 1; index < length; index++) {
	        var source = arguments[index],
	            keys = keysFunc(source),
	            l = keys.length;
	        for (var i = 0; i < l; i++) {
	          var key = keys[i];
	          if (!defaults || obj[key] === void 0) obj[key] = source[key];
	        }
	      }
	      return obj;
	    };
	  };
	
	  // Extend a given object with all the properties in passed-in object(s).
	  _.extend = createAssigner(_.allKeys);
	
	  // Assigns a given object with all the own properties in the passed-in object(s).
	  // (https://developer.mozilla.org/docs/Web/JavaScript/Reference/Global_Objects/Object/assign)
	  _.extendOwn = _.assign = createAssigner(_.keys);
	
	  // Returns the first key on an object that passes a predicate test.
	  _.findKey = function(obj, predicate, context) {
	    predicate = cb(predicate, context);
	    var keys = _.keys(obj), key;
	    for (var i = 0, length = keys.length; i < length; i++) {
	      key = keys[i];
	      if (predicate(obj[key], key, obj)) return key;
	    }
	  };
	
	  // Internal pick helper function to determine if `obj` has key `key`.
	  var keyInObj = function(value, key, obj) {
	    return key in obj;
	  };
	
	  // Return a copy of the object only containing the whitelisted properties.
	  _.pick = restArguments(function(obj, keys) {
	    var result = {}, iteratee = keys[0];
	    if (obj == null) return result;
	    if (_.isFunction(iteratee)) {
	      if (keys.length > 1) iteratee = optimizeCb(iteratee, keys[1]);
	      keys = _.allKeys(obj);
	    } else {
	      iteratee = keyInObj;
	      keys = flatten(keys, false, false);
	      obj = Object(obj);
	    }
	    for (var i = 0, length = keys.length; i < length; i++) {
	      var key = keys[i];
	      var value = obj[key];
	      if (iteratee(value, key, obj)) result[key] = value;
	    }
	    return result;
	  });
	
	  // Return a copy of the object without the blacklisted properties.
	  _.omit = restArguments(function(obj, keys) {
	    var iteratee = keys[0], context;
	    if (_.isFunction(iteratee)) {
	      iteratee = _.negate(iteratee);
	      if (keys.length > 1) context = keys[1];
	    } else {
	      keys = _.map(flatten(keys, false, false), String);
	      iteratee = function(value, key) {
	        return !_.contains(keys, key);
	      };
	    }
	    return _.pick(obj, iteratee, context);
	  });
	
	  // Fill in a given object with default properties.
	  _.defaults = createAssigner(_.allKeys, true);
	
	  // Creates an object that inherits from the given prototype object.
	  // If additional properties are provided then they will be added to the
	  // created object.
	  _.create = function(prototype, props) {
	    var result = baseCreate(prototype);
	    if (props) _.extendOwn(result, props);
	    return result;
	  };
	
	  // Create a (shallow-cloned) duplicate of an object.
	  _.clone = function(obj) {
	    if (!_.isObject(obj)) return obj;
	    return _.isArray(obj) ? obj.slice() : _.extend({}, obj);
	  };
	
	  // Invokes interceptor with the obj, and then returns obj.
	  // The primary purpose of this method is to "tap into" a method chain, in
	  // order to perform operations on intermediate results within the chain.
	  _.tap = function(obj, interceptor) {
	    interceptor(obj);
	    return obj;
	  };
	
	  // Returns whether an object has a given set of `key:value` pairs.
	  _.isMatch = function(object, attrs) {
	    var keys = _.keys(attrs), length = keys.length;
	    if (object == null) return !length;
	    var obj = Object(object);
	    for (var i = 0; i < length; i++) {
	      var key = keys[i];
	      if (attrs[key] !== obj[key] || !(key in obj)) return false;
	    }
	    return true;
	  };
	
	
	  // Internal recursive comparison function for `isEqual`.
	  var eq, deepEq;
	  eq = function(a, b, aStack, bStack) {
	    // Identical objects are equal. `0 === -0`, but they aren't identical.
	    // See the [Harmony `egal` proposal](http://wiki.ecmascript.org/doku.php?id=harmony:egal).
	    if (a === b) return a !== 0 || 1 / a === 1 / b;
	    // `null` or `undefined` only equal to itself (strict comparison).
	    if (a == null || b == null) return false;
	    // `NaN`s are equivalent, but non-reflexive.
	    if (a !== a) return b !== b;
	    // Exhaust primitive checks
	    var type = typeof a;
	    if (type !== 'function' && type !== 'object' && typeof b != 'object') return false;
	    return deepEq(a, b, aStack, bStack);
	  };
	
	  // Internal recursive comparison function for `isEqual`.
	  deepEq = function(a, b, aStack, bStack) {
	    // Unwrap any wrapped objects.
	    if (a instanceof _) a = a._wrapped;
	    if (b instanceof _) b = b._wrapped;
	    // Compare `[[Class]]` names.
	    var className = toString.call(a);
	    if (className !== toString.call(b)) return false;
	    switch (className) {
	      // Strings, numbers, regular expressions, dates, and booleans are compared by value.
	      case '[object RegExp]':
	      // RegExps are coerced to strings for comparison (Note: '' + /a/i === '/a/i')
	      case '[object String]':
	        // Primitives and their corresponding object wrappers are equivalent; thus, `"5"` is
	        // equivalent to `new String("5")`.
	        return '' + a === '' + b;
	      case '[object Number]':
	        // `NaN`s are equivalent, but non-reflexive.
	        // Object(NaN) is equivalent to NaN.
	        if (+a !== +a) return +b !== +b;
	        // An `egal` comparison is performed for other numeric values.
	        return +a === 0 ? 1 / +a === 1 / b : +a === +b;
	      case '[object Date]':
	      case '[object Boolean]':
	        // Coerce dates and booleans to numeric primitive values. Dates are compared by their
	        // millisecond representations. Note that invalid dates with millisecond representations
	        // of `NaN` are not equivalent.
	        return +a === +b;
	      case '[object Symbol]':
	        return SymbolProto.valueOf.call(a) === SymbolProto.valueOf.call(b);
	    }
	
	    var areArrays = className === '[object Array]';
	    if (!areArrays) {
	      if (typeof a != 'object' || typeof b != 'object') return false;
	
	      // Objects with different constructors are not equivalent, but `Object`s or `Array`s
	      // from different frames are.
	      var aCtor = a.constructor, bCtor = b.constructor;
	      if (aCtor !== bCtor && !(_.isFunction(aCtor) && aCtor instanceof aCtor &&
	                               _.isFunction(bCtor) && bCtor instanceof bCtor)
	                          && ('constructor' in a && 'constructor' in b)) {
	        return false;
	      }
	    }
	    // Assume equality for cyclic structures. The algorithm for detecting cyclic
	    // structures is adapted from ES 5.1 section 15.12.3, abstract operation `JO`.
	
	    // Initializing stack of traversed objects.
	    // It's done here since we only need them for objects and arrays comparison.
	    aStack = aStack || [];
	    bStack = bStack || [];
	    var length = aStack.length;
	    while (length--) {
	      // Linear search. Performance is inversely proportional to the number of
	      // unique nested structures.
	      if (aStack[length] === a) return bStack[length] === b;
	    }
	
	    // Add the first object to the stack of traversed objects.
	    aStack.push(a);
	    bStack.push(b);
	
	    // Recursively compare objects and arrays.
	    if (areArrays) {
	      // Compare array lengths to determine if a deep comparison is necessary.
	      length = a.length;
	      if (length !== b.length) return false;
	      // Deep compare the contents, ignoring non-numeric properties.
	      while (length--) {
	        if (!eq(a[length], b[length], aStack, bStack)) return false;
	      }
	    } else {
	      // Deep compare objects.
	      var keys = _.keys(a), key;
	      length = keys.length;
	      // Ensure that both objects contain the same number of properties before comparing deep equality.
	      if (_.keys(b).length !== length) return false;
	      while (length--) {
	        // Deep compare each member
	        key = keys[length];
	        if (!(has(b, key) && eq(a[key], b[key], aStack, bStack))) return false;
	      }
	    }
	    // Remove the first object from the stack of traversed objects.
	    aStack.pop();
	    bStack.pop();
	    return true;
	  };
	
	  // Perform a deep comparison to check if two objects are equal.
	  _.isEqual = function(a, b) {
	    return eq(a, b);
	  };
	
	  // Is a given array, string, or object empty?
	  // An "empty" object has no enumerable own-properties.
	  _.isEmpty = function(obj) {
	    if (obj == null) return true;
	    if (isArrayLike(obj) && (_.isArray(obj) || _.isString(obj) || _.isArguments(obj))) return obj.length === 0;
	    return _.keys(obj).length === 0;
	  };
	
	  // Is a given value a DOM element?
	  _.isElement = function(obj) {
	    return !!(obj && obj.nodeType === 1);
	  };
	
	  // Is a given value an array?
	  // Delegates to ECMA5's native Array.isArray
	  _.isArray = nativeIsArray || function(obj) {
	    return toString.call(obj) === '[object Array]';
	  };
	
	  // Is a given variable an object?
	  _.isObject = function(obj) {
	    var type = typeof obj;
	    return type === 'function' || type === 'object' && !!obj;
	  };
	
	  // Add some isType methods: isArguments, isFunction, isString, isNumber, isDate, isRegExp, isError, isMap, isWeakMap, isSet, isWeakSet.
	  _.each(['Arguments', 'Function', 'String', 'Number', 'Date', 'RegExp', 'Error', 'Symbol', 'Map', 'WeakMap', 'Set', 'WeakSet'], function(name) {
	    _['is' + name] = function(obj) {
	      return toString.call(obj) === '[object ' + name + ']';
	    };
	  });
	
	  // Define a fallback version of the method in browsers (ahem, IE < 9), where
	  // there isn't any inspectable "Arguments" type.
	  if (!_.isArguments(arguments)) {
	    _.isArguments = function(obj) {
	      return has(obj, 'callee');
	    };
	  }
	
	  // Optimize `isFunction` if appropriate. Work around some typeof bugs in old v8,
	  // IE 11 (#1621), Safari 8 (#1929), and PhantomJS (#2236).
	  var nodelist = root.document && root.document.childNodes;
	  if (typeof /./ != 'function' && typeof Int8Array != 'object' && typeof nodelist != 'function') {
	    _.isFunction = function(obj) {
	      return typeof obj == 'function' || false;
	    };
	  }
	
	  // Is a given object a finite number?
	  _.isFinite = function(obj) {
	    return !_.isSymbol(obj) && isFinite(obj) && !isNaN(parseFloat(obj));
	  };
	
	  // Is the given value `NaN`?
	  _.isNaN = function(obj) {
	    return _.isNumber(obj) && isNaN(obj);
	  };
	
	  // Is a given value a boolean?
	  _.isBoolean = function(obj) {
	    return obj === true || obj === false || toString.call(obj) === '[object Boolean]';
	  };
	
	  // Is a given value equal to null?
	  _.isNull = function(obj) {
	    return obj === null;
	  };
	
	  // Is a given variable undefined?
	  _.isUndefined = function(obj) {
	    return obj === void 0;
	  };
	
	  // Shortcut function for checking if an object has a given property directly
	  // on itself (in other words, not on a prototype).
	  _.has = function(obj, path) {
	    if (!_.isArray(path)) {
	      return has(obj, path);
	    }
	    var length = path.length;
	    for (var i = 0; i < length; i++) {
	      var key = path[i];
	      if (obj == null || !hasOwnProperty.call(obj, key)) {
	        return false;
	      }
	      obj = obj[key];
	    }
	    return !!length;
	  };
	
	  // Utility Functions
	  // -----------------
	
	  // Run Underscore.js in *noConflict* mode, returning the `_` variable to its
	  // previous owner. Returns a reference to the Underscore object.
	  _.noConflict = function() {
	    root._ = previousUnderscore;
	    return this;
	  };
	
	  // Keep the identity function around for default iteratees.
	  _.identity = function(value) {
	    return value;
	  };
	
	  // Predicate-generating functions. Often useful outside of Underscore.
	  _.constant = function(value) {
	    return function() {
	      return value;
	    };
	  };
	
	  _.noop = function(){};
	
	  // Creates a function that, when passed an object, will traverse that object’s
	  // properties down the given `path`, specified as an array of keys or indexes.
	  _.property = function(path) {
	    if (!_.isArray(path)) {
	      return shallowProperty(path);
	    }
	    return function(obj) {
	      return deepGet(obj, path);
	    };
	  };
	
	  // Generates a function for a given object that returns a given property.
	  _.propertyOf = function(obj) {
	    if (obj == null) {
	      return function(){};
	    }
	    return function(path) {
	      return !_.isArray(path) ? obj[path] : deepGet(obj, path);
	    };
	  };
	
	  // Returns a predicate for checking whether an object has a given set of
	  // `key:value` pairs.
	  _.matcher = _.matches = function(attrs) {
	    attrs = _.extendOwn({}, attrs);
	    return function(obj) {
	      return _.isMatch(obj, attrs);
	    };
	  };
	
	  // Run a function **n** times.
	  _.times = function(n, iteratee, context) {
	    var accum = Array(Math.max(0, n));
	    iteratee = optimizeCb(iteratee, context, 1);
	    for (var i = 0; i < n; i++) accum[i] = iteratee(i);
	    return accum;
	  };
	
	  // Return a random integer between min and max (inclusive).
	  _.random = function(min, max) {
	    if (max == null) {
	      max = min;
	      min = 0;
	    }
	    return min + Math.floor(Math.random() * (max - min + 1));
	  };
	
	  // A (possibly faster) way to get the current timestamp as an integer.
	  _.now = Date.now || function() {
	    return new Date().getTime();
	  };
	
	  // List of HTML entities for escaping.
	  var escapeMap = {
	    '&': '&amp;',
	    '<': '&lt;',
	    '>': '&gt;',
	    '"': '&quot;',
	    "'": '&#x27;',
	    '`': '&#x60;'
	  };
	  var unescapeMap = _.invert(escapeMap);
	
	  // Functions for escaping and unescaping strings to/from HTML interpolation.
	  var createEscaper = function(map) {
	    var escaper = function(match) {
	      return map[match];
	    };
	    // Regexes for identifying a key that needs to be escaped.
	    var source = '(?:' + _.keys(map).join('|') + ')';
	    var testRegexp = RegExp(source);
	    var replaceRegexp = RegExp(source, 'g');
	    return function(string) {
	      string = string == null ? '' : '' + string;
	      return testRegexp.test(string) ? string.replace(replaceRegexp, escaper) : string;
	    };
	  };
	  _.escape = createEscaper(escapeMap);
	  _.unescape = createEscaper(unescapeMap);
	
	  // Traverses the children of `obj` along `path`. If a child is a function, it
	  // is invoked with its parent as context. Returns the value of the final
	  // child, or `fallback` if any child is undefined.
	  _.result = function(obj, path, fallback) {
	    if (!_.isArray(path)) path = [path];
	    var length = path.length;
	    if (!length) {
	      return _.isFunction(fallback) ? fallback.call(obj) : fallback;
	    }
	    for (var i = 0; i < length; i++) {
	      var prop = obj == null ? void 0 : obj[path[i]];
	      if (prop === void 0) {
	        prop = fallback;
	        i = length; // Ensure we don't continue iterating.
	      }
	      obj = _.isFunction(prop) ? prop.call(obj) : prop;
	    }
	    return obj;
	  };
	
	  // Generate a unique integer id (unique within the entire client session).
	  // Useful for temporary DOM ids.
	  var idCounter = 0;
	  _.uniqueId = function(prefix) {
	    var id = ++idCounter + '';
	    return prefix ? prefix + id : id;
	  };
	
	  // By default, Underscore uses ERB-style template delimiters, change the
	  // following template settings to use alternative delimiters.
	  _.templateSettings = {
	    evaluate: /<%([\s\S]+?)%>/g,
	    interpolate: /<%=([\s\S]+?)%>/g,
	    escape: /<%-([\s\S]+?)%>/g
	  };
	
	  // When customizing `templateSettings`, if you don't want to define an
	  // interpolation, evaluation or escaping regex, we need one that is
	  // guaranteed not to match.
	  var noMatch = /(.)^/;
	
	  // Certain characters need to be escaped so that they can be put into a
	  // string literal.
	  var escapes = {
	    "'": "'",
	    '\\': '\\',
	    '\r': 'r',
	    '\n': 'n',
	    '\u2028': 'u2028',
	    '\u2029': 'u2029'
	  };
	
	  var escapeRegExp = /\\|'|\r|\n|\u2028|\u2029/g;
	
	  var escapeChar = function(match) {
	    return '\\' + escapes[match];
	  };
	
	  // JavaScript micro-templating, similar to John Resig's implementation.
	  // Underscore templating handles arbitrary delimiters, preserves whitespace,
	  // and correctly escapes quotes within interpolated code.
	  // NB: `oldSettings` only exists for backwards compatibility.
	  _.template = function(text, settings, oldSettings) {
	    if (!settings && oldSettings) settings = oldSettings;
	    settings = _.defaults({}, settings, _.templateSettings);
	
	    // Combine delimiters into one regular expression via alternation.
	    var matcher = RegExp([
	      (settings.escape || noMatch).source,
	      (settings.interpolate || noMatch).source,
	      (settings.evaluate || noMatch).source
	    ].join('|') + '|$', 'g');
	
	    // Compile the template source, escaping string literals appropriately.
	    var index = 0;
	    var source = "__p+='";
	    text.replace(matcher, function(match, escape, interpolate, evaluate, offset) {
	      source += text.slice(index, offset).replace(escapeRegExp, escapeChar);
	      index = offset + match.length;
	
	      if (escape) {
	        source += "'+\n((__t=(" + escape + "))==null?'':_.escape(__t))+\n'";
	      } else if (interpolate) {
	        source += "'+\n((__t=(" + interpolate + "))==null?'':__t)+\n'";
	      } else if (evaluate) {
	        source += "';\n" + evaluate + "\n__p+='";
	      }
	
	      // Adobe VMs need the match returned to produce the correct offset.
	      return match;
	    });
	    source += "';\n";
	
	    // If a variable is not specified, place data values in local scope.
	    if (!settings.variable) source = 'with(obj||{}){\n' + source + '}\n';
	
	    source = "var __t,__p='',__j=Array.prototype.join," +
	      "print=function(){__p+=__j.call(arguments,'');};\n" +
	      source + 'return __p;\n';
	
	    var render;
	    try {
	      render = new Function(settings.variable || 'obj', '_', source);
	    } catch (e) {
	      e.source = source;
	      throw e;
	    }
	
	    var template = function(data) {
	      return render.call(this, data, _);
	    };
	
	    // Provide the compiled source as a convenience for precompilation.
	    var argument = settings.variable || 'obj';
	    template.source = 'function(' + argument + '){\n' + source + '}';
	
	    return template;
	  };
	
	  // Add a "chain" function. Start chaining a wrapped Underscore object.
	  _.chain = function(obj) {
	    var instance = _(obj);
	    instance._chain = true;
	    return instance;
	  };
	
	  // OOP
	  // ---------------
	  // If Underscore is called as a function, it returns a wrapped object that
	  // can be used OO-style. This wrapper holds altered versions of all the
	  // underscore functions. Wrapped objects may be chained.
	
	  // Helper function to continue chaining intermediate results.
	  var chainResult = function(instance, obj) {
	    return instance._chain ? _(obj).chain() : obj;
	  };
	
	  // Add your own custom functions to the Underscore object.
	  _.mixin = function(obj) {
	    _.each(_.functions(obj), function(name) {
	      var func = _[name] = obj[name];
	      _.prototype[name] = function() {
	        var args = [this._wrapped];
	        push.apply(args, arguments);
	        return chainResult(this, func.apply(_, args));
	      };
	    });
	    return _;
	  };
	
	  // Add all of the Underscore functions to the wrapper object.
	  _.mixin(_);
	
	  // Add all mutator Array functions to the wrapper.
	  _.each(['pop', 'push', 'reverse', 'shift', 'sort', 'splice', 'unshift'], function(name) {
	    var method = ArrayProto[name];
	    _.prototype[name] = function() {
	      var obj = this._wrapped;
	      method.apply(obj, arguments);
	      if ((name === 'shift' || name === 'splice') && obj.length === 0) delete obj[0];
	      return chainResult(this, obj);
	    };
	  });
	
	  // Add all accessor Array functions to the wrapper.
	  _.each(['concat', 'join', 'slice'], function(name) {
	    var method = ArrayProto[name];
	    _.prototype[name] = function() {
	      return chainResult(this, method.apply(this._wrapped, arguments));
	    };
	  });
	
	  // Extracts the result from a wrapped and chained object.
	  _.prototype.value = function() {
	    return this._wrapped;
	  };
	
	  // Provide unwrapping proxy for some methods used in engine operations
	  // such as arithmetic and JSON stringification.
	  _.prototype.valueOf = _.prototype.toJSON = _.prototype.value;
	
	  _.prototype.toString = function() {
	    return String(this._wrapped);
	  };
	
	  // AMD registration happens at the end for compatibility with AMD loaders
	  // that may not enforce next-turn semantics on modules. Even though general
	  // practice for AMD registration is to be anonymous, underscore registers
	  // as a named module because, like jQuery, it is a base library that is
	  // popular enough to be bundled in a third party lib, but not be part of
	  // an AMD load request. Those cases could generate an error when an
	  // anonymous define() is called outside of a loader request.
	  if (true) {
	    !(__WEBPACK_AMD_DEFINE_ARRAY__ = [], __WEBPACK_AMD_DEFINE_RESULT__ = function() {
	      return _;
	    }.apply(exports, __WEBPACK_AMD_DEFINE_ARRAY__), __WEBPACK_AMD_DEFINE_RESULT__ !== undefined && (module.exports = __WEBPACK_AMD_DEFINE_RESULT__));
	  }
	}());
	
	/* WEBPACK VAR INJECTION */}.call(exports, (function() { return this; }()), __webpack_require__(4)(module)))

/***/ }),
/* 4 */
/***/ (function(module, exports) {

	module.exports = function(module) {
		if(!module.webpackPolyfill) {
			module.deprecate = function() {};
			module.paths = [];
			// module.parent = undefined by default
			module.children = [];
			module.webpackPolyfill = 1;
		}
		return module;
	}


/***/ }),
/* 5 */
/***/ (function(module, exports) {

	module.exports = {"name":"uqtools","version":"0.0.1","description":"UQTools Plotting Widget","author":"Markus Jerger","main":"src/index.js","keywords":["jupyter","widgets","ipython","ipywidgets"],"scripts":{"prepublish":"webpack","test":"echo \"Error: no test specified\" && exit 1"},"devDependencies":{"json-loader":"^0.5.4","webpack":"^1.12.14"},"dependencies":{"jupyter-js-widgets":"^2.1.4","underscore":"^1.8.3"}}

/***/ })
/******/ ])});;
//# sourceMappingURL=index.js.map