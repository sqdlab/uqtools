//require(["widgets/js/widget"], function(WidgetManager) {
    function Axes(view, ax) {
        // copy attributes: index, limits, navigable, zoomable
        $.extend(this, ax);
        // create in DOM
        this.$el = $('<div />')
            .addClass('Axes')
            .css('position', 'absolute')
            .css('left', this.x_min)
            .css('top', this.y_max)
            .css('width', this.x_max-this.x_min)
            .css('height', this.y_min-this.y_max)
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
        
        var _transform = function(x, y, x_min, x_max, y_min, y_max, u_min, u_max, v_min, v_max) {
            /**
             * transform coordinates between two linear, rectangular coordinate systems
             */
            var u_scale = (u_max - u_min) / (x_max - x_min),
                v_scale = (v_max - v_min) / (y_max - y_min);
            var u = u_min + (x - x_min) * u_scale,
                v = v_min + (y - y_min) * v_scale;
            return [u, v];
        }
            
        this.transform = function(u, v) {
            /**
             * tranform user coordinates into pixel coordinates relative to 
             * the top left corner of ax
             */
            var xy = _transform(u, v,
                                this.u_min, this.u_max, this.v_min, this.v_max,
                                this.x_min, this.x_max, this.y_min, this.y_max);

            return [xy[0] - this.x_min, xy[1] - this.y_max];
        }
        
        this.transformInverse = function(x, y) {
            /**
             * transform pixel x, y relative to the top left corner of
             * ax into user coordinates
             */
            return _transform(this.x_min + x, this.y_max + y, 
                              this.x_min, this.x_max, this.y_min, this.y_max,
                              this.u_min, this.u_max, this.v_min, this.v_max);
        }
    }
    
    
    var FigureView = IPython.DOMWidgetView.extend({
        render: function() {
            /**
             * Create plot image
             */
            //console.log('ConsoleView render');
            this.$el.addClass('Figure');
            // force position
            this.$el.css('position', 'relative')
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
            // install event handlers
            var data = $.extend({}, event.data, {canvas: canvas, zoom: zoom});
            $(document).on('mousemove', data, view.zoomMove);
            $(document).one('mouseup', data, view.zoomFinish);
            $(document).one('keydown', data, view.zoomAbort);
        },

        zoomMove: function(event) {
            /**
             * Resize zoom window according to mouse position
             */
            var ax = event.data.ax;
            var zoom = event.data.zoom;
            var canvas = event.data.canvas;
            zoom.end = ax.mouseXY(event);
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
            ctx.clearRect(x, y, w, h);
            ctx.strokeRect(x, y, w, h);
        },
        
        zoomAbort: function(event) {
            //console.log('zoomAbort');
            var view = event.data.view;
            // disable event handlers
            event.stopPropagation();
            $(document).off('mousemove', view.zoomMove);
            $(document).off('mouseup', view.zoomFinish);
            $(document).off('keydown', view.zoomAbort);
            // remove zoom box
            event.data.canvas.remove();
            
        },

        zoomFinish: function(event) {
            //console.log('zoomFinish');
            var view = event.data.view,
                ax = event.data.ax,
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
                (v != null) && ((v < attr.ax.v_min) || (v > attr.ax.v_max)) ||
                !attr.template && (u == null) && (v == null)) {
                return '!';
            }
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
            this.$el.addClass('Cursor');
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
            if (direction & 2) {
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
                x_invalid = (x < 0) || (x > ax.x_max - ax.x_min),
                y_invalid = (y < 0) || (y > ax.y_min - ax.y_max);
            x = Math.max(0, Math.min(ax.x_max-ax.x_min, x));
            y = Math.max(0, Math.min(ax.y_min-ax.y_max, y));
            
            if (direction & 1) {
                this.position[1] = uv[1];
                this.$el.children('.hRuler').css('top', y - 1);
                this.$el.children('.iRuler').css('top', y - 2);
                var hlabel = this.$el.children('.hLabel');
                if (y_invalid) {
                    hlabel.text('');
                } else {
                    var v_pixels = Math.abs(ax.y_max - ax.y_min),
                        v_digits = Math.ceil(Math.LOG10E * Math.log(v_pixels));
                    hlabel.text(uv[1].toPrecision(v_digits));
                    hlabel.css('top', y - hlabel.height()/2);
                }
            }
            if (direction & 2) {
                this.position[0] = uv[0];
                this.$el.children('.vRuler').css('left', x - 1);
                this.$el.children('.iRuler').css('left', x - 2);
                var vlabel = this.$el.children('.vLabel');
                if (x_invalid) {
                    vlabel.text('');
                } else {
                    var u_pixels = Math.abs(ax.x_max-ax.x_min),
                        u_digits = Math.ceil(Math.LOG10E * Math.log(u_pixels));
                    vlabel.text(uv[0].toPrecision(u_digits));
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
            view.render();
            this.$el.append(view.el);
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
                        if (cursor.isValid()) {
                            cursors.add(cursor);
                        } else {
                            cursor.destroy();
                        }
                    });
                }
                // create a view for the cursors
                var view = new CursorCollectionView({model: cursors});
                view.render();
                this.axes[idx].$el.append(view.el);
                // fire change if number of cursors changed (due to validation)
                if ((nb_cursors != undefined) && (nb_cursors.length != cursors.length)) {
                    this.cursorChange({collection: cursors});
                }
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
        
    
    // Register the views with the widget manager.
    IPython.WidgetManager.register_widget_view('FigureView', FigureView);
    IPython.WidgetManager.register_widget_view('ZoomFigureView', ZoomFigureView);
    IPython.WidgetManager.register_widget_view('ZoomCursorFigureView', ZoomCursorFigureView);
//});