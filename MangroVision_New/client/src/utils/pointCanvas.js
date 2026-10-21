import L from 'leaflet';

const PointCanvas = L.Canvas.extend({
  _redraw() {
    // fitBounds can redraw synchronously while an older frame is queued.
    // Cancel it before Leaflet clears its handle, and ignore removed canvases.
    L.Util.cancelAnimFrame(this._redrawRequest);
    if (this._map && this._ctx) L.Canvas.prototype._redraw.call(this);
  },
});

export const createPointCanvas = (options) => new PointCanvas(options);
