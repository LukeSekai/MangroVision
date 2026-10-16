import L from 'leaflet';

// A detached image keeps both its decoded pixels and any in-flight request.
// Reusing the node avoids assigning src again when Leaflet revisits a zoom.
// Bound the cache so exploring many areas cannot retain unlimited image memory.
export class TileImagePool {
  constructor({ maxTiles = 512, maxAge = 3600_000, now = Date.now } = {}) {
    this.maxTiles = maxTiles;
    this.maxAge = maxAge;
    this.now = now;
    this.images = new Map();
    this.order = new Map();
  }

  remove(tile, key) {
    const group = this.images.get(key);
    group?.delete(tile);
    if (!group?.size) this.images.delete(key);
    this.order.delete(tile);
  }

  take(key) {
    for (const tile of this.images.get(key) || []) {
      this.remove(tile, key);
      if (this.now() - tile.mvCreatedAt >= this.maxAge || tile.mvFailed
          || (tile.complete && !tile.naturalWidth)) continue;
      return tile;
    }
    return null;
  }

  release(key, tile) {
    if (tile.mvFailed || (tile.complete && !tile.naturalWidth)
        || this.now() - tile.mvCreatedAt >= this.maxAge) return;
    if (!this.images.has(key)) this.images.set(key, new Set());
    this.images.get(key).add(tile);
    this.order.delete(tile);
    this.order.set(tile, key);
    while (this.order.size > this.maxTiles) {
      const [oldest, oldKey] = this.order.entries().next().value;
      this.remove(oldest, oldKey);
    }
  }
}

const sharedImages = new TileImagePool();

// Leaflet 1.9 discards the img src when pruning a tile and aborts requests on
// zoom. Preserve those images instead; GridLayer still manages visible levels.
export const CachedTileLayer = L.TileLayer.extend({
  options: { keepBuffer: 4, updateWhenIdle: true, updateWhenZooming: false },

  createTile(coords, done) {
    const url = this.getTileUrl(coords);
    const key = JSON.stringify([url, this.options.crossOrigin, this.options.referrerPolicy]);
    const pool = this.options.imagePool || sharedImages;
    const cached = pool.take(key);
    const tile = cached || document.createElement('img');
    const lease = {};
    tile.mvLease = lease;
    tile.mvCacheKey = key;
    tile.mvOriginalUrl = url;
    tile.mvReused = Boolean(cached && tile.complete && tile.naturalWidth);
    tile.mvFailed = false;
    tile.classList.remove('leaflet-tile-loaded');
    tile.style.opacity = '';
    const ready = (error) => {
      if (tile.mvLease !== lease) return;
      tile.mvFailed = Boolean(error);
      done(error, tile);
    };
    L.DomEvent.on(tile, 'load', () => ready(null));
    L.DomEvent.on(tile, 'error', (error) => {
      ready(error);
      // A fallback image is display-only; never cache it as a successful tile.
      if (this.options.errorTileUrl && tile.getAttribute('src') !== this.options.errorTileUrl) {
        L.DomEvent.off(tile, 'load');
        tile.src = this.options.errorTileUrl;
      }
    });
    if (!cached) {
      if (this.options.crossOrigin || this.options.crossOrigin === '') {
        tile.crossOrigin = this.options.crossOrigin === true ? '' : this.options.crossOrigin;
      }
      if (typeof this.options.referrerPolicy === 'string') tile.referrerPolicy = this.options.referrerPolicy;
      tile.alt = '';
      tile.mvCreatedAt = pool.now();
      tile.src = url;
    } else if (tile.mvReused) {
      // Let GridLayer register the tile before delivering the cached result.
      queueMicrotask(() => ready(null));
    }
    return tile;
  },

  _abortLoading() {
    // Pending images move into the pool when pruned and finish downloading.
    // Zooming back can reattach the same request instead of restarting it.
  },

  _removeTile(key) {
    const tile = this._tiles[key]?.el;
    if (!tile) return;
    tile.mvLease = null;
    L.DomEvent.off(tile, 'load');
    L.DomEvent.off(tile, 'error');
    L.GridLayer.prototype._removeTile.call(this, key);
    if (tile.getAttribute('src') === tile.mvOriginalUrl) {
      (this.options.imagePool || sharedImages).release(tile.mvCacheKey, tile);
    }
  },

  _tileReady(coords, error, tile) {
    L.TileLayer.prototype._tileReady.call(this, coords, error, tile);
    const record = this._tiles?.[this._tileCoordsToKey(coords)];
    if (!error && tile.mvReused && record?.el === tile) {
      // Cached pixels need no loading fade, including with reduced motion off.
      record.loaded = Date.now() - 250;
      record.active = true;
      L.DomUtil.setOpacity(tile, 1);
    }
  },
});

export function cachedTileLayer(url, options) {
  return new CachedTileLayer(url, options);
}
