import test from 'node:test';
import assert from 'node:assert/strict';
import { connectMapLayerVisibility } from './mapLayerVisibility.js';

test('temporary layer filters preserve user preferences and detach their listeners', () => {
  const layers = { points: {}, forbidden: {} };
  const visible = new Set(Object.values(layers));
  let listener;
  const map = {
    on(events, callback) { listener = callback; },
    off(events, callback) { assert.equal(callback, listener); listener = null; },
    hasLayer(layer) { return visible.has(layer); },
    addLayer(layer) { visible.add(layer); listener?.({ type: 'overlayadd', layer }); },
    removeLayer(layer) { visible.delete(layer); listener?.({ type: 'overlayremove', layer }); },
  };
  const state = { layerVisibility: { points: true, forbidden: true },
    toggleLayer(key) { this.layerVisibility = { ...this.layerVisibility, [key]: !this.layerVisibility[key] }; } };
  const bridge = connectMapLayerVisibility(map, layers, { getState: () => state });
  bridge.sync({ ...state.layerVisibility, forbidden: false });
  assert.equal(map.hasLayer(layers.forbidden), false);
  assert.equal(state.layerVisibility.forbidden, true);
  map.removeLayer(layers.points);
  assert.equal(state.layerVisibility.points, false);
  bridge.sync(state.layerVisibility);
  assert.equal(map.hasLayer(layers.forbidden), true);
  assert.equal(map.hasLayer(layers.points), false);
  bridge.dispose();
  assert.equal(listener, null);
});
