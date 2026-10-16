# Map tile reuse on localhost and Vercel

All Leaflet maps use `cachedTileLayer`: the staff map, field map, seedling location
picker, monitoring reference map and public LIKE viewer. It retains detached
image elements by tile URL and request settings. Returning to the same zoom,
panning back or toggling a layer reattaches the existing image without assigning
its source again. Unfinished downloads continue through zoom changes and can be
reattached. Failed downloads and fallback images are not cached as successful
tiles. The same retained image is never lent to two visible maps at once.

The shared pool holds at most 512 off-screen tiles for up to one hour; visible
tiles remain managed by Leaflet. This bounds decoded-image memory. New tiles,
expired entries and entries evicted after extensive exploration can download.
Zooming beyond a layer's native resolution reuses its native tiles. Cached images
skip the loading fade; first-time imagery retains normal loading behavior.

Both FastAPI tile paths (exported images and GeoTIFF rendering) and the localhost
Vite tile middleware return `Cache-Control: public, max-age=3600`. Localhost also
returns ETag/Last-Modified and supports conditional requests. Vercel's tile rewrite
respects origin cache headers. Private API caching rules are unchanged.

When replacing imagery at the same tileset path, set a new public build-time
`VITE_TILESET_REVISION` value and rebuild/redeploy the frontend. It becomes the
tile URL's `v` parameter, so old imagery is not reused for the new revision.
Changing `VITE_TILESET_PATH` also changes the URL and cache key. Without a revision
change, already downloaded imagery can stay fresh for one hour.

Regression checks run real Leaflet zoom/pan/layer-toggle sequences in JSDOM with
instrumented image requests, including rapid zoom before downloads finish,
failed-download retries, native overzoom, eviction and revision changes. Local
HTTP and FastAPI checks verify cache headers and conditional responses. These
are automated checks; hosted browser-network verification still requires the
matching frontend and backend to be running.
