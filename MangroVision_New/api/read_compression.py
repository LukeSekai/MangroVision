"""Lossless transport compression for large workspace read responses."""

from starlette.middleware.gzip import GZipMiddleware


class WorkspaceReadCompression:
    # Restrict compression to data reads. Login/share secrets, writes, image
    # tiles, downloads, and processing streams keep their existing transport.
    PATHS = frozenset({
        '/api/planters/map-points',
        '/api/analyses/stats',
        '/api/dashboard/overview', '/api/dashboard/operations',
        '/api/dashboard/ecology', '/api/dashboard/sites',
    })

    def __init__(self, app):
        self.app = app
        self.compressed = GZipMiddleware(app, minimum_size=1024, compresslevel=3)

    async def __call__(self, scope, receive, send):
        compress = (
            scope['type'] == 'http'
            and scope.get('method') == 'GET'
            and scope.get('path') in self.PATHS
        )
        await (self.compressed if compress else self.app)(scope, receive, send)
