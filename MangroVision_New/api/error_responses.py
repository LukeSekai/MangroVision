"""JSON errors for API clients; diagnostic exceptions stay in backend logs."""
import logging

from fastapi.responses import JSONResponse
from sqlalchemy.exc import OperationalError, TimeoutError as DatabaseTimeout

logger = logging.getLogger(__name__)


def install_error_responses(app):
    async def database_unavailable(request, error):
        logger.warning('Database temporarily unavailable for %s %s: %s',
                       request.method, request.url.path, type(error).__name__)
        return JSONResponse(status_code=503, content={
            'detail': 'The database is busy or temporarily unavailable. Please try again shortly.',
        }, headers={'Cache-Control': 'no-store', 'Retry-After': '3'})

    async def internal_error(request, error):
        logger.error('Request failed: %s %s', request.method, request.url.path,
                     exc_info=(type(error), error, error.__traceback__))
        return JSONResponse(status_code=500, content={
            'detail': 'The server could not complete this request. Check the laptop backend log and try again.',
        }, headers={'Cache-Control': 'no-store'})

    app.add_exception_handler(DatabaseTimeout, database_unavailable)
    app.add_exception_handler(OperationalError, database_unavailable)
    app.add_exception_handler(Exception, internal_error)
