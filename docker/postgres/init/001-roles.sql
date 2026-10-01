DO $bootstrap$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'mangrovision') THEN
        CREATE ROLE mangrovision LOGIN PASSWORD 'mangrovision';
    END IF;
END
$bootstrap$;

DO $bootstrap$
BEGIN
    EXECUTE format(
        'GRANT CONNECT ON DATABASE %I TO mangrovision',
        current_database()
    );
END
$bootstrap$;
