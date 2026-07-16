import nox

nox.options.sessions = ['lint', 'test', 'integration']
nox.options.reuse_existing_virtualenvs = True


@nox.session
def lint(session):
    """Run ruff linting and format check."""
    session.install('ruff')
    session.run('ruff', 'check', '.')
    session.run('ruff', 'format', '--check', '.')


@nox.session
def fix(session):
    """Auto-fix lint errors and format code."""
    session.install('ruff')
    session.run('ruff', 'check', '--fix', '--unsafe-fixes', '.')
    session.run('ruff', 'format', '.')


@nox.session
def typecheck(session):
    """Run mypy type checking on core modules."""
    session.install(
        'mypy',
        'pandas',
        'numpy',
        'yfinance',
    )
    session.run(
        'mypy',
        'core/',
        'execution/',
        'data_fetch/',
        'utils/',
        '--ignore-missing-imports',
        '--no-strict-optional',
        '--allow-untyped-defs',
    )


@nox.session(python=['3.11', '3.12', '3.13', '3.14'])
def test(session):
    """Run pytest unit tests across Python 3.11+."""
    session.install('-r', 'requirements.txt')
    session.run(
        'pytest',
        'tests/unit/',
        '-v',
        '--tb=short',
        *session.posargs,
    )


@nox.session(python=['3.11', '3.12', '3.13', '3.14'])
def integration(session):
    """Run integration tests across Python 3.11+."""
    session.install('-r', 'requirements.txt')
    session.run(
        'pytest',
        'tests/integration/',
        '-v',
        '--tb=short',
        *session.posargs,
    )
