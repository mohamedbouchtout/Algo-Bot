import nox

nox.options.sessions = ['lint', 'test']
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


@nox.session
def test(session):
    """Run pytest unit tests."""
    session.install(
        'pytest',
        'pytest-cov',
        'pandas',
        'numpy',
        'yfinance',
        'torch',
    )
    session.run(
        'pytest',
        'tests/',
        '-v',
        '--tb=short',
        '-m',
        'not integration',
        *session.posargs,
    )
