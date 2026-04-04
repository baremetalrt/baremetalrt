# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = ['uvicorn.logging', 'uvicorn.protocols.http']
hiddenimports += collect_submodules('fastapi')
hiddenimports += collect_submodules('uvicorn')
hiddenimports += collect_submodules('starlette')
hiddenimports += collect_submodules('pystray')


a = Analysis(
    ['daemon/baremetalrt.py'],
    pathex=['daemon'],
    binaries=[],
    datas=[
        ('daemon/dashboard.html', '.'),
        ('daemon/model_registry.py', '.'),
        ('daemon/build_engine.py', '.'),
        ('daemon/benchmark.py', '.'),
        ('daemon/daemon.py', '.'),
        ('VERSION', '.'),
    ],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'torch', 'tensorrt', 'tensorrt_llm', 'transformers', 'numpy', 'cuda',
        'pandas', 'scipy', 'numba', 'pyarrow', 'llvmlite', 'lxml',
        'matplotlib', 'tkinter', 'test', 'unittest',
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='baremetalrt',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='installer/assets/icon.ico',
)
