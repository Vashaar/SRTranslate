# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules

provider_modules = collect_submodules("translator.providers")

a = Analysis(
    ['desktop_app.py'],
    pathex=[],
    binaries=[],
    datas=[('config.yaml', '.'), ('glossaries', 'glossaries'), ('assets', 'assets')],
    hiddenimports=provider_modules + ['pypdf'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['streamlit'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SRTranslate Desktop',
    icon='assets/app_icon.ico',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SRTranslate Desktop',
)
