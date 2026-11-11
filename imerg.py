"""La lluvia medida por los satélites: IMERG de la NASA (GPM).

IMERG junta lo que miden los satélites de microondas e infrarrojos (y
lo corrige con pluviómetros) y da la lluvia de cada día en todo el mundo
a 0,1°. Es de la NASA: uso libre, también comercial. Para bajarlo hace
falta una cuenta gratuita de Earthdata: su usuario y su contraseña van
en los secretos EARTHDATA_USER y EARTHDATA_PASS del repositorio, y el
flujo los deja en ~/.netrc.

Se usa la versión «Late» de cada día (sale uno o dos días después); los
días que aún no la tienen siguen con la lluvia del modelo GFS.
"""
import os
import re
import subprocess
import tempfile

import numpy as np

URL = 'https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDL.07/'


def ready():
    """¿Hay cuenta de Earthdata (en ~/.netrc)?"""
    p = os.path.expanduser('~/.netrc')
    return os.path.exists(p) and 'urs.earthdata.nasa.gov' in open(p).read()


def curl(url, out=None):
    # La NASA manda a entrar en urs.earthdata.nasa.gov y vuelve con una
    # galleta: curl lo hace solo con -n (la cuenta de ~/.netrc), -L y las
    # galletas en un fichero.
    jar = os.path.join(tempfile.gettempdir(), 'earthdata-galletas')
    cmd = ['curl', '-sSfL', '-n', '-c', jar, '-b', jar, '--retry', '2',
           '-m', '300', url]
    if out:
        cmd += ['-o', out]
    return subprocess.run(cmd, check=True, capture_output=True).stdout


_lists = {}


def file_for(day):
    """El nombre del fichero de ese día (AAAAMMDD), o None si aún no está."""
    folder = URL + day[:4] + '/' + day[4:6] + '/'
    if folder not in _lists:
        try:
            _lists[folder] = curl(folder).decode('utf-8', 'replace')
        except subprocess.CalledProcessError:
            _lists[folder] = ''
    names = sorted(set(re.findall(
        r'3B-DAY-L\.MS\.MRG\.3IMERG\.%s-S000000-E235959\.V\w+\.nc4' % day,
        _lists[folder])))
    return folder + names[-1] if names else None


def day_rain(day):
    """La lluvia de ese día (mm) en la rejilla de 720 × 361; NaN donde
    no hay dato. None si el día aún no está."""
    url = file_for(day)
    if not url:
        return None
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'dia.nc4')
        curl(url, path)
        import h5py
        with h5py.File(path, 'r') as f:
            v = f['precipitation']
            a = np.asarray(v[0] if v.ndim == 3 else v, dtype=np.float32)
    return to_grid(a)


def to_grid(a):
    """De IMERG (lon × lat: 3600 × 1800, de -179,95 a 179,95 y de -89,95
    a 89,95) a nuestra rejilla: filas de 90 a -90 y columnas de 0 a 359,5,
    cada 0,5°. Cada punto es la media de las celdas de 0,1° que caen a
    menos de 0,25° (6 × 6), sin contar las que no tienen dato."""
    if a.shape == (3600, 1800):
        a = a.T
    a = np.where(a >= 0, a, np.nan)[::-1]           # de norte a sur
    a = np.roll(a, -1800, axis=1)                   # desde la longitud 0
    # Margen: 3 filas arriba y abajo (sin dato) y la vuelta al mundo por
    # los lados (3 a la izquierda, 2 a la derecha).
    a = np.concatenate([a[:, -3:], a, a[:, :2]], axis=1)
    pad = np.full((3, a.shape[1]), np.nan, np.float32)
    a = np.concatenate([pad, a, pad], axis=0)
    total = np.zeros((361, 720), np.float32)
    count = np.zeros((361, 720), np.float32)
    for i in range(6):
        for j in range(6):
            s = a[i::5, j::5][:361, :720]
            ok = ~np.isnan(s)
            total += np.where(ok, s, 0)
            count += ok
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(count > 0, total / count, np.nan)
