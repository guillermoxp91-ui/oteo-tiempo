"""El tiempo para el mapa de Oteo.

Baja la última previsión pública GFS (NOAA, dominio público) y la deja en
imágenes pequeñas del mundo entero a medio grado, una por hora y capa:

  datos/manifest.json          {dir, t0, n, w, h, res}
  datos/<pasada>/wNN.jpg       viento: R = u, G = v  ->  (x / 2.55 - 50) m/s
  datos/<pasada>/rNN.png       lluvia (gris):   (x / 40)^2 mm/h
  datos/<pasada>/sNN.png       nieve (gris):    (x / 40)^2 mm/h de agua
  datos/<pasada>/cNN.png       energía para tormentas (gris): x * 20 J/kg (CAPE)

NN es la hora desde la pasada (00 a 48). Filas de norte (90°) a sur,
columnas desde la longitud 0 hacia el este.
"""
import datetime as dt
import json
import os
import shutil
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pygrib
from PIL import Image

BASE = 'https://noaa-gfs-bdp-pds.s3.amazonaws.com'
HOURS = 48
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datos')
WANT = {
    'UGRD:10 m above ground': 'u',
    'VGRD:10 m above ground': 'v',
    'PRATE:surface': 'rain',
    'CSNOW:surface': 'snow',
    'CAPE:surface': 'cape',
}


def get(url, rng=None, tries=4):
    for k in range(tries):
        try:
            req = urllib.request.Request(url)
            if rng:
                req.add_header('Range', 'bytes=%d-%s' % rng)
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


def key(c, f):
    return 'gfs.%s/%02d/atmos/gfs.t%02dz.pgrb2.0p25.f%03d' % (
        c.strftime('%Y%m%d'), c.hour, c.hour, f)


def cycle():
    """La última pasada del modelo que ya está entera."""
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    c = now.replace(hour=now.hour // 6 * 6, minute=0, second=0, microsecond=0)
    for _ in range(6):
        try:
            get(BASE + '/' + key(c, HOURS) + '.idx', tries=1)
            return c
        except Exception:
            c -= dt.timedelta(hours=6)
    raise SystemExit('No hay ninguna pasada reciente')


def hour(c, f):
    rows = [l.split(':') for l in
            get(BASE + '/' + key(c, f) + '.idx').decode().strip().split('\n')]
    fields = {}
    for i, r in enumerate(rows):
        name = WANT.get(r[3] + ':' + r[4])
        if not name:
            continue
        # De lluvia y nieve hay dos: la del instante y la media de la hora.
        # Se queda la media (la última que sale).
        end = int(rows[i + 1][1]) - 1 if i + 1 < len(rows) else ''
        fields[name] = pygrib.fromstring(
            get(BASE + '/' + key(c, f), (int(r[1]), end))).values
    return f, fields


def half(a):
    """De 0,25° a 0,5°: la media de cada 2×2."""
    a = np.asarray(a, dtype=np.float32)
    a = np.vstack([a, a[-1:]])  # 721 filas -> 722
    return a.reshape(361, 2, 720, 2).mean(axis=(1, 3))


def byte(a):
    return np.clip(np.rint(a), 0, 255).astype(np.uint8)


def main():
    t0 = time.time()
    c = cycle()
    name = c.strftime('%Y%m%d%H')
    print('pasada', name, flush=True)
    with ThreadPoolExecutor(8) as ex:
        got = list(ex.map(lambda f: hour(c, f), range(HOURS + 1)))
    shutil.rmtree(OUT, ignore_errors=True)
    d = os.path.join(OUT, name)
    os.makedirs(d)
    total = 0
    for f, fl in got:
        u, v, cape = half(fl['u']), half(fl['v']), half(fl['cape'])
        zero = np.zeros_like(u)
        rain = half(fl['rain']) * 3600 if 'rain' in fl else zero  # mm/h
        snow = rain * half(fl['snow']) if 'snow' in fl else zero
        Image.fromarray(np.dstack([byte((u + 50) * 2.55), byte((v + 50) * 2.55),
                                   byte(zero)]), 'RGB').save(
            '%s/w%02d.jpg' % (d, f), quality=90, subsampling=0)
        Image.fromarray(byte(np.sqrt(np.maximum(rain, 0)) * 40)).save(
            '%s/r%02d.png' % (d, f), optimize=True)
        Image.fromarray(byte(np.sqrt(np.maximum(snow, 0)) * 40)).save(
            '%s/s%02d.png' % (d, f), optimize=True)
        Image.fromarray(byte(cape / 20)).save('%s/c%02d.png' % (d, f), optimize=True)
    for n in os.listdir(d):
        total += os.path.getsize(os.path.join(d, n))
    epoch = int((c - dt.datetime(1970, 1, 1)).total_seconds())
    json.dump({'dir': name, 't0': epoch, 'n': HOURS + 1, 'w': 720, 'h': 361,
               'res': 0.5, 'hecho': int(time.time())},
              open(OUT + '/manifest.json', 'w'))
    open(OUT + '/.nojekyll', 'w').close()
    # Cloudflare Pages (tiempo.oteoapp.com): que la app pueda leerlas desde
    # su página (CORS), el índice siempre fresco y las imágenes de cada
    # pasada (que no cambian) guardadas en caché.
    open(OUT + '/_headers', 'w').write(
        '/*\n'
        '  Access-Control-Allow-Origin: *\n'
        '/manifest.json\n'
        '  Cache-Control: no-cache\n'
        '/*/*.png\n'
        '  Cache-Control: public, max-age=86400\n'
        '/*/*.jpg\n'
        '  Cache-Control: public, max-age=86400\n')
    print('hecho en %.0f s, %.1f MB' % (time.time() - t0, total / 1e6))


if __name__ == '__main__':
    sys.exit(main())
