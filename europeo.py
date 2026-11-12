"""La previsión del modelo europeo (ECMWF IFS) para el mapa de Oteo.

Datos abiertos del ECMWF (licencia CC BY 4.0: se pueden usar, también en
una app de pago, citando «ECMWF»). Se usan los de 0,1° (unos 9 km), que
salen con unas horas de retraso, cada 3 horas de previsión; entre medias
se rellenan las horas (el viento, la lluvia y la energía para tormentas,
a partes; la nieve, con lo que diga la hora de dato más cercana).

Se dejan dos juegos de imágenes, con el mismo formato que las del GFS
(ver hacer.py), para el mundo entero:

  datos/<pasada>/...     a medio grado (720 × 361): de lejos
  datos/<pasada>/f/...   a 0,2° (1800 × 901): al acercarse

y en manifest.json, «fino»: {dir, w, h, res} con el segundo juego.
"""
import datetime as dt
import json
import os
import shutil
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pygrib
from PIL import Image

import hacer

BASE = 'https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com'
HOURS = hacer.HOURS
STEPS = list(range(0, HOURS + 1, 3))
WANT = ('10u', '10v', 'tprate', 'ptype', 'mucape', '2t')


class Missing(Exception):
    pass


def get(url, rng=None, tries=8):
    # El servidor pide calma a veces (503): se espera y se reintenta. Si
    # el fichero no está (404/403), no se insiste.
    for k in range(tries):
        try:
            req = urllib.request.Request(url)
            if rng:
                req.add_header('Range', 'bytes=%d-%d' % rng)
            with urllib.request.urlopen(req, timeout=90) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                raise Missing(url)
            if k == tries - 1:
                raise
        except Exception:
            if k == tries - 1:
                raise
        time.sleep(min(30, 2 ** k))


def path(c, s, ext):
    return '%s/%s/%02dz/ifs/0p1/oper/%s0000-%dh-oper-fc-sfc.%s' % (
        BASE, c.strftime('%Y%m%d'), c.hour, c.strftime('%Y%m%d%H'), s, ext)


def cycle():
    """La última pasada que ya llega a las HOURS horas."""
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    c = now.replace(hour=now.hour // 6 * 6, minute=0, second=0, microsecond=0)
    for _ in range(8):
        try:
            get(path(c, HOURS, 'index'))
            return c
        except Missing:
            c -= dt.timedelta(hours=6)
    raise RuntimeError('no hay ninguna pasada reciente del europeo')


def down(a, f):
    """Del 0,1° (1801 × 3600, longitud desde -180) a un paso f veces mayor
    (f = 2: 0,2°; f = 5: 0,5°), con las columnas desde la longitud 0. Cada
    punto, la media de los de alrededor (no uno suelto: así no salen
    puntitos)."""
    a = np.roll(np.asarray(a, dtype=np.float32), -1800, axis=1)
    w = np.full(f, 1.0) if f % 2 else np.array([0.5] + [1.0] * (f - 1) + [0.5])
    w = w / w.sum()
    r = len(w) // 2
    # Por columnas (dando la vuelta al mundo) y luego por filas.
    b = sum(np.roll(a, r - i, axis=1) * w[i] for i in range(len(w)))
    p = np.concatenate([np.repeat(b[:1], r, 0), b, np.repeat(b[-1:], r, 0)])
    b = sum(p[i:i + b.shape[0]] * w[i] for i in range(len(w)))
    return b[::f, ::f]


def step(c, s):
    """De un paso: los campos que hacen falta, ya a 0,2° y a 0,5°."""
    rows = []
    for l in get(path(c, s, 'index')).decode().split('\n'):
        try:
            rows.append(json.loads(l))
        except ValueError:
            pass
    out = {}
    for r in rows:
        p = r.get('param')
        if p in WANT and p not in out:
            g = pygrib.fromstring(get(path(c, s, 'grib2'),
                                      (r['_offset'], r['_offset'] + r['_length'] - 1)))
            v = np.asarray(g.values, dtype=np.float32)
            if p == 'ptype':
                # Qué cae: 5 nieve, 6 nieve húmeda, 7 mezcla. Se guarda la
                # parte de nieve (0 a 1).
                v = np.where((v == 5) | (v == 6), 1.0, np.where(v == 7, 0.5, 0.0))
            out[p] = (down(v, 2), down(v, 5))
    return s, out


def write(d, f, u, v, rain, snow, cape, temp):
    """Las cuatro imágenes de una hora, como las del GFS (ver hacer.py)."""
    byte = hacer.byte
    zero = np.zeros_like(u)
    # Lo que no llega a una décima de mm por hora, nada (la app no lo pinta
    # y así las imágenes pesan bastante menos).
    rain = np.where(rain < 0.05, 0, rain)
    snow = np.where(snow < 0.05, 0, snow)
    Image.fromarray(np.dstack([byte((u + 50) * 2.55), byte((v + 50) * 2.55),
                               byte(zero)]), 'RGB').save(
        '%s/w%02d.jpg' % (d, f), quality=88, subsampling=0)
    Image.fromarray(byte(np.sqrt(np.maximum(rain, 0)) * 40)).save(
        '%s/r%02d.png' % (d, f), optimize=True)
    Image.fromarray(byte(np.sqrt(np.maximum(snow, 0)) * 40)).save(
        '%s/s%02d.png' % (d, f), optimize=True)
    Image.fromarray(byte(cape / 20)).save('%s/c%02d.png' % (d, f), optimize=True)
    # La temperatura a 2 m (de kelvin a °C), como las de lo que ha pasado.
    Image.fromarray(byte((temp - 273.15 + 60) * 2)).save(
        '%s/t%02d.png' % (d, f), optimize=True)


def main():
    t0 = time.time()
    c = cycle()
    name = c.strftime('%Y%m%d%H')
    print('europeo: pasada', name, flush=True)
    with ThreadPoolExecutor(2) as ex:
        got = dict(ex.map(lambda s: step(c, s), STEPS))
    # Lo que no traiga un paso (la lluvia del primero, por ejemplo), del
    # paso más cercano que sí lo tenga.
    for p in WANT:
        have = [s for s in STEPS if p in got[s]]
        if not have:
            raise RuntimeError('falta %s' % p)
        for s in STEPS:
            if p not in got[s]:
                got[s][p] = got[min(have, key=lambda h: abs(h - s))][p]
    shutil.rmtree(hacer.OUT, ignore_errors=True)
    d = os.path.join(hacer.OUT, name)
    fine = os.path.join(d, 'f')
    os.makedirs(fine)
    for f in range(HOURS + 1):
        s0 = f // 3 * 3
        s1 = min(s0 + 3, STEPS[-1])
        t = (f - s0) / 3 if s1 > s0 else 0.0
        near = s0 if t < 0.5 else s1
        for k, out in ((0, fine), (1, d)):
            def at(p):
                return got[s0][p][k] * (1 - t) + got[s1][p][k] * t
            rain = at('tprate') * 3600  # mm/h
            write(out, f, at('10u'), at('10v'), rain,
                  rain * got[near]['ptype'][k], at('mucape'), at('2t'))
    total = sum(os.path.getsize(os.path.join(r, n))
                for r, _, ns in os.walk(d) for n in ns)
    epoch = int((c - dt.datetime(1970, 1, 1)).total_seconds())
    json.dump({'dir': name, 't0': epoch, 'n': HOURS + 1, 'w': 720, 'h': 361,
               'res': 0.5, 'modelo': 'ECMWF', 'capas': 'wrsct',
               'fino': {'dir': name + '/f', 'w': 1800, 'h': 901, 'res': 0.2},
               'hecho': int(time.time())},
              open(hacer.OUT + '/manifest.json', 'w'))
    hacer.finish()
    print('europeo: hecho en %.0f s, %.1f MB' % (time.time() - t0, total / 1e6))
