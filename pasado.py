"""Lo que ha pasado: la lluvia caída y la temperatura que ha hecho.

De cada pasada del modelo GFS (NOAA, dominio público) se toman sus 6
primeras horas, que son casi lo observado (el modelo arranca de las
medidas de estaciones y satélites): la lluvia caída en esas 6 horas y la
máxima y la mínima de temperatura. Con cuatro pasadas al día salen los
días enteros. Se guarda un mes hacia atrás.

  datos/obs/b/<pasada>-{r,x,n}.png   cada bloque de 6 h (para rehacer los días)
  datos/obs/d/<AAAAMMDD>-{r,x,n}.png cada día (UTC)
  datos/obs/t/<N>-{r,x,n}.png        los últimos N días juntos (1, 3, 7, 14, 21, 30)
  datos/obs/manifest.json            {dias: [...], totales: [...], w, h, res, hecho}

  r: lluvia caída (gris): (x / 8)^2 mm  (hasta unos 1.000 mm)
  x, n: media de las máximas / mínimas (gris): x / 2 - 60 °C

Mismas filas y columnas que la previsión: de norte (90°) a sur, desde la
longitud 0 hacia el este, a medio grado (720 × 361).

Los bloques ya hechos se recuperan de la publicación anterior (la carpeta
datos/obs que deja el paso «Recuperar» del flujo); los que falten del
último mes se piden de nuevo (NOAA guarda las pasadas antiguas).
"""
import datetime as dt
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

import hacer

OUT = os.path.join(hacer.OUT, 'obs')
DAYS = 30
TOTALS = [1, 3, 7, 14, 21, 30]
# Como mucho tantos bloques nuevos por vez (la primera, el mes entero).
MAX_NEW = 130
WANT = {
    'APCP:surface': 'r',
    'TMAX:2 m above ground': 'x',
    'TMIN:2 m above ground': 'n',
}


def block(c):
    """Las 6 primeras horas de la pasada c: lluvia (mm), máx. y mín. (°C)."""
    url = hacer.BASE + '/' + hacer.key(c, 6)
    rows = [l.split(':') for l in
            hacer.get(url + '.idx').decode().strip().split('\n')]
    out = {}
    for i, r in enumerate(rows):
        name = WANT.get(r[3] + ':' + r[4])
        # De la lluvia hay dos iguales: vale la primera.
        if not name or name in out or not r[5].startswith('0-6 hour'):
            continue
        end = int(rows[i + 1][1]) - 1 if i + 1 < len(rows) else ''
        out[name] = hacer.half(import_grib(hacer.get(url, (int(r[1]), end))))
    if set(out) != set(WANT.values()):
        raise ValueError('faltan datos en %s' % c)
    out['x'] = out['x'] - 273.15
    out['n'] = out['n'] - 273.15
    return out


def import_grib(data):
    import pygrib
    return pygrib.fromstring(data).values


def rain_png(mm):
    return Image.fromarray(hacer.byte(np.sqrt(np.maximum(mm, 0)) * 8))


def temp_png(c):
    return Image.fromarray(hacer.byte((np.clip(c, -60, 67) + 60) * 2))


def read(path):
    return np.asarray(Image.open(path), dtype=np.float32)


def main():
    t0 = time.time()
    bdir, ddir, tdir = (os.path.join(OUT, x) for x in 'bdt')
    for d in (bdir, ddir, tdir):
        os.makedirs(d, exist_ok=True)
    # Las pasadas del último mes cuyas 6 primeras horas ya han pasado.
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    last = now.replace(hour=now.hour // 6 * 6, minute=0, second=0,
                       microsecond=0) - dt.timedelta(hours=6)
    first = (last - dt.timedelta(days=DAYS + 1)).replace(hour=0)
    cycles = []
    c = first
    while c <= last:
        cycles.append(c)
        c += dt.timedelta(hours=6)
    have = {n.split('-')[0] for n in os.listdir(bdir) if n.endswith('-r.png')}
    missing = [c for c in cycles if c.strftime('%Y%m%d%H') not in have]
    # Lo más reciente primero: si no da tiempo a todo, que esté lo de ayer.
    missing = sorted(missing, reverse=True)[:MAX_NEW]
    print('bloques que faltan:', len(missing), flush=True)

    def one(c):
        try:
            return c, block(c)
        except Exception as e:
            print('sin', c, e, flush=True)
            return c, None

    with ThreadPoolExecutor(8) as ex:
        for c, b in ex.map(one, missing):
            if b is None:
                continue
            n = c.strftime('%Y%m%d%H')
            rain_png(b['r']).save('%s/%s-r.png' % (bdir, n), optimize=True)
            temp_png(b['x']).save('%s/%s-x.png' % (bdir, n), optimize=True)
            temp_png(b['n']).save('%s/%s-n.png' % (bdir, n), optimize=True)
    # Fuera lo de hace más de un mes.
    keep = {c.strftime('%Y%m%d%H') for c in cycles}
    for n in os.listdir(bdir):
        if n.split('-')[0] not in keep:
            os.remove(os.path.join(bdir, n))
    for n in os.listdir(ddir):
        os.remove(os.path.join(ddir, n))
    for n in os.listdir(tdir):
        os.remove(os.path.join(tdir, n))
    # Los días enteros (sus cuatro bloques): lluvia sumada, la máxima más
    # alta y la mínima más baja.
    have = sorted({n.split('-')[0] for n in os.listdir(bdir)
                   if n.endswith('-r.png')})
    by_day = {}
    for n in have:
        by_day.setdefault(n[:8], []).append(n)
    days = []
    rain_d, max_d, min_d = {}, {}, {}
    for day, blocks in sorted(by_day.items()):
        if len(blocks) < 4:
            continue
        # Lluvia: se vuelve a mm (cada bloque se guardó redondeado).
        r = sum((read('%s/%s-r.png' % (bdir, b)) / 8) ** 2 for b in blocks)
        x = np.max([read('%s/%s-x.png' % (bdir, b)) for b in blocks], axis=0) / 2 - 60
        n = np.min([read('%s/%s-n.png' % (bdir, b)) for b in blocks], axis=0) / 2 - 60
        rain_png(r).save('%s/%s-r.png' % (ddir, day), optimize=True)
        temp_png(x).save('%s/%s-x.png' % (ddir, day), optimize=True)
        temp_png(n).save('%s/%s-n.png' % (ddir, day), optimize=True)
        rain_d[day], max_d[day], min_d[day] = r, x, n
        days.append(day)
    days = days[-DAYS:]
    # Los últimos N días juntos: la lluvia sumada; las temperaturas, la
    # media de las máximas y de las mínimas.
    totals = []
    for k in TOTALS:
        if len(days) < k:
            continue
        sel = days[-k:]
        rain_png(sum(rain_d[d] for d in sel)).save(
            '%s/%d-r.png' % (tdir, k), optimize=True)
        temp_png(np.mean([max_d[d] for d in sel], axis=0)).save(
            '%s/%d-x.png' % (tdir, k), optimize=True)
        temp_png(np.mean([min_d[d] for d in sel], axis=0)).save(
            '%s/%d-n.png' % (tdir, k), optimize=True)
        totals.append(k)
    json.dump({'dias': days, 'totales': totals, 'w': 720, 'h': 361,
               'res': 0.5, 'hecho': int(time.time())},
              open(os.path.join(OUT, 'manifest.json'), 'w'))
    print('pasado: %d días, totales %s, en %.0f s'
          % (len(days), totals, time.time() - t0))


if __name__ == '__main__':
    sys.exit(main())
