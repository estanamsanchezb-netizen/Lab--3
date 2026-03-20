# Lab--3
## Análisis espectral de la voz 

En esta práctica de laboratorio se trabajaron conceptos fundamentales del procesamiento de señales aplicados al análisis de la voz humana. En primer lugar, se realizaron grabaciones de voz de tres mujeres y tres hombres, asegurando condiciones similares de muestreo. Posteriormente, se aplicó la Transformada de Fourier para representar las señales en el dominio de la frecuencia y analizar su espectro. A partir de este análisis, se calcularon parámetros como el brillo, la intensidad, el jitter y el shimmer. Finalmente, se compararon los resultados entre voces masculinas y femeninas, identificando sus principales diferencias.

## Parte A
Se realizó la grabación de la voz de tres mujeres y tres hombres, quienes pronunciaron la misma frase: “Si hay una cosa que nadie ha podido comprar con dinero, ésa es el movimiento de la cola de un perro”. Cada archivo fue guardado en formato .wav con su respectiva identificación. Posteriormente, las señales fueron importadas a Python para ser graficadas en el dominio del tiempo. Luego, se aplicó la Transformada de Fourier con el fin de analizar el contenido frecuencial de cada señal. Finalmente, se calcularon parámetros como la frecuencia fundamental, la frecuencia media, el brillo y la intensidad.

**Graficado de la señales**

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# mujeres
fs1,signal1=  wavfile.read('/content/Mujer-1.m4a.wav')
fs2,signal2=  wavfile.read('/content/Mujer-2.m4a.wav')
fs3,signal3=  wavfile.read('/content/Mujer-3.m4a.wav')

# Hombres
fs4,signal4=  wavfile.read('/content/Hombre-1.m4a_1.wav')
fs5,signal5= wavfile.read('/content/Hombre-2.m4a.wav')
fs6,signal6= wavfile.read('/content/Hombre-3.m4a.wav')


duracion = len(signal1)/fs1

duracion = len(signal2)/fs1

duracion = len(signal3)/fs1



duracion = len(signal4)/fs1

duracion = len(signal5)/fs1

duracion = len(signal6)/fs1


Tiempo1 = np.arange(len(signal1)) / fs1
Tiempo2 = np.arange(len(signal2)) / fs2
Tiempo3 = np.arange(len(signal3)) / fs3
Tiempo4 = np.arange(len(signal4)) / fs4
Tiempo5 = np.arange(len(signal5)) / fs5
Tiempo6 = np.arange(len(signal6)) / fs6
fig, axs = plt.subplots(6,1, figsize=(14,10),sharex=False)

print(len(Tiempo3),len(signal3))

# graficas
axs[0].plot(Tiempo1, signal1, color='hotpink')
axs[0].set_title("Mujer 1")
axs[0].set_ylabel('Bits')
axs[0].set_xlabel('Tiempo (s)')

axs[1].plot(Tiempo2, signal2,color='deeppink')
axs[1].set_title("Mujer 2")
axs[1].set_ylabel('Bits')
axs[1].set_xlabel('Tiempo (s)')

axs[2].plot(Tiempo3, signal3, color='mediumvioletred')
axs[2].set_title("Mujer 3")
axs[2].set_ylabel('Bits')
axs[2].set_xlabel('Tiempo (s)')

axs[3].plot(Tiempo4, signal4, color='mediumpurple')
axs[3].set_title("Hombre 1")
axs[3].set_ylabel('Bits')
axs[3].set_xlabel('Tiempo (s)')

axs[4].plot(Tiempo5, signal5, color='mediumslateblue')
axs[4].set_title("Hombre 2")
axs[4].set_ylabel('Bits')
axs[4].set_xlabel('Tiempo (s)')

axs[5].plot(Tiempo6, signal6, color='rebeccapurple')
axs[5].set_title("Hombre 3")
axs[5].set_ylabel('Bits')
axs[5].set_xlabel('Tiempo (s)')

plt.tight_layout()
plt.show()
```
<img width="953" height="616" alt="image" src="https://github.com/user-attachments/assets/91c5f562-bfea-42d9-b600-a0bee891e0a6" />

**Aplicación de la Transformada de Fourier**
```python
import numpy as np
import matplotlib.pyplot as plt

seniales = [
    ("Mujer 1", signal1, fs1),
    ("Mujer 2", signal2, fs2),
    ("Mujer 3", signal3, fs3),
    ("Hombre 1", signal4, fs4),
    ("Hombre 2", signal5, fs5),
    ("Hombre 3", signal6, fs6)
]

# Colores
color_mujer = "deeppink"
color_hombre = "darkviolet"

# --- Graficar FFT de cada señal ---
plt.figure(figsize=(12, 20))

for i, (titulo, señal, fs) in enumerate(seniales, 1):

    N = len(señal)
    freqs = np.fft.rfftfreq(N, 1/fs)
    espectro = np.abs(np.fft.rfft(señal))

    # Elegir color según el título
    if "Mujer" in titulo:
        color = color_mujer
    else:
        color = color_hombre

    plt.subplot(6, 1, i)
    plt.semilogx(freqs, espectro, color=color, linewidth=1.8)
    plt.title(titulo)
    plt.ylabel('Amplitud')
    plt.grid(True)

    # Limitar el eje X desde 10 Hz
    plt.xlim(left=10)

plt.xlabel('Frecuencia (Hz)')
plt.tight_layout()
plt.show()
```
<img width="780" height="616" alt="image" src="https://github.com/user-attachments/assets/64ebffb5-1454-4e14-b980-554866f8832e" />

<img width="829" height="620" alt="image" src="https://github.com/user-attachments/assets/b3422fbb-9ce3-4371-b5d1-0b0a4f12387f" />

**Características de la señal**

```python
import numpy as np

def caracteristicas(señal, fs):
    # Si es estéreo, convertir a mono
    if señal.ndim > 1:
        señal = señal.mean(axis=1)

    # FFT solo en frecuencias positivas
    N = len(señal)
    freqs = np.fft.rfftfreq(N, 1/fs)
    espectro = np.abs(np.fft.rfft(señal))

    # a) Frecuencia fundamental = primer pico distinto de DC (descartamos muy bajas)
    espectro[0] = 0  # quitar DC
    idx = np.argmax(espectro)  # máximo pico del espectro
    Frecuencia = freqs[idx]

    # b) Frecuencia media (centroide espectral)
    fmedia = np.sum(freqs * espectro) / np.sum(espectro)

    # c) Brillo (igual que centroide espectral aquí)
    brillo = fmedia

    # d) Intensidad (energía)
    energia = np.sum(señal.astype(float)**2)

    return Frecuencia, fmedia, brillo, energia


# --- Lista de señales ya cargadas ---
seniales = [
    ("Mujer 1", signal1, fs1),
    ("Mujer 2", signal2, fs2),
    ("Mujer 3", signal3, fs3),
    ("Hombre 1", signal4, fs4),
    ("Hombre 2", signal5, fs5),
    ("Hombre 3", signal6, fs6)
]

# --- Calcular y mostrar resultados ---
for nombre, s, fs in seniales:
    Frecuencia, fmedia, brillo, energia = caracteristicas(s, fs)
    print(f"{nombre}:")
    print(f"  Frecuencia fundamental: {Frecuencia:.2f} Hz")
    print(f"  Frecuencia media:       {fmedia:.2f} Hz")
    print(f"  Brillo:                 {brillo:.2f} Hz")
    print(f"  Intensidad (energía):   {energia:.2e}\n")
```
<img width="344" height="523" alt="image" src="https://github.com/user-attachments/assets/fd0dc519-1fde-4654-aa85-0f59a1024ea2" />

<br>
<img width="213" height="895" alt="image" src="https://github.com/user-attachments/assets/4e54b87c-7fc1-413d-9795-0ff728e4b3f7" />


## Parte B
## Parte C 
1. ¿Qué diferencias se observan en la frecuencia fundamental?
 
Se observa que en general, las voces femeninas presentan una frecuencia fundamental mayor que las voces masculinas. las mujeres registraron valores alrededor de 236–238 Hz, mientras que los hombres presentan valores más bajos como 203 Hz y 95 Hz. Esto es consistente con la teoría, ya que las cuerdas vocales de las mujeres suelen ser más cortas y vibran a mayor frecuencia. Sin embargo, se presentan algunos valores atípicos en ambas poblaciones (como 512 Hz en mujeres y 470 Hz en hombres), los cuales pueden deberse a variaciones en la pronunciación o errores en la medición.

2. ¿Qué otras diferencias notan en términos de brillo, media o intensidad?
   
En cuanto al brillo y la frecuencia media, no se observa una diferencia tan marcada entre hombres y mujeres, ya que los valores son relativamente similares y en algunos casos se superponen. Por ejemplo, ambos grupos presentan valores cercanos a los 3000–4100 Hz.
Sin embargo, en la intensidad (energía) sí se nota que algunas voces masculinas alcanzan valores más altos (como 7.31e+12), en comparación con las femeninas, que en general presentan valores menores. Esto puede deberse a una mayor potencia vocal o volumen al momento de la grabación.

3. Redactar conclusiones sobre el comportamiento de la voz en hombres y mujeres a partir de los análisis realizados.
4. Discuta la importancia clínica del jitter y shimmer en el análisis de la voz. 
