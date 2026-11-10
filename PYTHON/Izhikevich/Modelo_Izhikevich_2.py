# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 18:59:55 2026
# Modelo de Izhikevich (2003)
# Simple Model of Spiking Neurons
# Simulación de diferentes tipos de neuronas
@author: Rogelio Manuel Higuera González 
TECNM campus Ixtapaluca
División de Ingeniería Electrónica 
"""
import numpy as np
import matplotlib.pyplot as plt
"""
Solo se va a obsrevar el disparo de una neurona y e van observar el 
comportamiento de diferentes neuronas
"""
"""
Modelo Izhikevich

    dv/dt = 0.04v²+5v+140-u+I              (1)
    du/dt = a(bv-u)                        (2)

Con una condición auxiliar de reinicio después de cada disparo:
    
    Si v ≥ 30mV, entoces v ← c y u ← u+d   (3)

Aquí, v y u son variables adimensionales, mientras que a, b, c y d son 
parámetros adimensionales. La notacón de derivada indica la diferenciación 
respecto al tiempo t. La variable v represneta el potencial de membrana de la 
neurona, mientras que u representa una variable de recuperación de membrana. 
Las corrientes sinápticaso o las corrientes continuas (DC) inyectadas se 
introducen mediante la variable I.
"""
Ne = 1 # Una neurona exitadora 
Ni = 0 # Ninguna neurona inhibitoria 
N = Ne + Ni # Número total de neuronas (solo una)
T = 1000 # Tiempo total (ms)
dt = 1.0 # Paso de simulación (ms)
tiempo = np.arange(1, T + 1) # Array de 1000 valores de 1 segundo
# Se crea un diccionario con diferentes neuronas (es un menú)
"""
Todas las células coticales excitadoras se dividen en las siguientes clases:
    
    Neuronas RS (Regular Spiking): las neuronas RS son las más comunes en la 
    corteza cerebral. Cuando se les aplica un estímulo prolongado (un escalón
    de corriente continua inyectada), estas neuronas generan inicialmente 
    algunos potenciales de acción con intervalos cortos entre disparos, pero 
    posteriormente dichos intervalos aumentan.
    
    Neuronas de disparo en ráfagas intrínsecas IB (Intrinsically Bursting): las 
    neuronas IB generan inicialmente una ráfaga característica de potenciales 
    de acción, seguida de disparos individuales repetitivos. Durante la ráfaga 
    inicial, la variable u aumneta progresivamente hasta que finalmente
    modifica la dinámica neuronal, pasando de un comportamiento de disparo en 
    ráfagas a uno de disparos individuales.
    
    Neuronas de disparo en ráfagas de alta frecuencia CH (Chattering): las 
    neuronas CH pueden generar ráfagas características de potenciales de acción 
    muy próximos entre sí. La frecuencia entre ráfagas puede alcanzar valores 
    de hasta 40 Hz. 
    
    Neuronas de disparo rápido FS (Fast Spiking): las neuronas FS pueden
    generar trenes periódicos de potenciales de acción a frecuencias 
    extremadamente elevadas, prácticamente sin presentar adaptación de 
    frecuencia (desaceleración).
    
    Neuronas de disparo de bajo umbral LTS (Low-Threshold Spiking): las 
    neuronas LTS también pueden generar trenes de potenciales de acción de 
    alta frecuencia, pero presentan una adaptación de frecuencia considerable.
    Estas neuronas tienen umbrales de disparo bajos. 
    
    Neuronas talamocorticales TC (Thalamo-Cortical): las neuronas TC presentan 
    dos regímenes de disparo:
        Cuando se encuentran en reposo, con un potencial de membrana cercano a 
        -60mV, y posteriormente son despolarizadas, presentan un comportamiento 
        de disparo tónico. Sin embargo, cuando se aplica un escalón de 
        corriente negativa que hiperpolariza la membrana hasta aproximadamente 
        -90mV, las neuronas generan una ráfaga de potenciales de acción por 
        rebote.
    Neuronas resonadoras RZ (Resonator): Las neuronas RZ presentan oscilaciones 
    subumbrales amortiguadas o sostenidas. Estas neuronas presentan resonancia 
    ante entradas rítmicas con una frecuencia apropiada, de manera similar al 
    modelo de resonancia y disparo (resonate and fire). Es importante observar 
    que existe una biestabilidad entre el estado de reposo y el estado de
    disparo repetitivo. La neurona puede cambiar entre estos dos estados 
    mediante estímulos breves aplicados en el momento adecuado.
"""
parametros = {
    "RS": (0.02, 0.20, -65, 8),
    "IB": (0.02, 0.20, -55, 4),
    "CH": (0.02, 0.20, -50, 2),
    "FS": (0.10, 0.20, -65, 2),
    "LTS": (0.02, 0.25, -65, 2),
    "TC": (0.02, 0.25, -65, 0.05),
    "RZ": (0.10, 0.26, -65, 2),
}
# Elección del tipo de neurona (cambian los parametros a, b, c y d)
tipo_neurona = "RZ"  # RS, IB, CH, FS, LTS, TC, RZ

"""
Parámetro a: describe la escala temporal de la variable de recuperación u. Los 
valores más pequeños producen una recuperación más lenta. Un valor típico es 
a = 0.02.

Parámetro b: describe la snsibilidad de la variable de recuperación u ante las
fluctuaciones subumbrales del potencial de membrana v. Los valores mayores 
acoplan más fuertemente v y u, lo que puede producir oscilaciones subumbrales 
y dinámicas de disparo de bajo umbral. Un valor típico es b = 0.2. El caso 
b < a corresponde a una bifurcación de tipo sanddle-node, mientras que b > a 
corresponde a una bifurcación de Adronov-Hopf del estado de reposo.

Parámetro c: describe el valor de reinicio del potencial de membrana v después 
de un disparo, causado por las conductacias rápidas de potasio (K+) de alto
umbral. Un valor típico es c = -65mV.

Parámetro d: describe el reinicio de la variable de recuperación u después de
un disparo, causado por las conductacias lentas de sodio (Na+) y potasio (K+) 
de alto umbral. Un valor típico es d = 2. 
"""
# La función .upper() convierte una cadena de texto a mayúsculas
a0, b0, c0, d0 = parametros[tipo_neurona.upper()]

# Parámetros como arreglos
# np.full() de NumPy crea un arreglo de N elementos, todo con el valor a0 
a = np.full(N, a0)
b = np.full(N, b0)
c = np.full(N, c0)
d = np.full(N, d0)

# Condiciones iniciales
v = np.full(N, -65.0)
u = b * v

# Preasignación de memoria
# Crea un array de 0 con dimension 1000, 1. Sirve para almacenar el potencial
# de membrana v de cada neurona durante toda la simulación
vplot = np.zeros((T, N))
uplot = np.zeros((T, N))
Iplot = np.zeros((T, N))

disparos = []

# Simulación
for t in range(T):

    # Corriente de entrada: escalón
    if 100 <= t + 1 <= 900:
        I = np.full(N, 10.0)
    else:
        I = np.zeros(N)

    Iplot[t, :] = I

    # Detección de disparos
    neuronas_activadas = np.flatnonzero(v >= 30)

    if neuronas_activadas.size > 0:
        for i in neuronas_activadas:
            disparos.append((t + 1, i + 1))

        # Reinicio neuronal
        v[neuronas_activadas] = c[neuronas_activadas]
        u[neuronas_activadas] += d[neuronas_activadas]

    # Integración de Euler: dos pasos de 0.5 ms
    v += 0.5 * dt * (0.04*v**2 + 5*v + 140 - u + I)
    v += 0.5 * dt * (0.04*v**2 + 5*v + 140 - u + I)

    # Actualización de recuperación
    u += dt * a * (b*v - u)

    # Almacenamiento de resultados
    vplot[t, :] = v
    uplot[t, :] = u

# Visualización
fig, ax = plt.subplots(4, 1, figsize=(11, 9),
                       sharex=True)

ax[0].plot(tiempo, vplot[:, 0], color="blue")
ax[0].set_ylabel("v (mV)")
ax[0].set_title(f"Potencial de membrana - {tipo_neurona}")
ax[0].grid(True)

ax[1].plot(tiempo, uplot[:, 0], color="red")
ax[1].set_ylabel("u")
ax[1].set_title("Variable de recuperación")
ax[1].grid(True)

ax[2].plot(tiempo, Iplot[:, 0], color="black")
ax[2].set_ylabel("I")
ax[2].set_title("Corriente de entrada")
ax[2].grid(True)

if disparos:
    disparos = np.array(disparos)
    ax[3].scatter(disparos[:, 0], disparos[:, 1], s=12)

ax[3].set_xlabel("Tiempo (ms)")
ax[3].set_ylabel("Neurona")
ax[3].set_title("Instantes de disparo")
ax[3].set_ylim(0.5, N + 0.5)
ax[3].grid(True)

ax[3].set_xlim(0, T)

plt.tight_layout()
plt.show()

# Resultados
n_spikes = len(disparos)

print(f"Tipo de neurona: {tipo_neurona}")
print(f"Parámetros: a={a0}, b={b0}, c={c0}, d={d0}")
print(f"Disparos registrados: {n_spikes}")
print(f"Frecuencia media: {n_spikes/(N*T/1000):.2f} Hz")
