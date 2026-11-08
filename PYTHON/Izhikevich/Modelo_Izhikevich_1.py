# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 10:06:59 2026
Modelo de Izhikevich
Paper: Simple Model of Spiking Neurons (2003)
Red de 800 neuronas excitadoras y 200 inhibidoras
@author: Rogelio Manuel Higuera González 
TECNM campus Ixtapaluca
División de Ingeniería Electrónica 
"""
import numpy as np
import matplotlib.pyplot as plt

# Número de neuronas
Ne = 800 #Neuronas exitadoras 
Ni = 200 #Neuronas inhibidoras 
N = Ne + Ni #Total de neuronas (en este caso 1000)

# Parámetros neuronales
# La función rand() crea valores a leatorios entre 0 y 1
re = np.random.rand(Ne) # Crea un array de 800 valores aleatorios entre 0 y 1
ri = np.random.rand(Ni) # Crea un array de 200 valores aleatorios entre 0 y 1
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
"""
Parámetro a: describe la escala temporal de la variable de recuperación u. Los 
valores más pequeños producen una recuperación más lenta. Un valor típico es 
a = 0.02.
"""
# La función concatenate junta dos array, la función ones crea un array de 1
a = np.concatenate((0.02*np.ones(Ne), 0.02+0.08*ri)) # 800 valores de 0.02 y 
                                                     # 200 valores aleatorios 
"""
Parámetro b: describe la snsibilidad de la variable de recuperación u ante las
fluctuaciones subumbrales del potencial de membrana v. Los valores mayores 
acoplan más fuertemente v y u, lo que puede producir oscilaciones subumbrales 
y dinámicas de disparo de bajo umbral. Un valor típico es b = 0.2. El caso 
b < a corresponde a una bifurcación de tipo sanddle-node, mientras que b > a 
corresponde a una bifurcación de Adronov-Hopf del estado de reposo.
"""
b = np.concatenate((0.2*np.ones(Ne), 0.25-0.05*ri)) # 800 valores de 0.2 y
                                                    # 200 valores aleatorios
"""
Parámetro c: describe el valor de reinicio del potencial de membrana v después 
de un disparo, causado por las conductacias rápidas de potasio (K+) de alto
umbral. Un valor típico es c = -65mV.
"""
c = np.concatenate((-65+15*re**2, -65*np.ones(Ni))) # 800 valores aleatorios y
                                                    # 200 valores de -65
"""
Parámetro d: describe el reinicio de la variable de recuperación u después de
un disparo, causado por las conductacias lentas de sodio (Na+) y potasio (K+) 
de alto umbral. Un valor típico es d = 2. 
"""
d = np.concatenate((8-6*re**2, 2*np.ones(Ni))) # 800 valores aleatorios y
                                               # 200 valores de 2
"""
Los pesos de las conexiones sinápticas entre las neuronas están definidos por 
la matríz:
    S = (sij)
De esta maera, cuando la neurona j genera un potencial de acción, la variable 
vi de la neurona i cambia instantáneamente en una cantidad sij. 
"""
# La función hstack une arrays horizontalmete 
# La primera parte crea un array 1000x800 con valores aleatorios entre 0 y 0.5
# La segunda parte crea un array 1000x200 don valores aleatorios entre -1 y 0
S = np.hstack((0.5*np.random.rand(N,Ne), -np.random.rand(N,Ni)))
# Condiciones iniciales 
v = -65*np.ones(N) # Se crea un array con 1000 valores de -65
u = b*v # Se crea un array de 1000 de valores aleatorios de la multiplicación
# Instantes de disparo neuronal 
disparos = [] # Crea una lista vacía, que se utiliza para almacenar los
             # instantes en los que las neuronas generan spikes 
# Simulación de 1000 ms
for t in range (1, N+1):
    # Entrada talámica 
    # Crea un array de 1000 valores
    # Los primero 800 valores aleatorios van de 0 a 5
    # Los ultimos 200 valores aleatorios van de 0 a 2
    I = np.concatenate((5*np.random.randn(Ne), 2*np.random.randn(Ni)))
    #Detección de disparos 
    """
    Recordar que si el potencial (v) supera los 30mV se detecta un disparo
    """
    # La función flatnonzero devuelve los índices de los elementos de un arreglo
    # cuyo valor es distinto de cero
    neuronas_activadas = np.flatnonzero(v>=30)
    # Registrar disparos 
    if neuronas_activadas.size>0:
        # La función extend agrega varios elementos a una lista existente
        # i índice de una neurona que disparó
        # for i in disparos recorre todas las neuronas que dispararon en ese
        # instante
        disparos.extend((t, int(i)+1) for i in neuronas_activadas)
        # Reinicio neuronal 
        v[neuronas_activadas] = c[neuronas_activadas]
        u[neuronas_activadas] = u[neuronas_activadas]+d[neuronas_activadas]
        # Corriente sináptica 
        # axis = 1 Indica que la suma se realiza horizontalmete, por filas
        # np.sum() suma las contribuciones sinápticas de las neuronas que si
        #dispararon
        I = I + np.sum(S[:,neuronas_activadas], axis = 1)
    """
    Método de Euler
    El método de Euler aproxima la solución de una ecuación diferencial 
    mediante:
        
        yn+1 = yn + hf(yn)
        
        Donde:
            yn : valor actual de la variable 
            yn+1 : valor calculado en el siguiente paso
            h: tamaño del paso de integración
            f(yn): derivada evaluada en el valor actual 
    
    Para el potencial de membrana:
        
        vn+1 = vn + h(0.04vn²+5vn+140-un+In)
        
    Izhikevich utiliza dos pasos de 0.5ms (por eso el 0.5)
    
        v = v+0.5*(0.04*v**2+5*v+140-u+I)
    
    Para u solo se utiliza un paso de 1ms 
        u = u+a*(b*v-u)
    
    """
    v = v+0.5*(0.04*v**2+5*v+140-u+I)
    v = v+0.5*(0.04*v**2+5*v+140-u+I)
    u = u+a*(b*v-u)

# Gráfica de activación neuronal
# La lista diaparos se convierte a un array con la función array
# reshape permite cambiar la forma o dimensiones de una array 
# con el -1 numpy calcula automáticamente el número de filas necesarias. 
# 2 establece que el arreglo debe tener 2 columnas 
disparos = np.array(disparos).reshape(-1,2)

plt.figure(figsize=(12,6))
if disparos.size > 0:
    plt.plot(disparos[:,0],disparos[:,1],'.',markersize=1)

plt.xlabel('Tiempo (ms)')
plt.ylabel('Índice de neurona')
plt.title('Modelo de Izhikevich - Actividad neuronal')

plt.xlim(0,1000)
plt.ylim(0,1000)
plt.grid(True,alpha=0.2)

plt.show()
        
        
        
    

                                                    
 




